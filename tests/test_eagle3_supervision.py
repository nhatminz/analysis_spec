import sys
import unittest
from pathlib import Path

from helper.eagle3_supervision import supervision_bounds

try:
    import torch
except ImportError:
    torch = None


class SupervisionBoundsTests(unittest.TestCase):
    def test_prompt_boundary_and_last_token(self):
        self.assertEqual(supervision_bounds(6, 2), (1, 5))
        self.assertEqual(supervision_bounds(2, 2), (1, 1))
        self.assertEqual(supervision_bounds(0, 0), (0, 0))

    def test_mask_indices_match_specforge_right_shift_without_torch(self):
        # TargetHead.preprocess applies padding(loss_mask, left=False):
        # raw positions 2..5 become supervision rows 1..4.
        vendor = (
            Path(__file__).resolve().parents[1] / "third_party" / "SpecForge" /
            "specforge" / "modeling" / "target" / "target_head.py"
        ).read_text()
        self.assertIn("target = padding(target, left=False)", vendor)
        self.assertIn("input_ids = padding(input_ids, left=False)", vendor)
        self.assertIn("loss_mask = padding(loss_mask, left=False)", vendor)
        raw_mask = [0, 0, 1, 1, 1, 1]
        specforge_shifted = raw_mask[1:] + [0]
        start, stop = supervision_bounds(len(raw_mask), prompt_length=2)
        ours = [int(start <= index < stop) for index in range(len(raw_mask))]
        self.assertEqual(ours, specforge_shifted)


@unittest.skipUnless(torch is not None, "PyTorch unavailable in this environment")
class SpecForgeAlignmentTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "third_party" / "SpecForge"))

    def test_tokens_final_teacher_and_mask_match_target_head_preprocess(self):
        from helper.eagle3_supervision import aligned_eagle3_row
        from specforge.modeling.target.target_head import TargetHead

        original_ids = torch.tensor([[10, 11, 12, 13, 14, 15]])
        shifted_decoder_ids = torch.tensor([11, 12, 13, 14, 15, 16])
        # The final two positions emulate target hidden gathered after a
        # verification segment, not just a contiguous prefill tensor.
        aux = torch.cat((torch.arange(12).reshape(4, 3), torch.arange(6).reshape(2, 3) + 100))
        final_hidden = torch.cat((torch.arange(8).reshape(4, 2), torch.arange(4).reshape(2, 2) + 200))
        raw_loss_mask = torch.tensor([[0, 0, 1, 1, 1, 1]])
        reference_ids, reference_hidden, reference_mask = TargetHead.preprocess(
            None, original_ids, final_hidden.unsqueeze(0), raw_loss_mask
        )
        ids, features, teacher, mask = aligned_eagle3_row(
            aux, final_hidden, shifted_decoder_ids, prompt_length=2
        )
        self.assertTrue(torch.equal(ids, shifted_decoder_ids))
        self.assertTrue(torch.equal(ids[:-1], reference_ids[0, :-1]))
        self.assertTrue(torch.equal(features, aux))
        self.assertTrue(torch.equal(teacher, reference_hidden[0]))
        self.assertTrue(torch.equal(mask, reference_mask[0].float()))
        self.assertEqual(mask[:, 0].tolist(), [0, 1, 1, 1, 1, 0])

    def test_left_padding_removed_and_budget_counts_only_valid_teachers(self):
        from helper.eagle3_supervision import aligned_eagle3_row, count_eagle3_supervision
        from specforge.modeling.target.target_head import TargetHead

        padded_ids = torch.tensor([0, 0, 10, 11, 12, 13])
        original = padded_ids[2:].unsqueeze(0)
        hidden = torch.tensor([[10.0], [11.0], [12.0], [13.0]])
        features = torch.arange(8.0).reshape(4, 2)
        shifted = torch.tensor([11, 12, 13, 14])
        _, ref_hidden, ref_mask = TargetHead.preprocess(
            None, original, hidden.unsqueeze(0), torch.tensor([[0, 0, 1, 1]])
        )
        _, _, teacher, mask = aligned_eagle3_row(features, hidden, shifted, 2, token_budget=1)
        self.assertTrue(torch.equal(teacher, ref_hidden[0]))
        self.assertEqual(mask[:, 0].tolist(), [0, 1, 0, 0])
        self.assertEqual(ref_mask[0, :, 0].tolist(), [0, 1, 1, 0])
        self.assertEqual(count_eagle3_supervision(
            [shifted], [hidden], torch.tensor([[0, 0, 1, 1]]), 1
        ), 2)
        hidden[2] = float('nan')
        _, _, teacher, mask = aligned_eagle3_row(features, hidden, shifted, 2)
        self.assertEqual(mask[:, 0].tolist(), [0, 0, 1, 0])
        self.assertTrue(torch.isfinite(teacher).all())


if __name__ == "__main__":
    unittest.main()
