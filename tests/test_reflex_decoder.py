"""Tiny synthetic CUDA rollout; no downloaded/pretrained model or full run."""
import unittest
from types import SimpleNamespace
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA unavailable')
class DecoderParityTests(unittest.TestCase):
    def test_zero_lr_tensor_verifier_matches_off_and_preserves_alignment(self):
        from transformers import DynamicCache
        from helper.specualtive_generate import speculative_generate_in_prompt_batches
        from helper.eagle3_supervision import aligned_eagle3_row
        from policy_lag_analysis import state_digest

        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(24, 8)
                self.calls = 0

            def forward(self, input_ids, past_key_values=None, **kwargs):
                self.calls += 1
                hidden = self.embedding(input_ids)
                cache = past_key_values if past_key_values is not None else DynamicCache()
                cache.update(hidden.unsqueeze(1), hidden.unsqueeze(1), 0)
                return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache,
                                       hidden_states=(hidden, hidden, hidden, hidden))

        class Target(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()
                self.lm_head = torch.nn.Linear(8, 24, bias=False)
                self._fastgrpo_eagle3_capture_layers = (0, 1, 2)

            @property
            def dtype(self):
                return next(self.parameters()).dtype

            @property
            def device(self):
                return next(self.parameters()).device

        class Draft(torch.nn.Module):
            is_eagle3_specforge = True
            compact_vocab_size = 17

            def __init__(self):
                super().__init__()
                self.target_model = Target()
                self.draft_model = torch.nn.Linear(8, 8, bias=False)
                self.embedding = torch.nn.Embedding(24, 8)
                self.head = torch.nn.Linear(8, 17, bias=False)

            @property
            def dtype(self):
                return next(self.parameters()).dtype

            @property
            def device(self):
                return next(self.parameters()).device

            def compact_to_target_ids(self, *, device):
                return torch.arange(17, device=device) + 2

            def compute_compact_logits(self, hidden):
                return self.head(hidden)

            def forward(self, hidden_states, input_ids, past_key_values=None, **kwargs):
                hidden = self.draft_model(hidden_states[..., :8]) + self.embedding(input_ids)
                key = hidden.unsqueeze(1)
                if past_key_values:
                    key = torch.cat((past_key_values[0][0], key), dim=-2)
                return dict(hidden_states=hidden, next_feature_states=hidden,
                            past_key_values=[[key, key.clone()]])

        torch.manual_seed(42)
        model = Draft().to(device='cuda', dtype=torch.bfloat16).eval()
        original = state_digest(model.state_dict())
        prompt = torch.tensor([[0, 0, 4, 5], [6, 7, 8, 9]], device='cuda')
        mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]], device='cuda')
        args = dict(prompt_batch_size=2, input_ids=prompt, attention_mask=mask, model=model,
                    tokenizer=SimpleNamespace(eos_token_id=23), do_sample=True,
                    repeated_generate_nums=2, max_length=24, temperature=1.0, top_p=0.95,
                    verification_capacity=24, max_verification_num=24, max_draft_k=3,
                    max_draft_token_length=3, min_draft_token_length=2,
                    return_all_draft_input=True, statistical_time=False)
        initial_rng = torch.cuda.get_rng_state()
        results = []
        for mode, learning_rate in (('off', 0.0), ('active', 0.0), ('active', 0.0),
                                    ('active', 0.05), ('active', 0.05)):
            model.target_model.model.calls = 0
            torch.cuda.set_rng_state(initial_rng)
            with torch.inference_mode(), mock.patch('torch.cuda.synchronize', side_effect=AssertionError('round sync')):
                result = speculative_generate_in_prompt_batches(**args, reflex_mode=mode,
                    reflex_lr=learning_rate, reflex_backend='triton', reflex_update_stream=True,
                    reflex_time_updates=mode == 'active')
            self.assertEqual(model.target_model.model.calls, 1 + max(result['response_verification_rounds']))
            self.assertEqual(state_digest(model.state_dict()), original)
            self.assertEqual(sum(result['response_accepted_length_sum']), result['total_acc_length'])
            self.assertEqual(sum(result['response_verification_rounds']), result['total_decoded_token_num'])
            if mode == 'active':
                self.assertTrue(result['reflex_state_initialized_zero'])
                self.assertTrue(result['reflex_state_cleared'])
                self.assertGreater(result['reflex_update_count'], 0)
                self.assertGreater(result['reflex_update_gpu_ms'], 0)
                self.assertGreaterEqual(result['reflex_update_exposed_ms'], 0)
            for index, (ids, features, teacher) in enumerate(zip(result['all_draft_input_ids'],
                    result['all_draft_input_states'], result['all_target_hidden_states'])):
                ids, unchanged, aligned, loss_mask = aligned_eagle3_row(features, teacher, ids,
                                                                       int(mask[index // 2].sum()))
                valid = loss_mask[:, 0].bool()
                self.assertTrue(torch.equal(unchanged, features))
                torch.testing.assert_close(aligned[valid], model.target_model.model.embedding(ids[valid]))
            results.append(result)
        for result in results[1:3]:
            for key in ('generated_token_ids', 'response_accepted_length_sum', 'response_verification_rounds'):
                self.assertEqual(result[key], results[0][key])
        # Nonzero A updates are also fully trajectory-local: replaying the
        # rollout starts from zero and reproduces the first active rollout.
        for key in ('generated_token_ids', 'response_accepted_length_sum', 'response_verification_rounds'):
            self.assertEqual(results[3][key], results[4][key])


if __name__ == '__main__':
    unittest.main()
