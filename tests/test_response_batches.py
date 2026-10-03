import unittest

from helper.response_batches import microbatch_index_groups, reorder_response_fields


class ResponseBatchTests(unittest.TestCase):
    def test_advantages_and_identities_follow_sequences_through_last_microbatch(self):
        lengths = [6, 2, 5, 3]
        fields = {
            "input_ids": [[10] * 6, [11] * 2, [12] * 5, [13] * 3],
            "attention_mask": [[1] * n for n in lengths],
            "loss_mask": [[0] + [1] * (n - 1) for n in lengths],
            "std_rewards": [1.0, -2.0, 3.0, -4.0],
            "response_ids": ["p0:r0", "p1:r0", "p2:r0", "p3:r0"],
        }
        permutation, sorted_fields = reorder_response_fields(lengths, fields)
        self.assertEqual(permutation, [1, 3, 2, 0])
        groups = list(microbatch_index_groups(
            [len(row) for row in sorted_fields["input_ids"]], 7, 4096
        ))
        self.assertEqual(groups, [[0, 1], [2], [3]])
        expected = {
            "p0:r0": (10, 1.0), "p1:r0": (11, -2.0),
            "p2:r0": (12, 3.0), "p3:r0": (13, -4.0),
        }
        for group in groups:
            for index in group:
                identity = sorted_fields["response_ids"][index]
                self.assertEqual(
                    (sorted_fields["input_ids"][index][0], sorted_fields["std_rewards"][index]),
                    expected[identity],
                )
        self.assertEqual(sorted_fields["response_ids"][groups[-1][0]], "p0:r0")

    def test_rejects_response_field_with_missing_row(self):
        with self.assertRaisesRegex(ValueError, "std_rewards"):
            reorder_response_fields([2, 3], {"std_rewards": [1.0]})


if __name__ == "__main__":
    unittest.main()
