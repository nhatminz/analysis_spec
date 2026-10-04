import unittest

from helper.rollout_merge import merge_rollout_outputs


def part(first_token, capture):
    return {
        'generated_token_ids': [[first_token], [first_token + 1]],
        'all_draft_input_states': [f'f{first_token}', f'f{first_token + 1}'] if capture else None,
        'all_target_hidden_states': [f'h{first_token}', f'h{first_token + 1}'] if capture else None,
        'all_draft_input_ids': [f'x{first_token}', f'x{first_token + 1}'] if capture else None,
        'response_accepted_length_sum': [2, 3],
        'response_verification_rounds': [1, 1],
        'response_generated_tokens': [1, 1],
        'total_acc_length': 5,
        'total_decoded_token_num': 2,
        'total_accepted_draft_tokens': 3,
        'total_proposed_draft_tokens': 4,
        'total_accepted_medusa_tokens': 3,
        'total_proposed_medusa_tokens': 4,
        'total_time_cost': 1.0,
        'target_time_cost': 0.2,
        'draft_time_cost': 0.3,
        'check_time_cost': 0.1,
        'prefill_time_cost': 0.1,
        'post_time_cost': 0.1,
        'max_sequence_length': 1,
        'draft_acceptance_rate': 0.75,
        'medusa_acceptance_rate': 0.75,
        'total_acc': 2.5,
    }


class RolloutMergeTests(unittest.TestCase):
    def test_chunks_keep_response_order_and_sum_aal_numerators(self):
        result = merge_rollout_outputs([part(10, True), part(20, True)])
        self.assertEqual(result['generated_token_ids'], [[10], [11], [20], [21]])
        self.assertEqual(result['all_draft_input_states'], ['f10', 'f11', 'f20', 'f21'])
        self.assertEqual(result['total_acc_length'], 10)
        self.assertEqual(result['total_decoded_token_num'], 4)
        self.assertEqual(result['total_acc'], 2.5)

    def test_evaluation_without_capture_and_mixed_capture_rejected(self):
        self.assertIsNone(merge_rollout_outputs([part(10, False), part(20, False)])[
            'all_target_hidden_states'
        ])
        with self.assertRaisesRegex(ValueError, 'inconsistent decoder field'):
            merge_rollout_outputs([part(10, True), part(20, False)])

    def test_reflex_chunk_timings_are_summed_and_per_update_recomputed(self):
        parts = [part(10, False), part(20, False)]
        for index, value in enumerate(parts):
            value.update(reflex_updates=2 + index, reflex_update_count=2 + index,
                         reflex_update_gpu_ms=3.0 + index, reflex_update_exposed_ms=0.5,
                         reflex_state_initialized_zero=True, reflex_state_cleared=True)
        result = merge_rollout_outputs(parts)
        self.assertEqual(result['reflex_update_count'], 5)
        self.assertEqual(result['reflex_update_gpu_ms'], 7.0)
        self.assertEqual(result['reflex_update_gpu_ms_per_update'], 7 / 5)
        self.assertEqual(result['reflex_update_exposed_ms'], 1.0)
        self.assertTrue(result['reflex_state_cleared'])


if __name__ == '__main__':
    unittest.main()
