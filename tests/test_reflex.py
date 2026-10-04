import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None


@unittest.skipUnless(torch is not None, 'PyTorch unavailable')
class ReflexTests(unittest.TestCase):
    def test_adapter_compact_api_and_head_do_not_register_draft_twice(self):
        from helper.eagle3_specforge import Eagle3FastGRPOAdapter, _TargetVocabHead
        class TinyDraft(torch.nn.Module):
            vocab_size = 8
            draft_vocab_size = 3
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(4, 3)
                self.register_buffer('d2t', torch.tensor([3, 4, 5]))
            def compute_logits(self, hidden):
                return self.head(hidden)
        adapter = Eagle3FastGRPOAdapter.__new__(Eagle3FastGRPOAdapter)
        torch.nn.Module.__init__(adapter)
        adapter.draft_model = TinyDraft()
        hidden = torch.randn(2, 1, 4)
        self.assertEqual(adapter.compact_vocab_size, 3)
        mapping = adapter.compact_to_target_ids(device='cpu')
        self.assertEqual(mapping.tolist(), [3, 5, 7])
        compact = adapter.compute_compact_logits(hidden)
        head = _TargetVocabHead(adapter.draft_model)
        self.assertEqual(list(head.parameters()), [])
        self.assertTrue(all(parameter.requires_grad for parameter in adapter.draft_model.parameters()))
        torch.testing.assert_close(head(hidden).index_select(-1, mapping), compact)

    def test_zero_reset_compact_mapping_and_trajectory_local_update(self):
        from helper.fast_lk_reflex import FastLKReflex, reflex_or_baseline_probabilities, topk_compact_candidates
        torch.manual_seed(42)
        reflex = FastLKReflex(feature_dim=2, backend='torch')
        reflex.start(2, 3, 4, 'cpu')
        logits = torch.tensor([[[2., 0., -1.]], [[2., 0., -1.]]])
        hidden = torch.ones(2, 1, 4)
        mapping = torch.tensor([1, 3, 6])
        baseline = reflex_or_baseline_probabilities(logits)
        values, compact, target = reflex.propose(logits, hidden, 3, mapping, root=True)
        expected = topk_compact_candidates(baseline, mapping, 3)
        for actual, reference in zip((values, compact, target), expected):
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        teacher = torch.zeros(2, 7)
        teacher[0, mapping] = baseline[0, 0]
        teacher[1, 6] = 1
        reflex.update_from_target_probs(teacher, mapping)
        self.assertLess(float(reflex.state[0].abs().max()), 1e-6)
        self.assertGreater(float(reflex.state[1].abs().max()), 0)
        self.assertFalse(reflex.state.requires_grad)
        self.assertIsNone(reflex.state.grad_fn)
        self.assertEqual(reflex.finish().updates, 2)
        reflex.remove_finished([0])
        self.assertEqual(reflex.active_trajectories, 1)
        reflex.clear()
        self.assertIsNone(reflex.state)
        reflex.start(2, 3, 4, 'cpu')
        self.assertEqual(int(torch.count_nonzero(reflex.state)), 0)
        self.assertEqual(reflex.finish().updates, 0)

    def test_sampling_feedback_reuses_logits_without_model_forward(self):
        from helper.sampling import sample_target_from_logits, build_sampling_probs
        from helper.fast_lk_reflex import FastLKReflex
        reflex = FastLKReflex(feature_dim=2, backend='torch')
        reflex.start(1, 3, 4, 'cpu')
        reflex.propose(torch.zeros(1, 1, 3), torch.ones(1, 1, 4), 2, torch.tensor([1, 3, 6]), root=True)
        logits = torch.tensor([[[0., 1., 2., 3., 4., 5., 6.]]])
        with mock.patch.object(reflex, 'update_from_target_probs', wraps=reflex.update_from_target_probs) as update:
            tokens, probs = sample_target_from_logits(logits, do_sample=True, temperature=0.8,
                top_p=0.95, top_k=None, eos_token_id=2, reflex=reflex, compact_to_target=torch.tensor([1, 3, 6]))
            self.assertEqual(tokens.shape, (1, 1))
            torch.testing.assert_close(update.call_args.args[0], probs[:, 0])
            torch.testing.assert_close(probs, build_sampling_probs(logits, 0.8, 0.95, None, 2))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA unavailable')
    def test_triton_update_stream_timing_and_zero_state_parity(self):
        from helper.fast_lk_reflex import FastLKReflex
        if __import__('importlib').util.find_spec('triton') is None:
            self.skipTest('Triton unavailable')
        torch.manual_seed(42)
        logits = torch.randn(2, 1, 17, device='cuda')
        hidden = torch.randn(2, 1, 8, device='cuda')
        mapping = torch.arange(17, device='cuda') + 2
        triton_reflex = FastLKReflex(feature_dim=8, backend='triton', time_updates=True)
        reference = FastLKReflex(feature_dim=8, backend='torch')
        for reflex in (triton_reflex, reference):
            reflex.start(2, 17, 8, 'cuda')
        active = triton_reflex.propose(logits, hidden, 4, mapping, root=True)
        off = reference.propose(logits, hidden, 4, mapping, root=True)
        torch.testing.assert_close(active[0], off[0], rtol=3e-5, atol=1e-6)
        self.assertTrue(torch.equal(active[2], off[2]))
        target = torch.randn(2, 23, device='cuda').softmax(-1)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        # No device synchronize in the feedback/update round.
        with mock.patch('torch.cuda.synchronize', side_effect=AssertionError('per-round synchronization')):
            with torch.cuda.stream(stream):
                triton_reflex.update_from_target_probs(target, mapping)
            torch.cuda.current_stream().wait_stream(stream)
            reference.update_from_target_probs(target, mapping)
            stats = triton_reflex.finish()
        torch.testing.assert_close(triton_reflex.state, reference.state, rtol=2e-4, atol=2e-6)
        self.assertEqual(stats.updates, 2)
        self.assertGreater(stats.update_gpu_ms, 0)
        self.assertEqual(set(stats.profile_sections_ms), {'feedback_update_ms'})
        triton_reflex.clear()
        self.assertIsNone(triton_reflex.state)


if __name__ == '__main__':
    unittest.main()
