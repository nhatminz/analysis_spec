import copy
import ast
import os
import random
import tempfile
import unittest
from pathlib import Path

from helper.policy_lag_protocol import branch_spec, is_analysis_boundary, paired_shadow_rollout

try:
    import torch
except ImportError:
    torch = None


class ScheduleTests(unittest.TestCase):
    def test_schedule_excludes_terminal_step(self):
        actual = [step for step in range(1, 601) if is_analysis_boundary(step, {1}, 5, 600)]
        self.assertEqual(actual, [1] + list(range(5, 600, 5)))
        self.assertEqual(len(actual), 120)
        self.assertFalse(is_analysis_boundary(600, {600}, 5, 600))

    def test_reflex_is_stale_plus_active_not_fresh_plus_active(self):
        self.assertEqual(branch_spec('stale'), ('stale', 'off'))
        self.assertEqual(branch_spec('fresh'), ('fresh', 'off'))
        self.assertEqual(branch_spec('reflex'), ('stale', 'active'))


@unittest.skipUnless(torch is not None, 'PyTorch unavailable')
class PairedShadowTests(unittest.TestCase):
    def setUp(self):
        from policy_lag_analysis import state_digest
        self.digest = state_digest
        torch.manual_seed(42)
        random.seed(42)
        self.draft = torch.nn.Linear(3, 2, bias=False)
        self.target = torch.nn.Linear(3, 2, bias=False)
        self.optimizer = torch.optim.AdamW(self.draft.parameters(), lr=1e-5)
        self.draft(torch.ones(1, 3)).square().mean().backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.base = copy.deepcopy(self.draft.state_dict())
        self.base_optimizer = copy.deepcopy(self.optimizer.state_dict())
        self.real_buffer = {'response_ids': [(4, 0), (4, 1)], 'advantages': [-1, 1]}

    def capture_rng(self):
        return {'python': random.getstate(), 'torch': torch.random.get_rng_state()}

    def restore_rng(self, state):
        random.setstate(state['python'])
        torch.random.set_rng_state(state['torch'])

    def fingerprint(self):
        return (self.digest(self.target.state_dict()), self.digest(self.optimizer.state_dict()),
                self.digest(self.real_buffer))

    def run_shadow(self, branch, branch_state, stale_state, rng, rollout):
        return paired_shadow_rollout(
            branch=branch, branch_state=branch_state, stale_state=stale_state, rng_state=rng,
            expected_draft_id=self.digest(branch_state), digest=self.digest,
            load_draft=self.draft.load_state_dict, read_draft=self.draft.state_dict,
            persistent_identity=self.fingerprint, capture_rng=self.capture_rng,
            restore_rng=self.restore_rng, rollout=rollout)

    def test_stale_fresh_have_identical_base_optimizer_and_lr(self):
        initial_ids = []
        branch_states = []
        for features in (torch.ones(2, 3), -torch.ones(2, 3)):
            self.draft.load_state_dict(self.base)
            self.optimizer.load_state_dict(copy.deepcopy(self.base_optimizer))
            initial_ids.append((self.digest(self.draft.state_dict()), self.digest(self.optimizer.state_dict())))
            self.assertEqual([group['lr'] for group in self.optimizer.param_groups], [1e-5])
            self.optimizer.zero_grad(set_to_none=True)
            (self.draft(features) - 1).square().mean().backward()
            self.optimizer.step()
            branch_states.append(copy.deepcopy(self.draft.state_dict()))
        self.assertEqual(initial_ids[0], initial_ids[1])
        self.assertNotEqual(self.digest(branch_states[0]), self.digest(branch_states[1]))

    def test_shadow_pairing_and_real_rng_state_are_restored(self):
        stale = self.base
        fresh = {name: value + 1 for name, value in stale.items()}
        initial = self.capture_rng()
        observed = []
        prompt_batch = [[11, 12], [21, 22]]

        def generate(mode):
            observed.append((mode, copy.deepcopy(prompt_batch), self.digest(self.capture_rng())))
            return torch.rand(3).tolist(), random.random()

        fingerprint = self.fingerprint()
        f = self.run_shadow('fresh', fresh, stale, initial, generate)
        r = self.run_shadow('reflex', stale, stale, initial, generate)
        real = generate('off')
        self.assertEqual(f, r)
        self.assertEqual(r, real)
        self.assertEqual({item[2] for item in observed}, {self.digest(initial)})
        self.assertEqual([item[0] for item in observed], ['off', 'active', 'off'])
        self.assertEqual(self.digest(self.draft.state_dict()), self.digest(stale))
        self.assertEqual(self.fingerprint(), fingerprint)

    def test_reflex_rejects_fresh_draft(self):
        fresh = {name: value + 1 for name, value in self.base.items()}
        with self.assertRaisesRegex(RuntimeError, 'phi_stale'):
            self.run_shadow('reflex', fresh, self.base, self.capture_rng(), lambda _: None)

    def test_resume_reconstructs_next_prompt_batch_without_consuming_rollout_rng(self):
        initial = torch.random.get_rng_state()
        generator = torch.Generator().manual_seed(42 + 3)
        batches = list(torch.utils.data.DataLoader(list(range(32)), batch_size=8, shuffle=True,
                                                 generator=generator, drop_last=True))
        resumed_generator = torch.Generator().manual_seed(42 + 3)
        resumed = list(torch.utils.data.DataLoader(list(range(32)), batch_size=8, shuffle=True,
                                                 generator=resumed_generator, drop_last=True))
        self.assertTrue(torch.equal(resumed[2], batches[2]))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), initial))

    def test_training_checkpoint_round_trip_restores_all_rngs_and_pending_state(self):
        import numpy as np
        # Exercise the actual save/load functions without importing the
        # top-level script (which parses CLI and loads the 3B target).
        source = Path(__file__).resolve().parents[1] / 'grpo_speculative.py'
        selected = {'_atomic_torch_save', '_prune_checkpoints',
                    'save_training_checkpoint', 'load_training_checkpoint'}
        definitions = [node for node in ast.parse(source.read_text()).body
                       if isinstance(node, ast.FunctionDef) and node.name in selected]
        self.assertEqual({node.name for node in definitions}, selected)
        namespace = dict(torch=torch, np=np, random=random, os=os, Path=Path,
                         _target_lora_state_dict=lambda model: model.state_dict(),
                         _load_target_lora_state_dict=lambda model, state: model.load_state_dict(state))
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
        model = type('ToyModel', (), {})()
        model.draft_model, model.target_model = self.draft, self.target
        target_optimizer = torch.optim.AdamW(self.target.parameters(), lr=1e-6)
        torch.manual_seed(49)
        random.seed(17)
        np.random.seed(31)

        def draw():
            return (random.random(), np.random.random(), torch.rand(2).tolist(),
                    torch.rand(2, device='cuda').cpu().tolist() if torch.cuda.is_available() else None)

        with tempfile.TemporaryDirectory() as directory:
            namespace['save_training_checkpoint'](
                directory, model=model, optimizer_target=target_optimizer, optimizer_draft=self.optimizer,
                epoch=3, next_batch=2, step=6, used_items=24, draft_step=11,
                draft_accumulated_step=11, batch_data=self.real_buffer, keep_last=3,
                target_optimizer_steps=6)
            expected = draw()
            self.draft.weight.data.add_(1)
            self.optimizer.param_groups[0]['lr'] = 9
            random.seed(99)
            np.random.seed(99)
            torch.manual_seed(99)
            restored = namespace['load_training_checkpoint'](
                Path(directory) / 'latest.pt', model=model,
                optimizer_target=target_optimizer, optimizer_draft=self.optimizer)
            self.assertEqual(draw(), expected)
            self.assertEqual(restored['target_optimizer_steps'], 6)
            self.assertEqual((restored['epoch'], restored['next_batch']), (3, 2))
            self.assertEqual(restored['batch_data'], self.real_buffer)
            self.assertEqual(self.digest(self.draft.state_dict()), self.digest(self.base))
            self.assertEqual(self.digest(self.optimizer.state_dict()), self.digest(self.base_optimizer))
            # Legacy checkpoints lacking Python/NumPy fields remain readable.
            restored.pop('python_rng_state')
            restored.pop('numpy_rng_state')
            legacy = Path(directory) / 'legacy.pt'
            torch.save(restored, legacy)
            namespace['load_training_checkpoint'](
                legacy, model=model, optimizer_target=target_optimizer, optimizer_draft=self.optimizer)

    def test_detects_optimizer_mutation_and_restores_stale_rng_on_error(self):
        initial = self.capture_rng()
        def bad_rollout(_):
            self.optimizer.param_groups[0]['lr'] = 9
            torch.rand(1)
        with self.assertRaisesRegex(RuntimeError, 'target/optimizer/GRPO'):
            self.run_shadow('reflex', self.base, self.base, initial, bad_rollout)
        self.assertEqual(self.digest(self.capture_rng()), self.digest(initial))
        self.assertEqual(self.digest(self.draft.state_dict()), self.digest(self.base))


if __name__ == '__main__':
    unittest.main()
