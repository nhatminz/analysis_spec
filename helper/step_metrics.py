"""Common FastGRPO/SpecNaacl step telemetry, independent of model algorithms."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist
from helper.opd_reflex import OPD_COUNTER_NAMES,GENERATION_COUNTER_NAMES


COUNTERS = (
    'wall_time_s', 'generation_time_s', 'target_train_time_s', 'draft_train_time_s',
    'rollout_tokens', 'accepted_tokens', 'verification_rounds',
    'accepted_draft_tokens', 'proposed_draft_tokens',
)+tuple(name for name in OPD_COUNTER_NAMES if name!='opd_active_rows_max')+GENERATION_COUNTER_NAMES
DERIVED_OPD_FIELDS=tuple(f'{prefix}_{name}' for prefix in ('step','cumulative') for name in (
    'opd_kl','opd_mean_union_size','opd_target_compact_mass','opd_target_mass_in_draft_top16',
    'opd_active_token_rows','opd_active_rows_mean','opd_active_rows_max','opd_sparse_rounds','opd_dense_rounds','opd_fused_rounds','opd_gemm_rounds',
    'mean_active_responses_per_verify_round','mean_verified_tree_nodes',
    'mean_verified_path_length','mean_frontier_per_visited_state'))
MEMORY_FIELDS = ('gpu_allocated_gb', 'gpu_reserved_gb', 'gpu_peak_allocated_gb', 'gpu_free_gb')
STEP_FIELDS = ('step', 'method', 'grpo_step', 'phase_time_basis') + tuple(
    f'{prefix}_{name}' for prefix in ('cumulative', 'step') for name in COUNTERS
) + (
    'step_generation_tokens_per_s', 'cumulative_generation_tokens_per_s',
    'step_aal', 'cumulative_aal', 'step_acceptance_rate', 'cumulative_acceptance_rate',
) + DERIVED_OPD_FIELDS + MEMORY_FIELDS + ('rollout_tokens', 'tokens_per_s', 'aal', 'acceptance_rate')


class PhaseTimings:
    """CUDA stream elapsed intervals; CPU monotonic durations without CUDA.

    CUDA events are read only after the existing metric scalar transfer has
    completed the main stream. There is no synchronize/wait in this class.
    """
    def __init__(self, device, target_s=0.0, draft_s=0.0):
        self.device = torch.device(device)
        self.totals = {'target': float(target_s), 'draft': float(draft_s)}
        self.pending = []
        self.basis = 'cuda_stream_elapsed' if self.device.type == 'cuda' else 'cpu_monotonic'

    def begin(self, phase):
        started = time.perf_counter()
        event = None
        if self.device.type == 'cuda':
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
        return phase, started, event

    def end(self, ticket):
        phase, started, event = ticket
        ended = None
        if event is not None:
            ended = torch.cuda.Event(enable_timing=True)
            ended.record(torch.cuda.current_stream(self.device))
        self.pending.append((phase, time.perf_counter() - started, event, ended))

    def resolve(self):
        for phase, host_s, started, ended in self.pending:
            if ended is not None:
                if not ended.query():
                    raise RuntimeError('phase timings must be resolved after the metric scalar transfer')
                elapsed = started.elapsed_time(ended) / 1000.0
            else:
                elapsed = host_s
            self.totals[phase] += elapsed
        self.pending.clear()
        return self.totals.copy()


def gpu_memory_stats(device):
    """Allocator snapshots in GiB; no synchronization, cache flush or reset."""
    device = torch.device(device)
    if device.type != 'cuda':
        return dict.fromkeys(MEMORY_FIELDS, 0.0)
    free, _ = torch.cuda.mem_get_info(device)
    values = (torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device),
              torch.cuda.max_memory_allocated(device), free)
    return {name: value / (1024 ** 3) for name, value in zip(MEMORY_FIELDS, values)}


def completed_step_snapshot(global_metrics, data, timings, device, wall_time_s):
    """Called after _aggregate_job_metrics' existing GPU-to-CPU transfer."""
    durations = timings.resolve()
    interval_max=float(global_metrics.get('opd_interval_active_rows_max',0.))
    data['opd_interval_active_rows_max']=0.
    data['_phase_target_time_s'] = durations['target']
    data['_phase_draft_time_s'] = durations['draft']
    memory = gpu_memory_stats(device)
    wall = float(wall_time_s)
    target, draft = durations['target'], durations['draft']
    if dist.is_initialized():
        # Job counters are summed upstream; durations/memory reflect the slowest
        # / largest rank, and free memory the most constrained rank.
        values = torch.tensor([wall, target, draft, *[memory[name] for name in MEMORY_FIELDS[:3]],
                               -memory['gpu_free_gb']], device=device, dtype=torch.float64)
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
        wall, target, draft, allocated, reserved, peak, negative_free = values.tolist()
        memory = dict(zip(MEMORY_FIELDS, (allocated, reserved, peak, -negative_free)))
    return {
        'cumulative_wall_time_s': wall,
        'cumulative_generation_time_s': float(global_metrics['generate_time_cost']),
        'cumulative_target_train_time_s': target,
        'cumulative_draft_train_time_s': draft,
        'cumulative_rollout_tokens': int(global_metrics['total_rollout_tokens']),
        'cumulative_accepted_tokens': int(global_metrics['total_acc_length']),
        'cumulative_verification_rounds': int(global_metrics['total_decoded_token_num']),
        'cumulative_accepted_draft_tokens': int(global_metrics['total_accepted_draft_tokens']),
        'cumulative_proposed_draft_tokens': int(global_metrics['total_proposed_draft_tokens']),
        'phase_time_basis': timings.basis,
        **{f'cumulative_{name}':float(global_metrics.get(name,0.)) for name in OPD_COUNTER_NAMES+GENERATION_COUNTER_NAMES},
        'step_opd_active_rows_max':interval_max,
        **memory,
    }


def step_record(step, current, previous, extras=None):
    """Exact counter differences, never a moving average of batch ratios."""
    result = dict(extras or {})
    result.update(current)
    result['step'] = int(step)
    for name in COUNTERS:
        field = f'cumulative_{name}'
        result[field]=current.get(field,0.)
        result[f'step_{name}'] = result[field] - previous.get(field, 0)
    for prefix in ('step', 'cumulative'):
        tokens = result[f'{prefix}_rollout_tokens']
        elapsed = result[f'{prefix}_generation_time_s']
        result[f'{prefix}_generation_tokens_per_s'] = tokens / elapsed if elapsed > 0 else 0.0
        rounds = result[f'{prefix}_verification_rounds']
        result[f'{prefix}_aal'] = result[f'{prefix}_accepted_tokens'] / rounds if rounds > 0 else 0.0
        proposed = result[f'{prefix}_proposed_draft_tokens']
        result[f'{prefix}_acceptance_rate'] = (
            result[f'{prefix}_accepted_draft_tokens'] / proposed if proposed > 0 else 0.0)
        def ratio(numerator,denominator):
            n=result[f'{prefix}_{numerator}'];d=result[f'{prefix}_{denominator}']
            return n/d if d>0 else None
        for name,numerator,denominator in (
            ('opd_kl','opd_kl_sum','opd_state_weight'),
            ('opd_mean_union_size','opd_union_size_sum','opd_selected_states'),
            ('opd_target_compact_mass','opd_compact_mass_sum','opd_selected_states'),
            ('opd_target_mass_in_draft_top16','opd_draft_topk_target_mass_sum','opd_selected_states'),
            ('opd_active_token_rows','opd_active_rows_sum','opd_rounds'),
            ('opd_active_rows_mean','opd_active_rows_sum','opd_rounds'),
            ('mean_active_responses_per_verify_round','active_response_rounds','verification_batches'),
            ('mean_verified_tree_nodes','verified_tree_nodes','verification_rounds'),
            ('mean_verified_path_length','accepted_tokens','verification_rounds'),
            ('mean_frontier_per_visited_state','opd_frontier_states','opd_visited_states'),
        ):result[f'{prefix}_{name}']=ratio(numerator,denominator)
        if result[f'{prefix}_opd_nonfinite_kl_states']>0:result[f'{prefix}_opd_kl']=None
        result[f'{prefix}_opd_sparse_rounds']=result[f'{prefix}_opd_proposal_mode_sparse_rounds']
        result[f'{prefix}_opd_dense_rounds']=result[f'{prefix}_opd_proposal_mode_dense_rounds']
        for mode in ('fused','gemm'):
            result[f'{prefix}_opd_'+mode+'_rounds']=result[f'{prefix}_opd_proposal_mode_'+mode+'_rounds']
    result.setdefault('step_opd_active_rows_max',0.)
    result.setdefault('cumulative_opd_active_rows_max',0.)
    # Legacy plotting columns keep their former cumulative meaning.
    result.update(rollout_tokens=result['cumulative_rollout_tokens'],
                  tokens_per_s=result['cumulative_rollout_tokens'] / max(result['cumulative_wall_time_s'], 1e-9),
                  aal=result['cumulative_aal'], acceptance_rate=result['cumulative_acceptance_rate'])
    return result


class StepMetricsWriter:
    """One row per inherited GRPO label, including repeated optimizer updates.

    FastGRPO can perform several optimizer updates with the same `step` label.
    Replace the pending snapshot until the label advances; then write it once.
    The final label is flushed at normal/short-run termination. State is saved
    with the existing per-rank batch_data checkpoint.
    """
    def __init__(self, jsonl_path, csv_path, *, enabled=True, append=False, baseline=None, state=None):
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)
        self.enabled = enabled
        self.previous = dict(baseline or {})
        self.pending = None
        self.last_emitted = None
        if state is not None:
            self.previous = dict(state['previous'])
            self.pending = state.get('pending')
            self.last_emitted = state.get('last_emitted')
            if self.pending is None and self.last_emitted is not None:
                # A completed short run may resume while the inherited label
                # still repeats. Reopen that label and retain its full interval.
                self.previous = dict(self.last_emitted['baseline'])
                self.pending = {key: self.last_emitted[key] for key in ('step', 'snapshot', 'extras')}
        if enabled:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            if not append or not self.csv_path.exists():
                with self.csv_path.open('w', newline='', encoding='utf-8') as stream:
                    csv.DictWriter(stream, fieldnames=STEP_FIELDS).writeheader()
            else:
                self._upgrade_csv_header()
                if self.pending is not None:
                    self._rewind_step_rows(self.pending['step'])

    def _rewind_step_rows(self, step):
        """Discard only telemetry past the restored checkpoint, with backups."""
        with self.csv_path.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        kept = [row for row in rows if int(row['step']) < step]
        if len(kept) != len(rows):
            backup = self.csv_path.with_suffix('.pre_resume.csv')
            if not backup.exists():
                backup.write_bytes(self.csv_path.read_bytes())
            temporary = self.csv_path.with_suffix('.csv.tmp')
            with temporary.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=STEP_FIELDS, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(kept)
            temporary.replace(self.csv_path)
        if self.jsonl_path.exists():
            lines = self.jsonl_path.read_text(encoding='utf-8').splitlines(keepends=True)
            retained = []
            for line in lines:
                row = json.loads(line)
                if row.get('phase') == 'target_train' and int(row.get('step', -1)) >= step:
                    continue
                retained.append(line)
            if len(retained) != len(lines):
                backup = self.jsonl_path.with_suffix('.pre_resume.jsonl')
                if not backup.exists():
                    backup.write_bytes(self.jsonl_path.read_bytes())
                temporary = self.jsonl_path.with_suffix('.jsonl.tmp')
                temporary.write_text(''.join(retained), encoding='utf-8')
                temporary.replace(self.jsonl_path)

    def _upgrade_csv_header(self):
        with self.csv_path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) == STEP_FIELDS:
                return
            rows = list(reader)
        # Existing values survive schema migration; unavailable historical
        # per-step fields remain blank rather than being reconstructed/guessed.
        backup = self.csv_path.with_suffix('.pre_step_metrics.csv')
        if not backup.exists():
            backup.write_bytes(self.csv_path.read_bytes())
        temporary = self.csv_path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=STEP_FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(self.csv_path)

    def advance(self, step):
        if self.pending is not None and int(step) != self.pending['step']:
            if int(step) < self.pending['step']:
                raise ValueError('GRPO step labels must be monotonic')
            self.flush()

    def submit(self, step, snapshot, extras):
        self.advance(step)
        snapshot=dict(snapshot)
        if self.pending is not None:
            snapshot['step_opd_active_rows_max']=max(snapshot.get('step_opd_active_rows_max',0.),
                self.pending['snapshot'].get('step_opd_active_rows_max',0.))
        self.pending = {'step': int(step), 'snapshot': snapshot, 'extras': dict(extras)}

    def flush(self):
        if self.pending is None:
            return None
        row = step_record(self.pending['step'], self.pending['snapshot'], self.previous,
                          self.pending['extras'])
        if self.enabled:
            with self.jsonl_path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
            with self.csv_path.open('a', newline='', encoding='utf-8') as stream:
                csv.DictWriter(stream, fieldnames=STEP_FIELDS, extrasaction='ignore').writerow(row)
        self.last_emitted = {**self.pending, 'baseline': self.previous.copy()}
        self.previous = dict(self.pending['snapshot'])
        self.pending = None
        return row

    def state_dict(self):
        return {'previous': self.previous.copy(), 'pending': self.pending,
                'last_emitted': self.last_emitted}
