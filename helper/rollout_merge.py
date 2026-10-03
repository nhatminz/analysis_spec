"""Combine sequential decoder chunks without changing prompt/response order."""

from __future__ import annotations


LIST_FIELDS = (
    'generated_token_ids', 'all_draft_input_states', 'all_target_hidden_states',
    'all_draft_input_ids', 'response_accepted_length_sum',
    'response_verification_rounds', 'response_generated_tokens',
)
SUM_FIELDS = (
    'total_acc_length', 'total_decoded_token_num',
    'total_accepted_draft_tokens', 'total_proposed_draft_tokens',
    'total_accepted_medusa_tokens', 'total_proposed_medusa_tokens',
    'total_time_cost', 'target_time_cost', 'draft_time_cost',
    'check_time_cost', 'prefill_time_cost', 'post_time_cost',
)


def merge_rollout_outputs(parts):
    if not parts:
        raise ValueError('cannot merge an empty decoder rollout')
    combined = dict(parts[0])
    for field in LIST_FIELDS:
        if combined[field] is not None:
            combined[field] = list(combined[field])
    for part in parts[1:]:
        for field in LIST_FIELDS:
            if (combined[field] is None) != (part[field] is None):
                raise ValueError(f'inconsistent decoder field {field} across prompt chunks')
            if part[field] is not None:
                combined[field].extend(part[field])
        for field in SUM_FIELDS:
            combined[field] += part[field]
        combined['max_sequence_length'] = max(
            combined['max_sequence_length'], part['max_sequence_length']
        )
    combined['draft_acceptance_rate'] = (
        combined['total_accepted_draft_tokens'] /
        max(combined['total_proposed_draft_tokens'], 1)
    )
    combined['medusa_acceptance_rate'] = combined['draft_acceptance_rate']
    combined['total_acc'] = (
        combined['total_acc_length'] / max(combined['total_decoded_token_num'], 1)
    )
    if any(len(combined[field]) != len(combined['generated_token_ids']) for field in (
        'response_accepted_length_sum', 'response_verification_rounds',
        'response_generated_tokens',
    )):
        raise RuntimeError('merged decoder response metrics have different row counts')
    return combined
