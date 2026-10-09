"""Strict SimpleLR official-split validation and immutable held-out selection."""
from pathlib import Path
import hashlib
import json
import random
import re
import unicodedata
import numpy as np
import pandas as pd
from motivation.state import atomic_json, file_hash
from helper.get_QAs import prompt_dict


def normalized_question(question):
    text = unicodedata.normalize('NFKC', question).strip()
    # SimpleLR's message wrapper is presentation, not question identity.
    text = re.sub(r'^Question:\s*', '', text)
    text = re.sub(r"\s*Answer:\s*Let's think step by step\.\s*$", '', text)
    return re.sub(r'\s+', ' ', text).strip()


def question_hash(question):
    return hashlib.sha256(normalized_question(question).encode()).hexdigest()


def extract_row(item, split, row,validate_answer=True):
    source=item.get('data_source')
    if source is not None and source not in ('simplelr','simplelr_abel','simplelr_abel_level3to5'):
        raise ValueError(f'{split} row {row}: data_source={source!r} is not SimpleLR')
    extra=item.get('extra_info')
    if isinstance(extra,dict) and extra.get('split') is not None and extra['split']!=split:
        raise ValueError(f'{split} row {row}: official split metadata says {extra["split"]!r}; refusing split swap')
    prompt = item.get('prompt')
    if isinstance(prompt, str):
        if prompt.lstrip().startswith('['):
            try: prompt = json.loads(prompt)
            except json.JSONDecodeError as e: raise ValueError(f'{split} row {row}: malformed JSON prompt') from e
        else: question = prompt
    if isinstance(prompt, (list, tuple, np.ndarray)):
        users = [m for m in prompt if isinstance(m, dict) and m.get('role', 'user') == 'user']
        if len(users) != 1:
            raise ValueError(f'{split} row {row}: expected one user message, found {len(users)}')
        question = users[0].get('content')
    elif not isinstance(prompt, str):
        # Supported already-extracted variant; never override a supplied prompt.
        question = item.get('question')
    if not isinstance(question, str) or not normalized_question(question):
        raise ValueError(f'{split} row {row}: missing nonempty prompt/question')
    reward = item.get('reward_model')
    if isinstance(reward, str):
        try: reward = json.loads(reward)
        except json.JSONDecodeError as e: raise ValueError(f'{split} row {row}: malformed reward_model') from e
    answer = reward.get('ground_truth') if isinstance(reward, dict) else item.get('answer')
    if answer is None or (isinstance(answer, float) and np.isnan(answer)):
        raise ValueError(f'{split} row {row}: missing reward_model.ground_truth/answer')
    answer = str(answer)
    if '####' in answer: answer = answer.split('####')[-1].strip()
    if '\\boxed{' not in answer: answer = f'\\boxed{{{answer}}}'
    from helper.rewards import parse_gold_answer
    if validate_answer:
        try: parse_gold_answer(answer)
        except ValueError as error: raise ValueError(f'{split} row {row}: {error}') from error
    return dict(id=f'{split}:{row}', split=split, row_id=row, source_id=str(item.get('unique_id', row)),
                question=question, answer=answer, question_sha256=question_hash(question))


def read_split(path, split,invalid_answer_policy='exclude'):
    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix != '.parquet': raise ValueError(f'SimpleLR {split} must be a Parquet file: {path}')
    table = pd.read_parquet(path)
    if invalid_answer_policy not in ('error','exclude'):raise ValueError('Invalid answer policy must be error or exclude')
    rows = [extract_row(item, split, i,validate_answer=False) for i, item in enumerate(table.to_dict('records'))]
    seen, unique, duplicates = {}, [], []
    for row in rows:
        key = row['question_sha256']
        if key in seen:
            first = seen[key]
            if row['answer'] != first['answer']:
                raise ValueError(f'{split}: conflicting answers for duplicate questions {first["id"]}/{row["id"]}')
            duplicates.append(dict(kept=first['id'], removed=row['id'], question_sha256=key))
        else: seen[key] = row; unique.append(row)
    from helper.rewards import parse_gold_answer
    import warnings
    raw_unique=len(unique);all_hashes=[r['question_sha256'] for r in unique];valid=[];invalid=[];text_fallback=[]
    for row in unique:
        try:
            parsed=parse_gold_answer(row['answer'])
            if all(isinstance(x,str) for x in parsed):text_fallback.append(row['id'])
            valid.append(row)
        except ValueError as error:
            if invalid_answer_policy=='error':raise ValueError(f'{row["id"]}: {error}') from error
            invalid.append(dict(id=row['id'],answer=row['answer'],reason=str(error),question_sha256=row['question_sha256']))
    if invalid:warnings.warn(f'{split}: excluding {len(invalid)} invalid gold labels: '+', '.join(r['id'] for r in invalid),RuntimeWarning)
    unique=valid
    info = dict(path=str(path), sha256=file_hash(path), raw_rows=len(rows), unique_rows=len(unique),raw_unique_rows=raw_unique,
                invalid_answer_policy=invalid_answer_policy,invalid_answers=invalid,exact_text_gold_ids=text_fallback,
                all_question_hashes=all_hashes,
                schema={k: str(v) for k, v in table.dtypes.items()}, duplicates=duplicates,
                rows=[{k: r[k] for k in ('id','row_id','source_id','question_sha256')} for r in unique])
    return unique, info


def messages(row):
    template = prompt_dict['math']
    return [dict(role='system', content=template['system_prompt']),
            dict(role='user', content=template['user_prompt'].format_map({'instruction': row['question']}))]


def render(row, tokenizer):
    return tokenizer.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True)


def tokenize(rows, tokenizer):
    return tokenizer([render(row, tokenizer) for row in rows], return_tensors='pt', padding=True,
                     add_special_tokens=False)


def prepare(train_path, test_path, output_dir, tokenizer, subset_seed=2026,invalid_answer_policy='exclude'):
    train, train_info = read_split(train_path, 'train',invalid_answer_policy)
    test, test_info = read_split(test_path, 'test',invalid_answer_policy)
    if train_info['path'] == test_info['path']: raise ValueError('Train and test must be separate official files')
    overlap = set(train_info['all_question_hashes']) & set(test_info['all_question_hashes'])
    if overlap:
        examples = [r['id'] for r in test if r['question_sha256'] in overlap][:10]
        raise ValueError(f'Train/test normalized-question leakage: {len(overlap)} questions, test IDs {examples}. '
                         'Supply disjoint official SimpleLR splits; no automatic removal across splits.')
    if len(test) < 64:
        raise ValueError(f'SimpleLR test has only {len(test)} unique questions; at least 64 disjoint test questions '
                         'are required. Supply the official test.parquet; train fallback is forbidden.')
    if len(train) < 8: raise ValueError('At least 8 unique training questions are required')
    ordered = list(test); random.Random(subset_seed).shuffle(ordered); ordered = ordered[:64]
    template = dict(math=prompt_dict['math'], chat_template=tokenizer.chat_template,
                    padding_side=tokenizer.padding_side, add_special_tokens=False, add_generation_prompt=True,
                    truncation=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    manifest = dict(format='simplelr_official_test_v1', eval_subset_seed=subset_seed, train=train_info, test=test_info,
                    prompt_rendering=template, prompt_template_sha256=hashlib.sha256(
                        json.dumps(template, sort_keys=True).encode()).hexdigest(), eval_responses_per_prompt=1,
                    subset=[{k: r[k] for k in ('id','row_id','source_id','question_sha256')} |
                            dict(ordinary_measurement=i < 16, rendered_sha256=hashlib.sha256(render(r, tokenizer).encode()).hexdigest(),
                                 token_ids_sha256=hashlib.sha256(json.dumps(tokenizer.encode(render(r, tokenizer),
                                     add_special_tokens=False)).encode()).hexdigest()) for i, r in enumerate(ordered)])
    path = Path(output_dir) / 'eval_subset_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest: raise ValueError('Stored evaluation subset/data/tokenizer differs; start a new output directory')
    else: atomic_json(path, manifest)
    return train, ordered, manifest
