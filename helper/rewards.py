"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
import warnings
from typing import Dict
from functools import lru_cache

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

warnings.filterwarnings(
    "ignore",
    message="equations=True in NormalizationConfig is deprecated.*",
)

_ANSWER_EXTRACTION_CONFIG = [
    LatexExtractionConfig(
        normalization_config=NormalizationConfig(
            nits=False,
            malformed_operators=False,
            basic_latex=True,
            boxed="all",
            units=True,
        ),
        boxed_match_priority=0,
        try_extract_without_anchor=False,
    )
]


@lru_cache(maxsize=16384)
def parse_gold_answer(solution):
    """A malformed gold label is a data error, never a correct completion."""
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError('Ground truth must be a nonempty mathematical answer')
    # SimpleLR contains e.g. 29,\!322,\!216 inside fractions. The parser
    # cannot read that grouping. Remove only explicit LaTeX-spaced thousands
    # groups, preserving ordinary commas in tuples, sets and decimal notation.
    normalized=re.sub(r'(?<![\w.])\d{1,3}(?:,(?:\\!)?\d{3})+(?![\d.])',
                      lambda m:m[0].replace(',', '').replace(r'\!', '') if r'\!' in m[0] else m[0],solution)
    try:
        parsed = parse(normalized, extraction_mode='first_match',
                       extraction_config=[LatexExtractionConfig()])
    except Exception as error:
        raise ValueError(f'Cannot parse ground truth {solution!r}') from error
    if not parsed or all(isinstance(x,str) and not x.strip() for x in parsed):
        raise ValueError(f'Cannot parse ground truth {solution!r}; fix the SimpleLR label before training')
    return tuple(parsed)


def accuracy_reward_func(completions, solution, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    if len(completions) != len(solution):
        raise ValueError('Completion and ground-truth counts differ')
    rewards = []
    for content, sol in zip(completions, solution):
        gold_parsed = list(parse_gold_answer(sol))
        try:
            answer_parsed = parse(
                content,
                extraction_config=_ANSWER_EXTRACTION_CONFIG,
                extraction_mode="first_match",
            )
            reward = float(verify(answer_parsed, gold_parsed))
        except Exception as error:
            warnings.warn(f'Completion verification failed: {error}', RuntimeWarning)
            reward = 0.0
        rewards.append(reward)

    return rewards


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("\n</think>\n") == 1:
            count += 1.0
        return count

    return [count_tags(c) for c in completions]
