"""FastGRPO source baseline and its full-vocabulary ReflexOPD proposal variant."""
from helper.fastgrpo_generate import sampling, get_adaptive_hyperparameters
from helper.method_config import resolve_method


def speculative_generate(*args, method="fastgrpo", **kwargs):
    method, _ = resolve_method(method)
    if method == "fastgrpo":
        from helper.fastgrpo_generate import speculative_generate as generate
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("opd_") and k != "kv_gather_strategy"}
    else:
        from helper.opd_generate import speculative_generate as generate
    return generate(*args, **kwargs)
