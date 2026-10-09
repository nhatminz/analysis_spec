"""API adapters for original FastGRPO on newer Transformers.

No attention, sampling, tree, loss or RNG arithmetic is changed. Transformers
4.x continues to use its original decoder/cache classes without an adapter.
"""
import inspect
import weakref
import torch
from transformers.cache_utils import DynamicCache as HF_DynamicCache


class _CacheTensorList(list):
    """A real list for torch.stack, with legacy indexed writes updating HF layers."""
    def __init__(self, cache, attribute):
        super().__init__()
        self._cache_ref, self.attribute = weakref.ref(cache), attribute

    @property
    def cache(self):
        cache = self._cache_ref()
        if cache is None:
            raise RuntimeError('cache view used after cache was released')
        return cache

    def __setitem__(self, index, tensor):
        if isinstance(index, slice):
            raise TypeError('FastGRPO cache adapter supports indexed writes only')
        setattr(self.cache.layers[index], self.attribute, tensor)
        super().__setitem__(index, tensor)

    def sync_layer(self, index):
        while list.__len__(self) <= index:
            list.append(self, None)
        list.__setitem__(self, index, getattr(self.cache.layers[index], self.attribute))


class LegacyDynamicCache(HF_DynamicCache):
    """Modern DynamicCache exposing the key/value lists used by upstream."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_cache = _CacheTensorList(self, 'keys')
        self.value_cache = _CacheTensorList(self, 'values')
        self._sync_lists()

    def _sync_lists(self):
        for index in range(len(self.layers)):
            self.key_cache.sync_layer(index)
            self.value_cache.sync_layer(index)

    def __getitem__(self, layer_idx):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __iter__(self):
        return iter(zip(self.key_cache, self.value_cache))

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        output = super().update(key_states, value_states, layer_idx, *args, **kwargs)
        self.key_cache.sync_layer(layer_idx)
        self.value_cache.sync_layer(layer_idx)
        return output

    def crop(self, max_length):
        super().crop(max_length)
        self._sync_lists()

    def batch_repeat_interleave(self, repeats):
        super().batch_repeat_interleave(repeats)
        self._sync_lists()

    def batch_select_indices(self, indices):
        super().batch_select_indices(indices)
        self._sync_lists()

    def reset(self):
        super().reset()
        self._sync_lists()


# Detect the actual API once at import, outside generation/training hot paths.
DynamicCache = HF_DynamicCache if hasattr(HF_DynamicCache(), 'key_cache') else LegacyDynamicCache
_DECODER_ADAPTERS = {}
_MISSING = object()


def _decoder_forward_adapter(native_forward):
    def forward(self, *args, past_key_value=_MISSING, **kwargs):
        legacy = past_key_value is not _MISSING
        if legacy:
            if 'past_key_values' in kwargs:
                raise ValueError('specify one cache argument')
            kwargs['past_key_values'] = past_key_value
            kwargs.pop('output_attentions', None)
        output = native_forward(self, *args, **kwargs)
        return (output,) if legacy and torch.is_tensor(output) else output
    return forward


def prepare_target_decoder_api(target):
    """Accept legacy singular cache/tuple calls while retaining native 5.x calls.

    Normal target pretrain/GRPO forwards still call the native plural API and
    receive tensors. Only the direct upstream speculative decoder calls receive
    a one-element tuple. Parameter names and state_dict keys stay identical.
    """
    backbone = target.model
    for layer in backbone.layers:
        original = type(layer)
        if getattr(original, '_fastgrpo_decoder_api_adapter', False):
            continue
        parameters = inspect.signature(original.forward).parameters
        if 'past_key_values' not in parameters or 'past_key_value' in parameters:
            continue
        adapter = _DECODER_ADAPTERS.get(original)
        if adapter is None:
            adapter = type(original.__name__+'FastGRPOAPI', (original,),
                           dict(forward=_decoder_forward_adapter(original.forward),
                                _fastgrpo_decoder_api_adapter=True))
            _DECODER_ADAPTERS[original] = adapter
        layer.__class__ = adapter
