"""StreamingLLM-style sink + sliding-window KV cache eviction.

Positional, attention-agnostic eviction: retains the first `sink_size`
tokens plus the most recent `window_size` tokens, dropping everything in
between. Unlike attention-score-based compressors (H2O, SnapKV, FastKV),
which token survives depends only on its position, never on what it
attends to -- this is the mechanism this project's safety experiment is
built to isolate.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers.cache_utils import Cache


def compute_position_ids(seen_tokens_before: int, num_new_tokens: int, device="cpu") -> torch.Tensor:
    """True absolute positions for the new tokens about to be cached.

    Not a "remap" in the renumbering sense -- just correct absolute
    positions, computed from a counter that (unlike Cache.get_seq_length())
    doesn't shrink under eviction.
    """
    return torch.arange(
        seen_tokens_before, seen_tokens_before + num_new_tokens,
        dtype=torch.long, device=device,
    )


class PositionalEvictionCache(Cache):
    """
    StreamingLLM-style sink + sliding-window eviction.

    Key design choices:
      - Eviction is uniform across layers -> retained-position bookkeeping
        (_seen_tokens, _retained_positions) is SHARED, computed once at
        layer_idx==0 and reused by every other layer in the same forward call.
      - Retained tokens keep their TRUE original absolute position (gaps
        allowed) -- no compaction, no re-rotation of already-cached keys.
      - get_seq_length() reports the RETAINED (physical) length, used for
        attention-mask sizing. _seen_tokens reports the TRUE total tokens
        ever appended, used for RoPE position_ids. These diverge once
        eviction starts -- that divergence is the whole point.
    """

    def __init__(self, num_layers: int, sink_size: int, window_size: int):
        super().__init__()
        self.sink_size = sink_size
        self.window_size = window_size
        self.max_retained = sink_size + window_size

        self.key_cache: List[Optional[torch.Tensor]] = [None] * num_layers
        self.value_cache: List[Optional[torch.Tensor]] = [None] * num_layers

        self._retained_positions: Optional[torch.Tensor] = None  # shared, true absolute positions
        self._seen_tokens: int = 0  # true total tokens ever appended (not retained count)

        # per-step scratch, computed once at layer_idx==0, reused by all layers
        self._current_keep_idx: Optional[torch.Tensor] = None

        # Mechanistic-verification log: one entry per update() call at
        # layer_idx==0, giving the exact retained absolute positions after
        # that step's eviction. This is what makes "system prompt fell out
        # of the window" a verified fact rather than an inference.
        self.position_log: List[Dict[str, Any]] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_new = key_states.shape[-2]
        device = key_states.device

        if layer_idx == 0:
            start = self._seen_tokens
            new_positions = torch.arange(start, start + num_new, dtype=torch.long, device=device)
            old_positions = self._retained_positions
            combined_positions = (
                new_positions if old_positions is None
                else torch.cat([old_positions.to(device), new_positions])
            )
            total = combined_positions.shape[0]

            if total <= self.max_retained:
                keep_idx = torch.arange(total, device=device)  # boundary case: no eviction
            else:
                keep_idx = torch.cat([
                    torch.arange(self.sink_size, device=device),
                    torch.arange(total - self.window_size, total, device=device),
                ])

            self._current_keep_idx = keep_idx
            self._retained_positions = combined_positions.index_select(0, keep_idx)
            self._seen_tokens += num_new

            self.position_log.append({
                "seen_tokens_after": self._seen_tokens,
                "num_new_this_call": num_new,
                "retained_positions": self._retained_positions.detach().cpu().tolist(),
            })

        # every layer applies the SAME keep_idx computed above
        if self.key_cache[layer_idx] is None:
            combined_key, combined_value = key_states, value_states
        else:
            combined_key = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            combined_value = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        keep_idx = self._current_keep_idx
        self.key_cache[layer_idx] = combined_key.index_select(-2, keep_idx)
        self.value_cache[layer_idx] = combined_value.index_select(-2, keep_idx)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_retained