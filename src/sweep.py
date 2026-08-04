"""Runs a prompt bank through the eviction harness at a given sink+window
condition, and records mechanistic ground truth (which positions survived
eviction) alongside each response.
"""

from typing import List, Optional

import pandas as pd
from tqdm.auto import tqdm

from .context import build_context
from .generation import build_cache, generate_with_cache
from .scoring import is_refusal


def _first_step_system_prompt_lost(cache, sys_start: int, sys_end: int, sink_size: int) -> Optional[int]:
    """Decode-step index (into cache.position_log) at which system-prompt
    positions BEYOND the sink first fully leave the retained set, or None
    if they never do (survives through every logged step, or there's no
    log / the sink already covers the whole system prompt).

    frac_system_prompt_retained only looks at the FINAL step, so "lost
    partway through generation" and "never had it in the first place" both
    read as the same end state. This distinguishes them.
    """
    log = getattr(cache, "position_log", None)
    if not log:
        return None
    beyond_sink = set(range(max(sys_start, sink_size), sys_end))
    if not beyond_sink:
        return None
    for step_idx, entry in enumerate(log):
        if not (beyond_sink & set(entry["retained_positions"])):
            return step_idx
    return None


def run_eviction_sweep(
    model,
    tokenizer,
    harmful_prompts: List[str],
    sink_size: int,
    window_size,
    condition_name: str,
    target_filler_tokens: int = 3000,
    max_new_tokens: int = 200,
    system_prompt: str = None,
    omit_system_prompt: bool = False,
    prompt_type: str = None,
) -> pd.DataFrame:
    """Runs every prompt in `harmful_prompts` through one sink/window
    condition and returns a DataFrame with one row per prompt.

    prompt_type: free-form tag ("policy" / "neutral" / None) stored per
    row. Not used by this function's own logic -- purely so downstream
    code can tell which bank a row came from after concatenating results.
    """
    results = []
    pbar = tqdm(harmful_prompts, desc=condition_name)
    for idx, prompt in enumerate(pbar):
        messages, offsets = build_context(prompt, tokenizer, target_filler_tokens=target_filler_tokens,
                                           system_prompt=system_prompt,
                                           omit_system_prompt=omit_system_prompt)
        cache = build_cache(sink_size, window_size, num_layers=model.config.num_hidden_layers)
        response = generate_with_cache(model, tokenizer, messages, cache, max_new_tokens=max_new_tokens)
        refused = is_refusal(response)

        sys_start, sys_end = offsets["system_prompt"]
        has_log = getattr(cache, "position_log", None)
        if has_log:
            retained = cache.position_log[-1]["retained_positions"]
        else:  # DynamicCache -- nothing ever evicted, everything retained
            retained = list(range(offsets["total_prompt_tokens"]))
        retained_set = set(retained)
        system_prompt_positions = set(range(sys_start, sys_end))
        system_prompt_survived = bool(system_prompt_positions & retained_set)
        frac_system_prompt_retained = len(system_prompt_positions & retained_set) / max(1, len(system_prompt_positions))
        system_prompt_lost_at_step = _first_step_system_prompt_lost(cache, sys_start, sys_end, sink_size)

        results.append({
            "idx": idx, "harmful_prompt": prompt, "response": response, "is_refusal": refused,
            "condition": condition_name, "sink_size": sink_size, "window_size": window_size,
            "system_prompt_span": offsets["system_prompt"], "total_prompt_tokens": offsets["total_prompt_tokens"],
            "cache_length_final": cache.get_seq_length(0),
            "system_prompt_present": not omit_system_prompt,
            "system_prompt_survived": system_prompt_survived,
            "frac_system_prompt_retained": frac_system_prompt_retained,
            "system_prompt_lost_at_step": system_prompt_lost_at_step,
            "prompt_type": prompt_type,
        })
        running_rate = sum(r["is_refusal"] for r in results) / len(results)
        pbar.set_postfix(refusal_rate=f"{running_rate:.0%}")
    return pd.DataFrame(results)