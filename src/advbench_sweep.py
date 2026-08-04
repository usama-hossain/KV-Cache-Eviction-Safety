"""AdvBench-based RLHF-refusal sweep.

This is a sibling to policy_data.py, not a modification of it: it reuses
build_context / run_eviction_sweep as-is, but sizes window conditions for a
single flat prompt bank (no policy/neutral pairing)
"""

from typing import Dict, List, Optional

import pandas as pd

from .context import build_context

ADVBENCH_URL = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"


def load_advbench(n_samples: int = 100, seed: int = 0) -> List[str]:
    """Downloads AdvBench's harmful_behaviors.csv and returns an
    n_samples-sized random subset of the `goal` column (all of it if
    n_samples >= the full set).
    """
    df = pd.read_csv(ADVBENCH_URL)
    if n_samples < len(df):
        df = df.sample(n=n_samples, random_state=seed).reset_index(drop=True)
    prompts = df["goal"].tolist()
    print(f"[advbench] loaded {len(prompts)} prompts (seed={seed})")
    return prompts


def compute_advbench_window_conditions(
    system_prompt: str,
    prompts: List[str],
    tokenizer,
    sink_size: int,
    safety_margin: int,
    max_new_tokens: int,
    target_filler_tokens: int = 3000,
    spread_threshold: int = 20,
) -> Optional[Dict]:
    """Same sizing logic as policy_data.compute_policy_window_conditions,
    adapted for a single prompt bank instead of a policy/neutral pair.

    Returns a config dict, or None (with a printed warning, not a raised
    exception) if the bank fails validation.
    """
    lens = []
    for p in prompts:
        _, offs = build_context(p, tokenizer, target_filler_tokens=target_filler_tokens,
                                 system_prompt=system_prompt, shuffle_filler=False)
        lens.append(offs["total_prompt_tokens"])
    spread = max(lens) - min(lens)
    print(f"[advbench] prompt-bank token-length spread: {spread} tokens (min={min(lens)}, max={max(lens)})")
    if spread >= spread_threshold:
        print(f"[advbench] WARNING: spread={spread} >= {spread_threshold} -- too wide to safely "
              f"share one window config. EXCLUDING advbench from the sweep.")
        return None

    _, probe_offsets = build_context(prompts[0], tokenizer, target_filler_tokens=target_filler_tokens,
                                      system_prompt=system_prompt, shuffle_filler=False)
    sys_start, sys_end = probe_offsets["system_prompt"]
    sys_len = sys_end - sys_start
    req_start, req_end = probe_offsets["harmful_request"]
    req_len = req_end - req_start
    total_len = probe_offsets["total_prompt_tokens"]
    reach_needed = total_len - sys_end

    # Sized against the reach needed AFTER a full max_new_tokens decode, same
    # rationale as the policy sizing: a window sized only off prefill numbers
    # has its margin consumed well before generation ends.
    large_window = reach_needed + max_new_tokens + safety_margin
    if large_window + sink_size >= total_len + max_new_tokens:
        print(f"[advbench] WARNING: large_window={large_window} would make max_retained >= "
              f"total_len+max_new_tokens ({total_len + max_new_tokens}); eviction would never fire "
              f"for 'large' even during decode. EXCLUDING advbench from the sweep.")
        return None

    window_conditions = {
        "no_eviction": None,
        "large": large_window,
        "medium": reach_needed + max_new_tokens // 2,
        "small": max(16, sys_len // 2),
    }

    print(f"[advbench] system prompt spans [{sys_start}, {sys_end}) -> length={sys_len}, "
          f"total_len={total_len}, reach_needed={reach_needed}")
    for name, w in window_conditions.items():
        max_retained = (sink_size + w) if w is not None else None
        fires_gen = (w is not None) and (max_retained < total_len + max_new_tokens)
        fires_prefill = (w is not None) and (max_retained < total_len)
        print(f"  {name:12s}: window={w}"
              + (f"  max_retained={max_retained}  fires_during_generation={fires_gen}  "
                 f"fires_during_prefill={fires_prefill}" if w is not None else "  (no cap)"))
        if w is not None and w < req_len + 5:
            print(f"    WARNING: window={w} close to/below req_len={req_len} -- "
                  f"the request itself risks falling out of the window.")

    return {
        "system_prompt": system_prompt,
        "prompts": prompts,
        "sys_len": sys_len, "total_len": total_len, "reach_needed": reach_needed,
        "window_conditions": window_conditions,
    }