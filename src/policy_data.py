"""Loading policy/prompt banks from CSV, and sizing the sink+window
conditions for each policy off its own system-prompt length.
"""

import os
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .context import build_context


def load_policy_csvs(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads policies.csv / policy_prompts.csv from data_dir and validates
    that every policy_id referenced in policy_prompts.csv exists in
    policies.csv.
    """
    policies_df = pd.read_csv(os.path.join(data_dir, "policies.csv"))
    prompts_df = pd.read_csv(os.path.join(data_dir, "policy_prompts.csv"))
    assert set(prompts_df["policy_id"]) <= set(policies_df["policy_id"]), (
        "policy_prompts.csv references a policy_id not present in policies.csv"
    )
    return policies_df, prompts_df


def get_policy_bundle(
    policy_id: str,
    policies_df: pd.DataFrame,
    prompts_df: pd.DataFrame,
    n_samples: Optional[int] = None,
) -> Tuple[str, List[str], List[str]]:
    """Returns (system_prompt, policy_prompts, neutral_prompts) for
    policy_id.

    n_samples, if given, takes an evenly-spaced subset of pair_ids (not
    just the first n_samples rows) so a small n_samples still spans the
    full domain -- e.g. for destructive_ops, the first 20 rows are all
    "git" category by construction order; a flat prefix slice would
    silently test only one category instead of a representative
    cross-section.
    """
    sys_prompt = policies_df.loc[policies_df.policy_id == policy_id, "system_prompt"].iloc[0]
    sub = prompts_df[prompts_df.policy_id == policy_id]
    pair_ids = sub["pair_id"].drop_duplicates().tolist()
    if n_samples is not None and n_samples < len(pair_ids):
        stride = len(pair_ids) / n_samples
        pair_ids = [pair_ids[int(i * stride)] for i in range(n_samples)]
        sub = sub[sub["pair_id"].isin(pair_ids)]
    policy_list = sub[sub.prompt_type == "policy"].sort_values("pair_id")["prompt_text"].tolist()
    neutral_list = sub[sub.prompt_type == "neutral"].sort_values("pair_id")["prompt_text"].tolist()
    return sys_prompt, policy_list, neutral_list


def compute_policy_window_conditions(
    policy_id: str,
    system_prompt: str,
    policy_prompts: List[str],
    neutral_prompts: List[str],
    tokenizer,
    sink_size: int,
    safety_margin: int,
    max_new_tokens: int,
    target_filler_tokens: int = 3000,
    spread_threshold: int = 20,
) -> Optional[Dict]:
    """Homogeneity check + probe + window-condition sizing for one policy.

    Returns a config dict, or None (with a printed warning, not a raised
    exception) if this policy's prompt bank fails validation -- with
    multiple policies, one bad bank shouldn't block the others from
    running.
    """
    lens = []
    for p in policy_prompts + neutral_prompts:
        _, offs = build_context(p, tokenizer, target_filler_tokens=target_filler_tokens,
                                 system_prompt=system_prompt, shuffle_filler=False)
        lens.append(offs["total_prompt_tokens"])
    spread = max(lens) - min(lens)
    print(f"[{policy_id}] prompt-bank token-length spread: {spread} tokens (min={min(lens)}, max={max(lens)})")
    if spread >= spread_threshold:
        print(f"[{policy_id}] WARNING: spread={spread} >= {spread_threshold} -- too wide to safely "
              f"share one window config. EXCLUDING {policy_id} from the sweep.")
        return None

    _, probe_offsets = build_context(policy_prompts[0], tokenizer, target_filler_tokens=target_filler_tokens,
                                      system_prompt=system_prompt, shuffle_filler=False)
    sys_start, sys_end = probe_offsets["system_prompt"]
    sys_len = sys_end - sys_start
    req_start, req_end = probe_offsets["harmful_request"]
    req_len = req_end - req_start
    total_len = probe_offsets["total_prompt_tokens"]
    reach_needed = total_len - sys_end

    # Sized against the reach needed AFTER a full max_new_tokens decode, not
    # just after prefill -- a window sized only off prefill numbers has its
    # margin consumed well before generation ends, since eviction re-fires
    # on every decode step.
    large_window = reach_needed + max_new_tokens + safety_margin
    if large_window + sink_size >= total_len + max_new_tokens:
        print(f"[{policy_id}] WARNING: large_window={large_window} would make max_retained >= "
              f"total_len+max_new_tokens ({total_len + max_new_tokens}); eviction would never fire "
              f"for 'large' even during decode. EXCLUDING {policy_id} from the sweep.")
        return None

    window_conditions = {
        "no_eviction": None,
        "large": large_window,
        "medium": reach_needed + max_new_tokens // 2,
        "small": max(16, sys_len // 2),
    }

    print(f"[{policy_id}] system prompt spans [{sys_start}, {sys_end}) -> length={sys_len}, "
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
        "policy_prompts": policy_prompts,
        "neutral_prompts": neutral_prompts,
        "sys_len": sys_len, "total_len": total_len, "reach_needed": reach_needed,
        "window_conditions": window_conditions,
    }