"""Positional KV cache eviction and safety-alignment collapse.

Modules:
  eviction_cache  - PositionalEvictionCache (StreamingLLM-style sink +
                    sliding-window eviction) and its position-id helper.
  context         - padded-conversation construction with per-segment
                    token-span tracking (build_context).
  generation      - manual prefill/decode loop and cache construction,
                    shared by every sweep condition.
  scoring         - refusal / confirmation-seeking / policy-compliance
                    detection from response text.
  policy_data     - loading policy/prompt CSVs and calibrating sink+window
                    sizes per policy.
  sweep           - orchestrates one prompt bank x one window condition
                    into a results DataFrame, with mechanistic logging of
                    which token positions survived eviction.
"""