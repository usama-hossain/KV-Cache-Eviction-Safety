"""Manual prefill+decode generation loop, and cache construction, shared by
every arm of the sweep -- the no-eviction control and every eviction
condition run through identical harness code.
"""

import torch
from transformers.cache_utils import DynamicCache

from .eviction_cache import PositionalEvictionCache, compute_position_ids


def build_cache(sink_size: int, window_size, num_layers: int):
    """window_size=None -> unmodified DynamicCache (no-eviction control).
    Otherwise -> PositionalEvictionCache with the given sink/window sizes.
    """
    if window_size is None:
        return DynamicCache()
    return PositionalEvictionCache(
        num_layers=num_layers,
        sink_size=sink_size,
        window_size=window_size,
    )


@torch.no_grad()
def generate_with_cache(model, tokenizer, messages, cache, max_new_tokens: int = 200) -> str:
    """Works with EITHER a plain DynamicCache or PositionalEvictionCache, so
    the no-eviction control and every eviction condition run through
    identical harness code. Bypasses model.generate() to keep explicit
    control over absolute position_ids under eviction.
    """
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    input_ids = enc["input_ids"]
    seen = 0
    generated_ids = []

    # ---- prefill: whole prompt in one forward call ----
    prompt_len = input_ids.shape[1]
    position_ids = compute_position_ids(seen, prompt_len, device=model.device).unsqueeze(0)
    attention_mask = torch.ones((1, prompt_len), dtype=torch.long, device=model.device)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=position_ids[0],
        past_key_values=cache,
        use_cache=True,
    )
    seen += prompt_len
    next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    if next_token.item() != tokenizer.eos_token_id:
        generated_ids.append(next_token.item())

    # ---- decode: one token at a time ----
    for _ in range(max_new_tokens - 1):
        retained_len = cache.get_seq_length(0)  # physical cache length AFTER previous step's trim
        position_ids = compute_position_ids(seen, 1, device=model.device).unsqueeze(0)
        attention_mask = torch.ones((1, retained_len + 1), dtype=torch.long, device=model.device)
        out = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=position_ids[0],
            past_key_values=cache,
            use_cache=True,
        )
        seen += 1
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        tok_id = next_token.item()
        if tok_id == tokenizer.eos_token_id:
            break
        generated_ids.append(tok_id)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)