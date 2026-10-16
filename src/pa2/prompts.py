"""The single prompt format shared by SFT, DPO, and generation.

We use Qwen's ChatML layout written out explicitly (rather than relying on
``tokenizer.apply_chat_template``) so the exact string is fixed in code and
documented in the report:

    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    {response}<|im_end|>

The response is followed by ``tokenizer.eos_token``. ``load_tokenizer`` sets
that to ``<|im_end|>`` for Qwen so the model learns to close its turn, and
generation stops on it.
"""

from __future__ import annotations

SYSTEM_PROMPT = "You are a helpful assistant."

CHATML_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

IM_END = "<|im_end|>"
ENDOFTEXT = "<|endoftext|>"


def format_prompt(instruction: str, system: str = SYSTEM_PROMPT) -> str:
    """Wrap a raw user instruction in the ChatML prompt (ends at the assistant header)."""

    return CHATML_TEMPLATE.format(system=system, instruction=instruction.strip())


def load_tokenizer(model_name_or_path: str):
    """Load a tokenizer configured for this project's prompt format.

    For Qwen-style tokenizers, EOS becomes ``<|im_end|>`` (end of assistant turn)
    and padding uses ``<|endoftext|>``. Padding never affects the loss because
    labels and log-probs are masked by position, not by token id.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    vocab = tokenizer.get_vocab()
    if IM_END in vocab:
        tokenizer.eos_token = IM_END
        if ENDOFTEXT in vocab:
            tokenizer.pad_token = ENDOFTEXT
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def stop_token_ids(tokenizer) -> list[int]:
    """Token ids that should end generation (EOS plus ChatML/endoftext markers if present)."""

    ids = []
    vocab = tokenizer.get_vocab()
    for token in (tokenizer.eos_token, IM_END, ENDOFTEXT):
        if token is not None and token in vocab and vocab[token] not in ids:
            ids.append(vocab[token])
    return ids
