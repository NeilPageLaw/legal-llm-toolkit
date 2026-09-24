"""
Prompt templates for turning legal samples into training text.
"""

ALPACA_WITH_CONTEXT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{context}\n\n### Response:\n"
)

ALPACA_NO_CONTEXT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:\n"
)

TEMPLATES = ("alpaca", "chatml")


def to_instruction_format(
    instruction: str,
    response: str | None = None,
    context: str | None = None,
    template: str = "alpaca",
    system: str | None = None,
) -> str:
    """
    Format an instruction-tuning example as a single training string.

    Args:
        instruction: The task, e.g. "Identify the governing law clause."
        response: The target answer. Omit it to build a prompt for generation.
        context: Optional supporting text, e.g. the contract or judgment extract.
        template: "alpaca", "chatml", or a custom format string using any of
            the placeholders {instruction}, {context}, {response} and {system}.
        system: Optional system message (used by "chatml" and custom templates).

    Returns:
        The formatted text. Without a response the text ends where the model
        should start generating.

    Example:
        >>> print(to_instruction_format("Name the parties.", "Acme Ltd and Beta LLP"))
        Below is an instruction that describes a task. ...
    """
    if template == "alpaca":
        if context:
            prompt = ALPACA_WITH_CONTEXT.format(instruction=instruction, context=context)
        else:
            prompt = ALPACA_NO_CONTEXT.format(instruction=instruction)
        return prompt + (response or "")

    if template == "chatml":
        parts = []
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>\n")
        user = f"{instruction}\n\n{context}" if context else instruction
        parts.append(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
        if response is not None:
            parts.append(f"{response}<|im_end|>")
        return "".join(parts)

    if "{instruction}" in template:
        return template.format(
            instruction=instruction,
            context=context or "",
            response=response or "",
            system=system or "",
        )

    raise ValueError(
        f"Unknown template: {template!r}. Use one of {TEMPLATES} "
        "or a format string containing {instruction}."
    )
