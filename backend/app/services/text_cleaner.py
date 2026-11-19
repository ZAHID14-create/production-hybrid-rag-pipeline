import re


def clean_text(text: str) -> str:
    """
    Cleans and normalizes extracted PDF text while strictly preserving:
    - Meaningful technical terminology (acronyms, code identifiers, model names)
    - Numerical data, percentages, currencies, dates
    - Paragraph and sentence boundaries

    Performs:
    1. Line-break normalization (\r\n and \r -> \n)
    2. Unicode whitespace normalization (non-breaking spaces, em/en spaces -> standard space)
    3. Trimming trailing/leading spaces on each individual line
    4. Collapsing horizontal spaces (consecutive spaces/tabs -> single space)
    5. Collapsing excessive vertical whitespace (3+ consecutive newlines -> double newline)
    """
    if not text:
        return ""

    # 1. Normalize carriage returns to standard linefeeds
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Normalize non-standard Unicode spaces to standard ASCII space (0x20)
    # \u00a0 (NBSP), \u2000-\u200b (various typographic spaces), \u3000 (ideographic space)
    text = re.sub(r"[\u00a0\u2000-\u200b\u3000]", " ", text)

    # 3. Collapse multiple horizontal spaces and tabs into a single space
    text = re.sub(r"[ \t]+", " ", text)

    # 4. Clean up spaces around newlines (trailing/leading per line)
    lines = [line.strip() for line in text.split("\n")]

    # 5. Rejoin and collapse excessive consecutive blank lines (3+ -> 2)
    rejoined = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", rejoined)

    return cleaned.strip()
