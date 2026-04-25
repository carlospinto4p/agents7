"""Human-like typo injection for chat responses.

Introduces subtle, realistic typos into text to make
agent responses feel more natural and human.

Typo types:
- Swap two adjacent characters (e.g., "the" → "teh")
- Double a character (e.g., "hello" → "helllo")
- Drop a character (e.g., "world" → "wrld")
- Miss a space between words (e.g., "is a" → "isa")
"""

import random

# Minimum word length to apply a typo (short words
# like "a", "is", "to" are left alone).
_MIN_TYPO_WORD_LEN = 4


def apply_typos(
    text: str,
    rate: float,
    rng: random.Random | None = None,
) -> str:
    """Introduce random typos into text.

    :param text: Original text.
    :param rate: Probability (0.0–1.0) that each eligible
        word gets a typo. 0.0 means no typos, 0.05 means
        ~5% of words are affected.
    :param rng: Optional ``Random`` instance for
        reproducibility. Uses ``random`` module if None.
    :return: Text with occasional typos injected.
    """
    if rate <= 0:
        return text
    if rng is None:
        rng = random.Random()

    words = text.split(" ")
    result = []
    for i, word in enumerate(words):
        if len(word) >= _MIN_TYPO_WORD_LEN and rng.random() < rate:
            word = _inject_typo(word, i, len(words), rng)
        result.append(word)
    return " ".join(result)


def _inject_typo(
    word: str,
    word_idx: int,
    total_words: int,
    rng: random.Random,
) -> str:
    """Apply a single random typo to a word.

    Picks from: swap adjacent, double char, drop char,
    or miss space (merge with next word placeholder).
    """
    # Separate any trailing punctuation
    core = word
    trailing = ""
    while core and not core[-1].isalnum():
        trailing = core[-1] + trailing
        core = core[:-1]

    if len(core) < _MIN_TYPO_WORD_LEN:
        return word

    typo_type = rng.choice(["swap", "double", "drop"])

    if typo_type == "swap":
        # Swap two adjacent characters (not first char
        # to avoid capitalization issues)
        idx = rng.randint(1, len(core) - 2)
        chars = list(core)
        chars[idx], chars[idx + 1] = (
            chars[idx + 1],
            chars[idx],
        )
        core = "".join(chars)

    elif typo_type == "double":
        # Double a random character
        idx = rng.randint(1, len(core) - 1)
        core = core[:idx] + core[idx] + core[idx:]

    elif typo_type == "drop":
        # Drop a random character (not first/last)
        if len(core) > _MIN_TYPO_WORD_LEN:
            idx = rng.randint(1, len(core) - 2)
            core = core[:idx] + core[idx + 1 :]

    return core + trailing
