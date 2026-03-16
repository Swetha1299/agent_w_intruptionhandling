"""Sign-token to natural-English conversion."""

from __future__ import annotations

from typing import Sequence


class SignTextConverter:
    """Converts sign tokens into readable English text."""

    _phrase_map = {
        ("HELLO", "HOW", "YOU"): "Hello, how are you?",
        ("HELLO",): "Hello!",
        ("HOW", "YOU"): "How are you?",
        ("YOU",): "You.",
        ("HOW",): "How?",
    }

    def to_text(self, tokens: Sequence[str]) -> str:
        """Convert token list to normalized English sentence."""
        if not tokens:
            return ""

        key = tuple(tokens)
        if key in self._phrase_map:
            return self._phrase_map[key]

        # Fallback grammatical smoothing
        phrase = " ".join(tok.lower() for tok in tokens)
        phrase = phrase.strip()
        if not phrase:
            return ""
        phrase = phrase[0].upper() + phrase[1:]
        if phrase[-1] not in ".!?":
            phrase += "."
        return phrase
