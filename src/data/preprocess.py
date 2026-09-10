"""
Text Preprocessing & Tokenisation
==================================
Provides ``clean_text`` for normalising raw claim strings and a
``Tokenizer`` class that wraps a HuggingFace ``AutoTokenizer`` for
batch tokenisation with padding / truncation.

Usage::

    from src.data.preprocess import clean_text, Tokenizer

    tokenizer = Tokenizer("roberta-base", max_length=512)

    text = clean_text("<b>Some &amp; claim</b>  ")
    tokens = tokenizer.tokenize([text])          # dict of (1, 512) tensors
    tokenizer.save_tokens(tokens, "cache.pt")
    loaded = Tokenizer.load_tokens("cache.pt")
"""
from __future__ import annotations

import html
import math
import re
from pathlib import Path
from typing import Optional, Union

import torch
from transformers import AutoTokenizer

from src.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: Optional[str]) -> str:
    """Normalise a raw claim string.

    Steps (in order):

    1. Coerce ``None`` / NaN to an empty string.
    2. Decode HTML entities (``&amp;`` → ``&``, ``&lt;`` → ``<``, etc.).
    3. Strip HTML / XML tags.
    4. Collapse consecutive whitespace (spaces, tabs, newlines) to a single space.
    5. Strip leading / trailing whitespace.

    Args:
        text: Raw input string, or ``None`` / NaN.

    Returns:
        Cleaned string; empty string if input is ``None`` or NaN.
    """
    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    text = str(text)
    text = html.unescape(text)           # &amp; → &
    text = re.sub(r"<[^>]+>", " ", text) # strip HTML tags
    text = re.sub(r"\s+", " ", text)     # normalise whitespace
    return text.strip()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class Tokenizer:
    """Thin wrapper around ``transformers.AutoTokenizer``.

    Handles batch tokenisation with padding and truncation, and provides
    helpers for caching tokenised tensors to disk.

    Args:
        model_name: HuggingFace model identifier (e.g. ``"roberta-base"``).
        max_length: Maximum sequence length for truncation / padding.
    """

    def __init__(self, model_name: str, max_length: int = 512) -> None:
        self.model_name = model_name
        self.max_length = max_length
        log.info(f"Loading tokenizer: {model_name}")
        self._tok = AutoTokenizer.from_pretrained(model_name)
        log.info(f"Tokenizer ready (max_length={max_length})")

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------

    def tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Batch-tokenise *texts* and return padded / truncated tensors.

        Args:
            texts: List of (already-cleaned) strings to tokenise.

        Returns:
            Dict with at minimum ``"input_ids"`` and ``"attention_mask"``
            tensors of shape ``(len(texts), max_length)``.
        """
        encoding = self._tok(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return dict(encoding)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def save_tokens(
        self,
        tokens: dict[str, torch.Tensor],
        path: Union[str, Path],
    ) -> None:
        """Save a tokenised-tensor dict to disk.

        Args:
            tokens: Output of :meth:`tokenize`.
            path:   Destination file path (``.pt``). Parent dirs are created
                    automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tokens, path)
        log.info(f"Saved token cache → {path}")

    @staticmethod
    def load_tokens(path: Union[str, Path]) -> dict[str, torch.Tensor]:
        """Load a tokenised-tensor dict from disk.

        Args:
            path: Path previously written by :meth:`save_tokens`.

        Returns:
            Dict of tensors, same structure as :meth:`tokenize` output.

        Raises:
            FileNotFoundError: If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Token cache not found: {path}")
        return torch.load(path, map_location="cpu", weights_only=False)
