from __future__ import annotations

import logging
import re
from typing import Iterable

logger = logging.getLogger(__name__)

CLAIM_NORMALIZER = re.compile(r"\s+")


def clean_claim_text(text: str) -> str:
    """Basic normalization to reduce whitespace and metadata noise."""
    text = CLAIM_NORMALIZER.sub(" ", text).strip()
    text = text.replace(";", ",")
    return text


def generate_negative_prompt(hints: Iterable[str]) -> str:
    """Combine negative prompt hints into a single comma-separated string."""
    return ", ".join(sorted(set(hints)))
