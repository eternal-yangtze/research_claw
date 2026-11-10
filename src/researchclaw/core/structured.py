"""Conservative formatting recovery for a single structured model answer."""
from __future__ import annotations

import json
import re


def parse_single_object(text: str) -> dict:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I)
    parsed, end = json.JSONDecoder().raw_decode(value)
    trailer = value[end:].strip()
    # A stray closing quote/brace is a formatting error, not a second response.
    # Never discard a second JSON object, an explanation, or arbitrary text.
    if trailer and (len(trailer) > 3 or any(c not in '\"}' for c in trailer)):
        raise ValueError("Additional content after the JSON object")
    if not isinstance(parsed, dict):
        raise ValueError("Expected exactly one JSON object")
    return parsed
