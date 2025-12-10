import json
from typing import Optional


def strip_code_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def safe_json_extract(text: str) -> Optional[str]:
    """
    Try to extract the first JSON object from arbitrary text.
    - Strips code fences if present
    - Finds the first '{' and last '}' and returns the substring
    - Validates it's JSON; returns None if not valid
    """
    if not text:
        return None
    s = strip_code_fences(text)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = s[start : end + 1].strip()
    try:
        json.loads(candidate)
    except Exception:
        return None
    return candidate


