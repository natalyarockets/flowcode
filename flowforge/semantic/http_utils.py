"""Shared HTTP helpers for semantic adapters."""

import time
from typing import Dict, Any, Optional

import requests


def post_with_retries(
    url: str,
    payload: Dict[str, Any],
    *,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
    max_attempts: int = 3,
    backoff: float = 0.6,
) -> requests.Response:
    """Post with simple exponential backoff for transient failures."""

    attempt = 0
    while attempt < max_attempts:
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            attempt += 1
            if attempt >= max_attempts:
                raise
            sleep_time = backoff * (2 ** (attempt - 1))
            time.sleep(sleep_time)
