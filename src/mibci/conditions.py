"""Names of experimental conditions, such as 'relevance-16-prune50'."""

from __future__ import annotations


def parse_condition(condition: str) -> dict:
    """'proposed-8-prune25' -> method proposed, 8 electrodes, 25% of filters pruned."""
    parts = condition.split("-")
    return {
        "method": "full" if condition.startswith("full") else parts[0],
        "electrodes": int(parts[1]),
        "pruned": int(parts[2].removeprefix("prune")) if len(parts) > 2 else 0,
    }
