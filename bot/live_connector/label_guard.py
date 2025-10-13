from typing import Iterable, Set


class LabelLimiter:
    def __init__(self, whitelist: Iterable[str] | None = None, max_items: int = 100):
        self.whitelist: Set[str] = set(whitelist or [])
        self.max_items = max_items
        self.seen: Set[str] = set()


def allow(self, value: str) -> bool:
    if self.whitelist and value not in self.whitelist:
        return False
    if value in self.seen:
        return True
    if len(self.seen) >= self.max_items:
        return False
    self.seen.add(value)
    return True
