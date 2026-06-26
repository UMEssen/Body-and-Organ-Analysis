"""Track cross-module test dependencies within a single test run."""

_attempted: set[str] = set()
_completed: set[str] = set()


def mark_attempted(name: str) -> None:
    _attempted.add(name)


def mark_complete(name: str) -> None:
    _completed.add(name)


def has_attempted(name: str) -> bool:
    return name in _attempted


def has_completed(name: str) -> bool:
    return name in _completed
