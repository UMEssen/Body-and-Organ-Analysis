import pytest

_MODULE_ORDER = {"test_cli": 0, "test_results": 2}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Pin test_cli before everything else and test_results last (cross-module deps)."""
    del config

    def order(item: pytest.Item) -> int:
        mod = getattr(item, "module", None)
        name = mod.__name__.rsplit(".", 1)[-1] if mod is not None else ""
        return _MODULE_ORDER.get(name, 1)

    items.sort(key=order)
