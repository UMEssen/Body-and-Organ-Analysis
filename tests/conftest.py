_MODULE_ORDER = {"test_cli": 0, "test_results": 2}


def pytest_collection_modifyitems(config, items):
    """Pin test_cli before everything else and test_results last (cross-module deps)."""
    items.sort(
        key=lambda i: _MODULE_ORDER.get(i.module.__name__.rsplit(".", 1)[-1], 1)
    )
