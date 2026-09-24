import os
from collections.abc import Iterator
from unittest import mock

import pytest


@pytest.fixture
def env(request: pytest.FixtureRequest) -> Iterator[None]:
    """Run the test with only the environment given via indirect parametrization."""
    with mock.patch.dict(os.environ, getattr(request, "param", {}), clear=True):
        yield
