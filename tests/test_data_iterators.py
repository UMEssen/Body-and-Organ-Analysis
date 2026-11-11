"""The nnU-Net preprocessing iterators must not fork the Celery worker.

Forking the multithreaded worker process left a Manager child stuck on an
inherited lock, and the task hung while waiting for it.
"""

import multiprocessing.managers
from collections.abc import Callable
from typing import Any
from unittest import mock

import pytest

import body_organ_analysis  # noqa: F401  (puts _external on sys.path)

# isort: split
from nnunetv2.inference import data_iterators


def _fromfiles(n: int) -> Any:
    return data_iterators.preprocessing_iterator_fromfiles(
        [[f"img{i}.nii.gz"] for i in range(n)], None, None, None, {}, None, n
    )


def _fromnpy(n: int) -> Any:
    return data_iterators.preprocessing_iterator_fromnpy(
        [None] * n, None, [{}] * n, None, None, {}, None, n
    )


@pytest.mark.parametrize("make_iterator", [_fromfiles, _fromnpy])
def test_managers_are_spawned(
    make_iterator: Callable[[int], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    spawn_ctx = mock.MagicMock()
    manager = spawn_ctx.Manager.return_value
    # Every worker reports done with an empty queue, so the iterator ends at once.
    manager.Event.return_value.is_set.return_value = True
    manager.Queue.return_value.empty.return_value = True

    def get_context(method: str | None = None) -> Any:
        assert method == "spawn"
        return spawn_ctx

    def no_default_manager(*_: Any, **__: Any) -> None:
        pytest.fail("a Manager was started outside the spawn context")

    monkeypatch.setattr(data_iterators.multiprocessing, "get_context", get_context)
    monkeypatch.setattr(
        multiprocessing.managers.BaseManager, "start", no_default_manager
    )

    assert list(make_iterator(2)) == []
    spawn_ctx.Manager.assert_called_once_with()
    assert spawn_ctx.Process.call_count == 2
