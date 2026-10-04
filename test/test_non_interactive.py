from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from qseek import console as console_module
from qseek.apps.qseek import UsageError, summarize_rundir
from qseek.search import SearchProgress
from qseek.utils import LogCounter

if TYPE_CHECKING:
    from pathlib import Path


def test_report(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    monkeypatch.setattr(console_module, "NON_INTERACTIVE", False)
    console_module.report("key", "value")
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(console_module, "NON_INTERACTIVE", True)
    console_module.report("key", "value")
    assert capsys.readouterr().out == "key: value\n"


def test_log_counter():
    counter = LogCounter()
    logger = logging.getLogger("test_log_counter")
    logger.propagate = False
    logger.addHandler(counter)
    logger.setLevel(logging.INFO)

    logger.info("not counted")
    logger.warning("warning")
    logger.error("error")
    logger.error("error")
    assert (counter.warnings, counter.errors) == (1, 2)


def test_progress_loads_old_file():
    progress = SearchProgress.model_validate_json(
        '{"time_progress": "2024-05-20T23:55:11.948485Z"}'
    )
    assert progress.time_progress == datetime(
        2024, 5, 20, 23, 55, 11, 948485, tzinfo=timezone.utc
    )
    assert progress.percent == 0.0
    assert progress.n_events == 0


def test_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
):
    monkeypatch.setattr(console_module, "NON_INTERACTIVE", True)
    with pytest.raises(UsageError):
        summarize_rundir(tmp_path)

    (tmp_path / "search.json").write_text("{}")
    (tmp_path / "progress.json").write_text(
        SearchProgress(n_events=3).model_dump_json()
    )
    summarize_rundir(tmp_path)
    out = capsys.readouterr().out
    assert "detections: 3" in out
    assert "status: incomplete" in out

    (tmp_path / "results.json").write_text("{}")
    summarize_rundir(tmp_path)
    assert "status: finished" in capsys.readouterr().out
