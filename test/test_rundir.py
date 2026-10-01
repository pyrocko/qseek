from __future__ import annotations

import logging
from pathlib import Path

import pytest

from qseek.search import Search


@pytest.fixture
def search(tmp_path: Path) -> Search:
    search = Search(project_dir=tmp_path)
    search._config_stem = "my-search"
    return search


def _close_log_handlers(rundir: Path) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler) and Path(
            handler.baseFilename
        ).is_relative_to(rundir.parent):
            root.removeHandler(handler)
            handler.close()


def test_init_rundir(search: Search, tmp_path: Path) -> None:
    rundir = tmp_path / "my-search"
    try:
        search.init_rundir()
        assert rundir.is_dir()
        (rundir / "marker").touch()

        with pytest.raises(FileExistsError):
            search.init_rundir()

        # --force: keep the old run as a backup
        search.init_rundir(force=True)
        backups = list(tmp_path.glob("my-search.bak-*"))
        assert len(backups) == 1
        assert (backups[0] / "marker").exists()
        assert not (rundir / "marker").exists()

        # --force --no-backup: overwrite the old run
        (rundir / "marker").touch()
        search.init_rundir(force=True, create_backup=False)
        assert rundir.is_dir()
        assert not (rundir / "marker").exists()
        assert not (tmp_path / "my-search-del").exists()
        assert len(list(tmp_path.glob("my-search.bak-*"))) == 1
    finally:
        _close_log_handlers(rundir)
