from pathlib import Path

import pytest
from pydantic import ValidationError

from qseek.base import Model
from qseek.types import FilePath, allow_non_existing_paths


def test_model_non_existsing_paths() -> None:
    class TestModelStrict(Model):
        file: FilePath
        b: int = 42

    allow_non_existing_paths(True)
    model = TestModelStrict(
        file=Path("/tmp/non-existing-file"),
    )

    _ = TestModelStrict.model_validate_json(model.model_dump_json())

    allow_non_existing_paths(False)
    with pytest.raises(ValidationError):
        _ = TestModelStrict.model_validate_json(model.model_dump_json())

    allow_non_existing_paths(True)

    class TestModelStrictList(Model):
        files: list[FilePath]
        b: int = 42

    model = TestModelStrictList(
        files=[
            Path("/tmp/non-existing-file1"),
            Path("/tmp/non-existing-file2"),
        ],
    )

    allow_non_existing_paths(False)
    with pytest.raises(ValidationError):
        model = TestModelStrictList(
            files=[
                Path("/tmp/non-existing-file1"),
                Path("/tmp/non-existing-file2"),
            ],
        )
