from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    ValidationError,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)
from pydantic.types import PathType

ALLOW_NON_EXISTING_PATHS = False


def allow_non_existing_paths(allow: bool) -> None:
    global ALLOW_NON_EXISTING_PATHS
    ALLOW_NON_EXISTING_PATHS = allow


def path_exists_validator(value: Any, handler: ValidatorFunctionWrapHandler) -> Path:
    try:
        return handler(value)
    except ValidationError as e:
        if not ALLOW_NON_EXISTING_PATHS:
            raise e
        return Path(value)


DirectoryPath = Annotated[Path, PathType("dir"), WrapValidator(path_exists_validator)]
FilePath = Annotated[Path, PathType("file"), WrapValidator(path_exists_validator)]
