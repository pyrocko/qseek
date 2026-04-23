from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_validator,
)

ALLOW_NON_EXISTING_PATHS = False


def allow_non_existing_paths(allow: bool) -> None:
    global ALLOW_NON_EXISTING_PATHS
    ALLOW_NON_EXISTING_PATHS = allow


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="wrap")
    @classmethod
    def _check_paths(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        try:
            return handler(value)
        except ValidationError as e:
            if ALLOW_NON_EXISTING_PATHS:
                errors = e.errors()
                for error in errors:
                    if error["type"] in (
                        "path_not_file",
                        "path_not_dir",
                        "path_not_exists",
                    ):
                        if isinstance(value, list):
                            return [Path(v) for v in value]
                        return Path(value)
            raise e
