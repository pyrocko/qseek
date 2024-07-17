from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Exporter(BaseModel):
    async def export(self, rundir: Path, outdir: Path) -> Path:
        raise NotImplementedError

    @classmethod
    def get_subclasses(cls) -> tuple[type[Exporter], ...]:
        """Get the subclasses of this class.

        Returns:
            list[type]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())
