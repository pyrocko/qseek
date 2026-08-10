from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from qseek.base import Model

if TYPE_CHECKING:
    from qseek.models.detection import EventDetection
    from qseek.search import Search
    from qseek.waveforms.base import WaveformBatch


class Callback(Model):
    """Base class for search lifecycle callbacks."""

    callback: Literal["Callback"] = "Callback"

    async def on_start(self, search: Search) -> None:
        """Called once after the search has been prepared."""

    async def on_stop(self, search: Search) -> None:
        """Called once after the search has finished."""

    async def on_batch_start(self, batch: WaveformBatch) -> None:
        """Called before a waveform batch is processed."""

    async def on_batch_end(self, batch: WaveformBatch) -> None:
        """Called after a waveform batch has been processed."""

    async def on_new_detection(self, detection: EventDetection) -> None:
        """Called for every new event detection."""

    @classmethod
    def get_subclasses(cls) -> tuple[type[Callback], ...]:
        """Get the subclasses of this class.

        Returns:
            tuple[type[Callback], ...]: The subclasses of this class.
        """
        return tuple(cls.__subclasses__())
