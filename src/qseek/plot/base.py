from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from matplotlib import pyplot as plt
from pydantic import BaseModel, PositiveFloat, PositiveInt, PrivateAttr

from qseek.models.detection import EventDetection

if TYPE_CHECKING:
    from matplotlib.figure import Figure

INCH = 2.54

logger = logging.getLogger(__name__)


class LassieFigure(BaseModel):
    filename: str
    plot_name: str

    width: PositiveFloat = 8.0
    height: PositiveFloat = 3.0

    nrows: PositiveInt = 1
    ncols: PositiveInt = 1

    _figure: Figure = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._figure = plt.figure(figsize=(self.width / INCH, self.height / INCH))

    def get_axes(self) -> plt.Axes:
        return self._figure.gca()

    def iter_axes(self) -> Iterator[plt.Axes]:
        yield from self._figure.subplots(self.nrows, self.ncols)

    def save(self, dir: Path, dpi: int = 300) -> None:
        filename = dir / self.filename
        logger.info("saving plot to %s", filename)
        self._figure.savefig(str(filename), dpi=dpi)
        plt.close(self._figure)

    def show(self) -> None:
        self._figure.show()
        plt.close(self._figure)


class BasePlot(BaseModel):
    figure: Figure | None = None

    width: PositiveFloat = 20.0
    height: PositiveFloat = 12.0

    nrows: PositiveInt = 1
    ncols: PositiveInt = 1

    default_filename: str

    def new_figure(self, filename: str) -> LassieFigure:
        return LassieFigure(
            filename=filename,
            plot_name=self.__class__.__name__,
            width=self.width,
            height=self.height,
            nrows=self.nrows,
            ncols=self.ncols,
        )

    def get_figure(self) -> LassieFigure:
        raise NotImplementedError

    def get_event_figure(self, event: EventDetection) -> LassieFigure:
        raise NotImplementedError
