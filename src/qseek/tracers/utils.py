import logging
from functools import cached_property
from hashlib import sha1
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Any, Literal, Sequence

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FilePath,
    PrivateAttr,
    StringConstraints,
)
from pyrocko import orthodrome as od
from pyrocko.cake import LayeredModel, load_model
from pyrocko.plot.cake_plot import my_model_plot as earthmodel_plot
from typing_extensions import Self

from qseek.models.location import Location
from qseek.models.station import Stations
from qseek.octree import Node, get_node_coordinates
from qseek.utils import CACHE_DIR

logger = logging.getLogger(__name__)

# TODO: Move to a separate file
DEFAULT_VELOCITY_MODEL = """
-1.00    5.50    3.59    2.7
 0.00    5.50    3.59    2.7
 1.00    5.50    3.59    2.7
 1.00    6.00    3.92    2.7
 4.00    6.00    3.92    2.7
 4.00    6.20    4.05    2.7
 8.00    6.20    4.05    2.7
 8.00    6.30    4.12    2.7
13.00    6.30    4.12    2.7
13.00    6.40    4.18    2.7
17.00    6.40    4.18    2.7
17.00    6.50    4.25    2.7
22.00    6.50    4.25    2.7
22.00    6.60    4.31    2.7
26.00    6.60    4.31    2.7
26.00    6.80    4.44    2.7
30.00    6.80    4.44    2.7
30.00    8.10    5.29    2.7
45.00    8.10    5.29    2.7
"""

DEFAULT_VELOCITY_MODEL_FILE = CACHE_DIR / "velocity_models" / "default.nd"
if not DEFAULT_VELOCITY_MODEL_FILE.exists():
    DEFAULT_VELOCITY_MODEL_FILE.parent.mkdir(exist_ok=True)
    DEFAULT_VELOCITY_MODEL_FILE.write_text(DEFAULT_VELOCITY_MODEL)


class LayeredEarthModel1D(BaseModel):
    filename: FilePath | None = Field(
        default=DEFAULT_VELOCITY_MODEL_FILE,
        description="Path to velocity model.",
    )
    format: Literal["nd", "hyposat"] = Field(
        default="nd",
        description="Format of the velocity model. `nd` or `hyposat` is supported.",
    )
    crust2_profile: (
        Annotated[str, StringConstraints(to_upper=True)] | tuple[float, float]
    ) = Field(
        default="",
        description="Crust2 profile name or `[lat, lon]` coordinates.",
    )

    raw_file_data: str | None = Field(
        default=None,
        description="Raw `.nd` file data.",
    )
    _layered_model: LayeredModel = PrivateAttr()

    _raw_file_data: str | None = PrivateAttr(None)

    model_config = ConfigDict(ignored_types=(cached_property,))

    def model_post_init(self, context: Any) -> Self:
        if self.filename is not None and self.raw_file_data is None:
            if self.filename == DEFAULT_VELOCITY_MODEL_FILE:
                logger.warning(
                    "Using default velocity model - Consider specifying a custom model!"
                )
            logger.info("loading velocity model from %s", self.filename)
            self._raw_file_data = self.filename.read_text()

        if self.raw_file_data is not None:
            self._raw_file_data = self.raw_file_data

        if self._raw_file_data is not None:
            with NamedTemporaryFile("w") as tmpfile:
                tmpfile.write(self._raw_file_data)
                tmpfile.flush()
                self._layered_model = load_model(
                    tmpfile.name,
                    format=self.format,
                    crust2_profile=self.crust2_profile or None,
                )
        elif self.crust2_profile:
            self._layered_model = load_model(crust2_profile=self.crust2_profile)
        else:
            raise AttributeError("No velocity model or crust2 profile defined.")
        return self

    @property
    def layered_model(self) -> LayeredModel:
        return self._layered_model

    def fortify(self) -> None:
        """Fortify the model for faster access."""
        self.raw_file_data = self._raw_file_data

    def get_profile_vp(self) -> np.ndarray:
        return self.layered_model.profile("vp")

    def get_profile_vs(self) -> np.ndarray:
        return self.layered_model.profile("vs")

    def get_profile_depth(self) -> np.ndarray:
        return self.layered_model.profile("z")

    def save_plot(self, filename: Path) -> None:
        """Plot the layered model and save the figure to a file.

        Args:
            filename (Path): The path to save the figure.
        """
        import matplotlib.pyplot as plt

        fig = plt.figure()
        earthmodel_plot(self.layered_model, fig=fig)
        fig.savefig(filename, dpi=300)

        logger.info("saved earth model plot to %s", filename)

    @cached_property
    def hash(self) -> str:
        model_serialised = BytesIO()
        for param in ("z", "vp", "vs", "rho"):
            self.layered_model.profile(param).dump(model_serialised)
        return sha1(model_serialised.getvalue()).hexdigest()


def surface_distances(nodes: Sequence[Node], stations: Stations) -> np.ndarray:
    """Returns the surface distance from all nodes to all stations.

    Args:
        nodes (Sequence[Node]): Nodes to calculate distance from.
        stations (Stations): Stations to calculate distance to.

    Returns:
        np.ndarray: Distances in shape (n-nodes, n-stations).
    """
    node_coords = get_node_coordinates(nodes, system="geographic")
    n_nodes = node_coords.shape[0]

    node_coords = np.repeat(node_coords, stations.n_stations, axis=0)
    sta_coords = np.vstack(n_nodes * [stations.get_coordinates(system="geographic")])

    return od.distance_accurate50m_numpy(
        node_coords[:, 0], node_coords[:, 1], sta_coords[:, 0], sta_coords[:, 1]
    ).reshape(-1, stations.n_stations)


def surface_distances_reference(
    nodes: Sequence[Node],
    stations: Stations,
    reference: Location,
) -> np.ndarray:
    """Returns the surface distance from all nodes to all stations.

    Args:
        nodes (Sequence[Node]): Nodes to calculate distance from.
        stations (Stations): Stations to calculate distance to.
        reference (Location): Reference location for the nodes.

    Returns:
        np.ndarray: Distances in shape (n-nodes, n-stations).
    """
    node_offsets = np.asarray([(node.east, node.north) for node in nodes])
    n_nodes = node_offsets.shape[0]
    result = np.empty((stations.n_stations, n_nodes), dtype=float)

    for idx, station in enumerate(stations):
        offset_east, offset_north, _ = station.offset_from(reference)
        distances = np.linalg.norm(
            node_offsets - np.array([offset_east, offset_north]), axis=1
        )
        result[idx] = distances
    return result.T.copy()
