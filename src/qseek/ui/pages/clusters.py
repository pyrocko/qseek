from nicegui import ui

from qseek.ui.analysis.cluster import ClusterDBScan, labels_to_colors
from qseek.ui.components.magnitudes import MagnitudeRate
from qseek.ui.components.map import OverviewMap
from qseek.ui.state import get_tab_state
from qseek.ui.utils import stat_card


async def clusters_page() -> None:
    state = get_tab_state()
    catalog = await state.get_catalog()

    epsilon = 1000  # in meters
    min_samples = 30

    clustering = ClusterDBScan(catalog, epsilon=1000, min_samples=30)
    with state.loading_message("Calculating clusters..."):
        labels = await clustering.get_labels()

    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = (labels == -1).sum()
    n_clustered = catalog.n_events - n_noise

    cluster_colors = labels_to_colors(labels)

    with ui.row().classes("w-full items-stretch"):
        stat_card(
            "Clusters",
            str(n_clusters),
            icon="spoke",
            subtitle=f"{n_clustered} events clustered "
            f"({n_clustered / catalog.n_events * 100:.1f}%)",
            tooltip="Number of clusters identified by DBSCAN.",
        )
        stat_card(
            "Noise Events",
            str(n_noise),
            icon="close",
            subtitle="Not in any cluster",
            tooltip="Number of events classified as noise"
            "(not belonging to any cluster).",
        )
        stat_card(
            "Epsilon",
            f"{epsilon} m",
            icon="square_foot",
            subtitle="Neighborhood radius",
            tooltip="Maximum distance between two samples for them to be considered as "
            "in the same neighborhood.",
        )
        stat_card(
            "Min Samples",
            str(min_samples),
            icon="people",
            subtitle="Min samples in neighborhood",
            tooltip="Minimum number of samples in a neighborhood for a point to be "
            "considered as a core point.",
        )

    with ui.row().classes("w-full flex-1 items-stretch"):
        with ui.card().classes("col-12"):
            overview = OverviewMap(catalog)
            overview.header(
                title="Spatial Clusters of Detected Events",
                description=(
                    "DBSCAN spatial clustering of seismicity catalog. Clusters are "
                    "colored by label, with noise events shown in gray. "
                ),
            )
            await overview.view(marker_colors=cluster_colors)
        with ui.card().classes("col-12"):
            rate = MagnitudeRate(catalog)
            rate.header()
            await rate.view(
                show_semblance=not catalog.has_magnitudes(),
                show_density=False,
                marker_colors=cluster_colors,
            )
