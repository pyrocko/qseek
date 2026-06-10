from nicegui import ui

from qseek.ui.cluster import ClusterDBScan, labels_to_colors
from qseek.ui.components.magnitudes import MagnitudeRate
from qseek.ui.components.map import OverviewMap
from qseek.ui.state import get_tab_state


async def clusters_page() -> None:
    state = get_tab_state()
    catalog = await state.get_catalog()

    clustering = ClusterDBScan(catalog, epsilon=1000, min_samples=30)
    with state.loading_message("Calculating clusters..."):
        labels = await clustering.get_labels()

    set_labels = set(labels)
    n_clusters = len(set_labels) - (1 if -1 in set_labels else 0)
    ui.label(f"Identified {n_clusters} clusters using DBSCAN").classes(
        "text-body1 font-medium mb-4"
    )

    cluster_colors = labels_to_colors(labels)

    with ui.row().classes("w-full flex-1 items-stretch"):
        with ui.card().classes("col-12"):
            overview = OverviewMap(catalog)
            overview.header()
            await overview.view(marker_colors=cluster_colors)
        with ui.card().classes("col-12"):
            rate = MagnitudeRate(catalog)
            rate.header()
            await rate.view(
                show_semblance=not catalog.has_magnitudes(),
                show_density=False,
                marker_colors=cluster_colors,
            )
