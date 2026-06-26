from pathlib import Path
from tempfile import NamedTemporaryFile

from nicegui import background_tasks, binding, ui

from qseek.ui.analysis.cluster import ClusterDBScan
from qseek.ui.components.cluster import ClusterAnalysis, MagnitudeRateCluster
from qseek.ui.components.map import OverviewMap
from qseek.ui.state import get_tab_state
from qseek.ui.utils import StatCard, card_header


@binding.bindable_dataclass
class DBScanParameters:
    epsilon: float = 1000.0  # in meters
    min_samples: int = 30


@binding.bindable_dataclass
class ClusterStats:
    n_clusters: int = 0
    n_noise: int = 0
    n_clustered: int = 0


def config_dialog(params: DBScanParameters) -> ui.dialog:
    with (
        ui.dialog() as dialog,
        ui.card().classes("w-[480px] gap-0 !p-0 overflow-hidden rounded-xl shadow-2xl"),
    ):
        # ── Header ──────────────────────────────────────────────────────
        with ui.row().classes("items-center w-full px-5 pt-5 pb-4 gap-3"):
            ui.icon("settings", size="lg").classes("text-primary")
            with ui.column().classes("gap-0 flex-1"):
                ui.label("DBSCAN Parameters").classes(
                    "text-base font-bold leading-snug"
                )
                ui.label("Tune spatial clustering neighbourhood and density").classes(
                    "text-xs text-grey-6 leading-tight"
                )
            ui.button(icon="close", on_click=dialog.close).props(
                "flat round dense size=sm color=grey-7"
            )

        ui.separator().classes("opacity-20")

        # ── Controls ────────────────────────────────────────────────────
        with ui.column().classes("w-full px-6 py-5 gap-7"):
            # ── Epsilon ──
            with ui.column().classes("gap-2 w-full"):
                with ui.row().classes("items-center gap-1.5"):
                    ui.icon("square_foot", size="xs").classes("text-grey-5")
                    ui.label("Epsilon (m)").classes(
                        "text-sm font-semibold text-grey-8 tracking-wide"
                    )
                ui.slider(min=100, max=5000, step=100).classes("w-full").props(
                    "color=primary label"
                ).bind_value(params, "epsilon")
                ui.label().classes(
                    "text-xs text-grey-6 font-mono text-right"
                ).bind_text_from(
                    params,
                    "epsilon",
                    backward=lambda v: f"{v:,.0f} m neighbourhood radius",
                )

            # ── Min Samples ──
            with ui.column().classes("gap-2 w-full"):
                with ui.row().classes("items-center gap-1.5"):
                    ui.icon("people", size="xs").classes("text-grey-5")
                    ui.label("Min Samples").classes(
                        "text-sm font-semibold text-grey-8 tracking-wide"
                    )
                ui.slider(min=1, max=100, step=1).classes("w-full").props(
                    "color=secondary label"
                ).bind_value(params, "min_samples")
                ui.label().classes(
                    "text-xs text-grey-6 font-mono text-right"
                ).bind_text_from(
                    params,
                    "min_samples",
                    backward=lambda v: f"{v} min. events in neighbourhood",
                )

        ui.separator().classes("opacity-20")

        # ── Footer ──────────────────────────────────────────────────────
        with ui.row().classes("w-full px-6 py-4 justify-end gap-2"):
            ui.button("Cancel", on_click=lambda: dialog.submit(False)).props(
                "flat color=grey-7"
            ).classes("text-sm")
            ui.button("Apply", on_click=lambda: dialog.submit(True)).props(
                "unelevated color=primary"
            ).classes("text-sm")

    return dialog


async def clusters_page() -> None:
    state = get_tab_state()
    catalog = await state.get_catalog()

    params = DBScanParameters()
    stats = ClusterStats()
    dbscan = ClusterDBScan(catalog)

    dialog_dbscan = config_dialog(params)

    async def show_dialog() -> None:
        epsilon_before = params.epsilon
        min_samples_before = params.min_samples
        result = await dialog_dbscan
        if not result:
            params.epsilon = epsilon_before
            params.min_samples = min_samples_before
        else:
            await update_clusters()

    async def download_catalog() -> None:
        if dbscan.labels is None or dbscan.colors is None:
            ui.notify(
                "Please run DBSCAN clustering before downloading.",
                color="warning",
            )
            return
        ui.notify("Preparing download...", color="primary")
        cluster_info = [
            {
                "cluster_label": int(label),
                "cluster_color": color,
            }
            for label, color in zip(dbscan.labels, dbscan.colors, strict=True)
        ]
        with NamedTemporaryFile("w") as f:
            fp = Path(f.name)
            full_catalog = catalog.full_catalog
            try:
                btn_download.disable()
                await full_catalog.export_csv(fp, additional_data=cluster_info)
                ui.download(
                    fp.read_bytes(),
                    filename="clustered-catalog.csv",
                    media_type="text/csv",
                )
            finally:
                btn_download.enable()

    with ui.row().classes("w-full items-stretch"), ui.button_group().classes("ml-auto"):
        ui.button(
            "DBScan",
            icon="settings",
            on_click=show_dialog,
        ).props("push color=gray-700").classes("ml-auto")
        btn_download = ui.button(icon="download", on_click=download_catalog).props(
            "outline color=gray-400"
        )

    with ui.row().classes("w-full items-stretch"):
        card_clusters = StatCard(
            "Clusters",
            icon="spoke",
            tooltip="Number of clusters identified by DBSCAN.",
        )
        card_clusters.bind_value(stats, "n_clusters")
        card_clusters.bind_subtitle(
            stats,
            "n_clustered",
            backward=lambda v: f"{v} events clustered ({v / catalog.n_events * 100:.1f}%)",
        )
        card_noise = StatCard(
            "Noise Events",
            icon="close",
            tooltip="Number of events classified as noise"
            "(not belonging to any cluster).",
        )
        card_noise.bind_value(stats, "n_noise")
        card_noise.bind_subtitle(
            stats,
            "n_noise",
            backward=lambda v: f"Not in any cluster ({v / catalog.n_events * 100:.1f}%)",
        )
        StatCard(
            "Epsilon",
            icon="square_foot",
            subtitle="Neighborhood radius",
            tooltip="Maximum distance between two samples for them to be considered as "
            "in the same neighborhood.",
        ).bind_value(params, "epsilon", backward=lambda v: f"{v} m")
        StatCard(
            "Min Samples",
            icon="people",
            subtitle="Min samples in neighborhood",
            tooltip="Minimum number of samples in a neighborhood for a point to be "
            "considered as a core point.",
        ).bind_value(params, "min_samples")

    with ui.row().classes("w-full flex-1 items-stretch"):
        map_ = OverviewMap(catalog.lats.mean(), catalog.lons.mean())
        map_.set_title("Event Clusters")
        map_.set_description(
            "DBSCAN clustering of detected events based on their spatial proximity."
        )

        with ui.card().classes("col-12"):
            card_header(MagnitudeRateCluster.title, MagnitudeRateCluster.description)
            mag_rate = MagnitudeRateCluster()

        if catalog.has_magnitudes():
            with ui.card().classes("col-12"):
                card_header(ClusterAnalysis.title, ClusterAnalysis.description)
                analysis = ClusterAnalysis()

    async def update_clusters() -> None:
        with state.loading_message("Calculating clusters..."):
            try:
                labels = await dbscan.cluster(
                    epsilon=params.epsilon,
                    min_samples=params.min_samples,
                )
            except Exception as e:
                ui.notify(f"Error calculating clusters: {e}", color="negative")
                return
            await map_.initialize()
            stats.n_clusters = dbscan.n_unique_labels()
            stats.n_noise = (labels == -1).sum()
            stats.n_clustered = catalog.n_events - stats.n_noise

            cluster_colors = dbscan.colors

            background_tasks.create(
                map_.add_event_markers(
                    catalog.events,
                    marker_colors=cluster_colors,
                    highlight_latest=False,
                )
            )

            await mag_rate.plot_events(
                catalog.events,
                labels,
                show_semblance=not catalog.has_magnitudes(),
            )
            if catalog.has_magnitudes():
                await analysis.plot_clusters(catalog.events, labels)

    background_tasks.create(update_clusters())
