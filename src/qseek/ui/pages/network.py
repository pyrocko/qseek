from __future__ import annotations

import numpy as np
from nicegui import ui
from pyrocko import orthodrome as od

from qseek.ui.base import Page
from qseek.ui.components import map as network_map_components
from qseek.ui.components import statistics as network_statistics_components
from qseek.ui.state import get_tab_state
from qseek.ui.utils import stat_card


class NetworkPage(Page):
    async def render(self) -> None:
        state = get_tab_state()
        catalog = await state.get_filtered_catalog()

        rows = []
        for ev in catalog.events:
            for receiver in ev.event.receivers:
                if not receiver.phase_arrivals:
                    continue
                start_time, _ = receiver.get_arrivals_time_window()
                rows.append(
                    {
                        "station": receiver.nsl.pretty_str(strip=True),
                        "lat": float(receiver.effective_lat),
                        "lon": float(receiver.effective_lon),
                        "start": start_time,
                    }
                )

        unique_stations = {row["station"]: row for row in rows}
        n_stations = len(unique_stations)

        max_active_stations = 0
        min_active_stations = 0
        if rows:
            times = np.asarray([row["start"].timestamp() for row in rows], dtype=float)
            one_week = 24.0 * 3600.0 * 7.0
            if times.max() <= times.min():
                time_edges = np.array(
                    [times.min() - one_week / 2.0, times.max() + one_week / 2.0],
                    dtype=float,
                )
            else:
                time_edges = np.arange(times.min(), times.max() + one_week, one_week)
                if len(time_edges) < 2:
                    time_edges = np.array([times.min(), times.min() + one_week])

            station_counts = np.zeros(len(time_edges) - 1, dtype=np.int32)
            for station in unique_stations:
                station_times = np.asarray(
                    [
                        row["start"].timestamp()
                        for row in rows
                        if row["station"] == station
                    ],
                    dtype=float,
                )
                counts, _ = np.histogram(station_times, time_edges)
                station_counts += (counts > 0).astype(np.int32)

            if station_counts.size > 0:
                max_active_stations = int(np.max(station_counts))
                min_active_stations = int(np.min(station_counts))

        median_interstation_km = 0.0
        extent_km = 0.0
        station_list = list(unique_stations.values())
        if len(station_list) >= 2:
            distances_km = []
            for i in range(len(station_list)):
                for j in range(i + 1, len(station_list)):
                    d_m = od.distance_accurate50m_numpy(
                        station_list[i]["lat"],
                        station_list[i]["lon"],
                        station_list[j]["lat"],
                        station_list[j]["lon"],
                    )[0]
                    distances_km.append(float(d_m) / 1000.0)

            if distances_km:
                distances_km_arr = np.asarray(distances_km, dtype=float)
                median_interstation_km = float(np.median(distances_km_arr))
                min_interstation_km = float(np.min(distances_km_arr))
                max_interstation_km = float(np.max(distances_km_arr))
                extent_km = float(np.max(distances_km_arr))

        with ui.row().classes("w-full items-center gap-2 mb-1"):
            ui.label("Network").classes("text-h1")

        with ui.row().classes("items-center gap-4 w-full"):
            stat_card(
                "Stations",
                str(n_stations),
                icon="sensors",
                subtitle="unique active stations",
                tooltip="Number of unique stations with phase arrivals in filtered events.",
            )
            stat_card(
                "Max Active",
                str(max_active_stations),
                icon="trending_up",
                subtitle="stations",
                tooltip="Maximum number of active stations in any 1-day time bin.",
            )
            stat_card(
                "Min Active",
                str(min_active_stations),
                icon="trending_down",
                subtitle="stations",
                tooltip="Minimum number of active stations in any 1-day time bin.",
            )
            stat_card(
                "Median Interstation",
                f"{median_interstation_km:.0f} km",
                icon="straighten",
                subtitle=f"Min {min_interstation_km:.0f} km / Max {max_interstation_km:.0f} km",
                tooltip="Median pairwise distance between active stations.",
            )
            stat_card(
                "Network Extent",
                f"{extent_km:.0f} km",
                icon="open_in_full",
                subtitle="maximum pairwise distance",
                tooltip="Largest pairwise distance between active stations.",
            )

        with ui.row().classes("items-start gap-4 w-full"):
            await network_map_components.NetworkMap().render()
            await network_statistics_components.StationActivityOverview().render()
            await network_statistics_components.NStations_over_time().render()
