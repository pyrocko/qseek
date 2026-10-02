---
icon: lucide/crosshair
---

# Relocate with HypoDD

[HypoDD](https://www.ldeo.columbia.edu/~felixw/hypoDD.html) relocates earthquakes with the double-difference method (Waldhauser and Ellsworth, 2000). It inverts the differences of the travel times of event pairs at common stations. Errors of the velocity model largely cancel for close events, so HypoDD sharpens the relative locations of a cluster without station corrections.

`qseek export hypodd` writes a HypoDD project folder from a run: the picks, the stations, the velocity model and the control files for ph2dt and hypoDD, ready to run.

On the [Campi Flegrei example](../getting-started/quick-start.md), 20 May 2024, Qseek exports 424 detections with 5894 picks. ph2dt links 347 of them and hypoDD relocates 323 in 2.5 s. On these 323 events, the median absolute double-difference residual of the catalog differential times falls from 72 ms at the Qseek locations to 52 ms at the HypoDD locations, and from 65 ms to 33 ms for P.

!!! abstract "Citation"
    Waldhauser, F., and W. L. Ellsworth (2000). A double-difference earthquake location algorithm: Method and application to the northern Hayward fault, California. *Bulletin of the Seismological Society of America*, 90(6), 1353–1368. [doi:10.1785/0120000006](https://doi.org/10.1785/0120000006)

## Export a run

Export the detections of a finished run. Start the export in the directory of the search configuration, so that the velocity model file is found:

```sh title="Export a run to HypoDD"
qseek export hypodd my-search/ my-search-hypodd/
```

Qseek writes these files:

| File | Content |
| --- | --- |
| `phase.dat` | Detections and their picks, the input of ph2dt |
| `station.dat` | Stations with their elevation in meters |
| `ph2dt.inp` | Control file of ph2dt |
| `hypoDD.inp` | Control file of hypoDD, catalog differential times only |
| `event_ids.csv` | HypoDD event ID, Qseek detection UID and origin time |
| `stations.csv` | HypoDD station label and station code (NSL) |
| `velocity_model.csv` | The layered velocity model in `hypoDD.inp` |
| `export_info.json` | Settings of the export |
| `run.sh` | Runs ph2dt and hypoDD |

The export selects the detections and picks:

- A pick needs a confidence of at least [`min_pick_confidence`][qseek.exporters.hypodd.HypoDD.min_pick_confidence] (default 0.3), and its residual to the modeled arrival must not exceed [`max_residual`][qseek.exporters.hypodd.HypoDD.max_residual] (default 1 s). The confidence is the pick weight in `phase.dat`.
- A detection needs at least [`min_picks`][qseek.exporters.hypodd.HypoDD.min_picks] of these picks (default 6).
- The travel times are the observed picks minus the origin time. Station corrections of the run are not applied; the double-difference method does not need them.

HypoDD needs unique integer event IDs and station labels of up to 7 characters. Qseek numbers the detections in time order and uses the station code as the label, or network and station code if the station code is not unique. Map the IDs in `hypoDD.reloc` back to the detections with `event_ids.csv`.

## Run HypoDD

Build ph2dt and hypoDD from the [HypoDD distribution](https://www.ldeo.columbia.edu/~felixw/hypoDD.html) (version 2.1), then run both in the project folder:

```sh title="Run ph2dt and hypoDD"
cd my-search-hypodd/
HYPODD_BIN=~/src/HypoDD/bin ./run.sh
```

`HYPODD_BIN` is the directory of the binaries; leave it out if they are in your `PATH`. ph2dt writes the catalog differential times `dt.ct` and the initial locations `event.sel`, hypoDD the relocations `hypoDD.reloc` and its log `hypoDD.log`.

Check the log before you use the relocations:

- **Linked events:** ph2dt lists the events it selected and the weakly linked events in `ph2dt.log`. Events without enough links to their neighbors are not relocated.
- **Condition number:** the LSQR solver should reach a condition number (`CND` in the iteration table) of about 40 to 80. Raise `DAMP` in `hypoDD.inp` if it is higher, lower it if it is lower. On Campi Flegrei, the default damping of 80 gives a CND of 41 to 49.
- **Shifts:** the mean shifts `DX`, `DY`, `DZ` should fall to the noise level of the data within the last iterations. The centroid shift `OS` should stay below the location uncertainty of the detections; HypoDD does not constrain the absolute position of a cluster well.

!!! warning
    The errors of the LSQR solver in `hypoDD.reloc` are not meaningful. Use the SVD solver on small clusters, below 200 events, or a bootstrap for error estimates.

## Velocity model

The export takes the 1D velocity model of the ray tracer of the P phase: the [Pyrocko Cake](../configuration/ray-tracers.md#pyrocko-cake) or [fast marching](../configuration/ray-tracers.md#fast-marching) model, written as layers with their P velocity and Vp/Vs ratio (`IMOD=1`). A [constant velocity](../configuration/ray-tracers.md#constant-velocity) becomes HypoDD's straight-ray model (`IMOD=5`). 3D models are not exported.

HypoDD needs layers of constant velocity. Gradient layers are split into layers of at most [`max_layer_thickness`][qseek.exporters.hypodd.HypoDD.max_layer_thickness] (default 500 m) down to the bottom of the search volume. Each layer gets the harmonic mean velocity of its depth range, which keeps the vertical travel time. HypoDD allows 30 layers; raise `max_layer_thickness` if the model needs more.

Depths in HypoDD are in kilometers below sea level, like the depths of Qseek. HypoDD places the top of the model at the elevation of each station, so the velocity of the first layer applies from the station down to the second layer. The top of the first layer is written as 1 km above sea level: hypoDD 2.1 fails with NaN for an event at the top of the first layer. Detections above sea level are set to 0 km depth.

## Settings

Change the selection and the parameters of ph2dt and hypoDD with a JSON file:

```sh title="Export with your settings"
qseek export hypodd my-search/ my-search-hypodd/ --config hypodd.json
```

```python exec='on'
from qseek.utils import json_example
from qseek.exporters.hypodd import HypoDD

print(json_example(HypoDD()))
```

Distances are in meters, as everywhere in Qseek; the export converts them to the kilometers of HypoDD. The ph2dt defaults follow the HypoDD user guide for a dense local network: event pairs up to 5 km apart and at least 8 differential times per pair. The three default iteration sets weight P twice as strongly as S, then remove outliers beyond 6 and 4 standard deviations and limit the pair separation to 4 and 2 km.

The control files are plain text with comments. You can also edit `ph2dt.inp` and `hypoDD.inp` in the project folder and run HypoDD again.

<div class="qs-config" markdown>

::: qseek.exporters.hypodd.HypoDD
    options:
      heading_level: 3

::: qseek.exporters.hypodd.Ph2DTSettings
    options:
      heading_level: 3

::: qseek.exporters.hypodd.HypoDDSettings
    options:
      heading_level: 3

::: qseek.exporters.hypodd.IterationSet
    options:
      heading_level: 3

</div>
