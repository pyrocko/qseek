---
icon: lucide/file-output
---

# Exporters

`qseek export` writes the detections of a run into the input formats of other programs. Use it to relocate the detections with HypoDD or to invert for a 1D velocity model with VELEST.

| Exporter | Output |
| --- | --- |
| `hypodd` | A HypoDD project folder for double-difference relocation, ready to run |
| `velest` | A VELEST project folder for the inversion of a 1D velocity model and station corrections |
| `simple` | The travel times of the picks as CSV files, one per detection |

## Export a run

Start the export in the directory of the search configuration, so that relative paths in the configuration of the run, e.g. of the velocity model, are found:

```sh title="Export a run"
qseek export list                                 # list the exporters
qseek export hypodd my-search/ my-search-hypodd/  # export the run my-search/
```

The export directory must not exist. `--force` replaces an existing directory, but only after the export succeeded. `--config` reads the settings of the exporter from a JSON file:

```sh title="Export with your settings"
qseek export hypodd my-search/ my-search-hypodd/ --config hypodd.json
```

Each exporter writes its settings to `export_info.json` in the export directory, so you can reproduce the export.

!!! abstract "Citations"
    Waldhauser, F., and W. L. Ellsworth (2000). A double-difference earthquake location algorithm: Method and application to the northern Hayward fault, California. *Bulletin of the Seismological Society of America*, 90(6), 1353–1368. [doi:10.1785/0120000006](https://doi.org/10.1785/0120000006)

    Kissling, E., W. L. Ellsworth, D. Eberhart-Phillips, and U. Kradolfer (1994). Initial reference models in local earthquake tomography. *Journal of Geophysical Research*, 99(B10), 19635–19646. [doi:10.1029/93JB03138](https://doi.org/10.1029/93JB03138)

## HypoDD

[HypoDD](https://www.ldeo.columbia.edu/~felixw/hypoDD.html) relocates earthquakes with the double-difference method (Waldhauser and Ellsworth, 2000). It inverts the travel time differences of event pairs at common stations, so errors of the velocity model largely cancel for close events and the relative locations of a cluster sharpen.

`qseek export hypodd` writes a complete project folder for HypoDD 2.1: the picks, the stations, the velocity model and the control files of ph2dt and hypoDD. Run it with the HypoDD binaries:

```sh title="Export and relocate"
qseek export hypodd my-search/ my-search-hypodd/
cd my-search-hypodd/
HYPODD_BIN=~/src/HypoDD/bin ./run.sh
```

`run.sh` runs ph2dt and hypoDD, then converts the relocations with `hypodd_results.py`:

| File | Content |
| --- | --- |
| `hypodd_relocations.csv` | The relocated events with ISO 8601 origin times, the HypoDD statistics, the Qseek detection, the shift from the Qseek location and a `WKT_geom` column for QGIS |
| `hypodd_relocations.yaml` | The relocated events as Pyrocko events, if Pyrocko is installed |
| `hypoDD.reloc`, `hypoDD.log` | The relocations and the log of hypoDD |

What the export does:

- **Selection:** picks with a confidence of at least 0.3 and a residual of at most 1 s to the modeled arrival, detections with at least 6 of these picks. The confidence is the pick weight in HypoDD. `max_rms` and `min_distance_border` select detections by their residual RMS and their distance to the border of the search volume.
- **Velocity model:** the 1D model of the ray tracer of the P phase, from Pyrocko Cake or fast marching. Gradient layers are split into constant velocity layers of at most 500 m, each with the harmonic mean velocity of its depth range. A constant velocity becomes HypoDD's straight-ray model.
- **Data:** catalog differential times, which ph2dt forms from the picks. With `cross_correlation`, the export also correlates the waveforms of close events for differential times in `dt.cc`. Station corrections of the run are not applied, the double-difference method does not need them.

On the Campi Flegrei example of the [quick start](../getting-started/quick-start.md), 20 May 2024, Qseek exports 424 detections with 5893 picks and hypoDD relocates 340 of them in about 2 s.

The guide [relocate with HypoDD](../guides/relocate-hypodd.md) explains how to check and tune the relocation, how the export handles depths, elevations and air-quakes, every column of the CSV file and all settings.

## VELEST

[VELEST](https://github.com/Dal-mzhang/REAL) inverts the travel times of local earthquakes jointly for their hypocenters, a 1D velocity model and station corrections: the minimum 1D model (Kissling et al., 1994). It also locates single events in a given model.

`qseek export velest` writes a VELEST project folder for the joint inversion. The export asks for the selection of the detections and picks:

| Prompt | Default | Selects |
| --- | --- | --- |
| Minimum event semblance | 0.2 | Detections by their semblance |
| Minimum number of receivers (P phase) | 10 | Detections with at least this many P picks |
| Minimum distance to border (meters) | 500 | Detections at least this far from the border of the search volume |
| Minimum pick probability for P and S phase | 0.3 | Picks with a higher confidence |
| Maximum travel time delay | 2.5 | Picks at most this many seconds after the modeled arrival |

Qseek writes these files:

| File | Content |
| --- | --- |
| `velest.cmn` | VELEST control file |
| `phase_velest.pha` | Detections and their picks with travel times |
| `stations_velest.sta` | Stations with their elevation |
| `model.mod` | Initial P and S velocity model |
| `export_info.json` | Settings of the export |
| `README.md` | How to install and run VELEST |

What the export does:

- **Pick weights:** VELEST weights picks by classes. The pick confidence becomes class 0 from 0.8, class 1 from 0.6, class 2 from 0.4 and class 3 below 0.4.
- **Velocity model:** the layers of the [Pyrocko Cake](../configuration/ray-tracers.md#pyrocko-cake) model of the first ray tracer, with their P and S velocities. The export needs a Cake ray tracer as the first ray tracer of the search.
- **Control file:** joint inversion (`isingle=0`) with 9 iterations (`ittmax`), the velocity model updated every third iteration (`invertratio=3`) and a Vp/Vs ratio of 1.65. The reference point is the center of the search volume; VELEST counts longitudes positive to the west, so its sign is flipped. Station elevations and station corrections are off.

Run VELEST in the project folder. Build it from the `src/VELEST` folder of the [REAL repository](https://github.com/Dal-mzhang/REAL) and copy the region files from its VELEST demo:

```sh title="Run VELEST"
cd my-search-velest/
cp ~/src/REAL/demo_real/VELEST/region* .
velest
```

VELEST writes its report to `main.out`, the final hypocenters to `final.cnv` and the station corrections to `stacor.dat`. To only locate the events in the initial model, set `isingle` to 1 in `velest.cmn`. Check the parameters of `velest.cmn` against the VELEST documentation before you invert; the control file has many more options than the export sets.
