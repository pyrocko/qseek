---
icon: lucide/history
---

# Manage runs

A run is a search and its [run directory](../results/run-directory.md). This guide shows how to restart, continue and re-evaluate runs.

## Start again

`qseek search` does not overwrite an existing run directory. To start the search again with the same name:

```sh title="Start a search again"
qseek search my-search.json --force      # keep the old run as a backup
qseek search my-search.json --force --no-backup  # overwrite the old run
```

With `--force`, Qseek renames the old run directory with a `.bak-<time>` suffix.

## Continue a search

If a search stops, e.g. on a machine restart, continue it where it stopped:

```sh title="Continue a search"
qseek continue my-search/
```

Qseek reads the configuration of the run from its `search.json` and the progress from `progress.json`. Detections after the last processed window are removed and searched again.

## Recalculate magnitudes and features

To change the magnitudes or features of a finished run, edit the `magnitudes` and `features` in the run's `search.json` and recalculate them for all detections:

```sh title="Recalculate magnitudes and features"
qseek feature-extraction my-search/ --recalculate
```

`--nparallel` sets how many detections are processed in parallel.

## Search again with station corrections

After a first run, extract [station corrections](../configuration/station-corrections.md#extract-corrections-from-a-previous-run) from its travel time residuals and run a second search with them. The corrections refine the locations and can reveal more events.

## Export the detections

Export the detections of a run to other formats, e.g. a HypoDD project for double-difference relocation:

```sh title="Export detections"
qseek export list
qseek export hypodd my-search/ hypodd-project/
```

| Format | Description |
| --- | --- |
| `hypodd` | A HypoDD project folder for double-difference relocation, see [relocate with HypoDD](relocate-hypodd.md). |
| `simple` | Travel times of the picks in CSV format. |
| `velest` | A VELEST project folder for 1D velocity model inversion. |

`--force` overwrites an existing export directory. `--config` reads the settings of the export module from a JSON file.
