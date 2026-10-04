---
icon: lucide/flask-conical
---

# Playground

The [Qseek playground](https://github.com/pyrocko/qseek-playground) holds worked examples on real seismic data. Each example downloads its waveforms, runs a search and measures the result. Use it to reproduce the [quick start](quick-start.md) with one command per step, to see what a configuration change does to a search, and to check changes to Qseek itself.

| Example | Data | Result |
| --- | --- | --- |
| Campi Flegrei | 1 day, 18 stations of the INGV network, 20 May 2024 | 732 detections; all 45 events of the INGV catalog detected |

## Set up the playground

You need [uv](https://docs.astral.sh/uv/), [just](https://just.systems/), Python 3.12 or newer and about 1 GB of disk space per example. The playground runs Qseek from a source checkout next to it:

```sh title="Set up the playground"
git clone https://github.com/pyrocko/qseek.git
git clone https://github.com/pyrocko/qseek-playground.git
cd qseek-playground
just setup
```

`just setup` installs the checkout into `../qseek/.venv` and compiles its C extensions. Set `QSEEK_DIR` to use a checkout in another place.

## Run an example

```sh title="Run the Campi Flegrei example"
just download campi-flegrei      # waveforms and station metadata, about 1 GB
just search campi-flegrei dev    # search into campi-flegrei/runs/dev/
just explore campi-flegrei dev   # open the run in the web UI
```

`just search` writes the run directory `campi-flegrei/runs/dev/` and ends with one line of metrics: the number of detections, how many events of the reference catalog Qseek found and how far its locations are from the catalog. Add `--verbose` to print all metrics.

## Change the configuration and compare

Override a field of the configuration for one run with `--set`, then compare the run with the baseline of the example:

```sh title="Try other PhaseNet weights"
just search campi-flegrei original --set image_function.pretrained=original
just compare campi-flegrei original
```

The comparison pairs the detections of both runs by origin time. It reports the detections lost and added, how far the paired detections moved, and how their picks, residuals and semblance changed. `just compare` prints only the rows that changed; add `--full` for all rows. `just dashboard` shows the same comparison with maps and histograms in the browser.

A run of the same Qseek version on the same machine reproduces the baseline exactly. The playground runs `qseek --non-interactive search`, which prints only errors and a few status lines; the full log stays in `runs/<run>/qseek.log`.

To try several values of a parameter, run `just sweep campi-flegrei levels --vary octree.n_levels=3,4`. It runs one search per value and lists the runs in one table with their `--set` overrides.

## Relocate with HypoDD

`just hypodd` relocates the detections of a run with [HypoDD](../guides/relocate-hypodd.md) and compares the locations with those of Qseek. The [playground README](https://github.com/pyrocko/qseek-playground#relocate-with-hypodd) lists the steps.

## Next steps

- [Set up your own search](first-search.md) when you are ready to use your data.
- [Tune the detection](../guides/tune-detection.md) shows which settings to try with `--set`.
