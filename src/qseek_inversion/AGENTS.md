# AGENTS.md — qseek_inversion

Data-driven inversion of a 1D layered velocity model from a qseek run. Read `AGENTS.md` at the repository root first; its conventions (uv, ruff, ty, async, Pydantic, logging) apply here too.

## The method

1. Qseek searches a dataset and writes a run directory (`search.json`, `detections.json`).
2. The inversion imports that run, selects events and their waveforms, and computes their image functions once.
3. For each trial velocity model, a new `FastMarchingTracer` computes the travel times and `OctreeSearch` locates the selected events again.
4. The fitness of a model is the cumulative semblance of the located events: a model that focuses the image functions better gives a higher stack.

No phase picks or associations are needed. The search is the same as in `qseek search`, with `node_interpolation`, `attach_arrivals` and station corrections off.

The optimizer is a first, simple one: hill climbing. Each trial perturbs the P velocities of the current best model by up to `perturbation` (relative, Vp/Vs kept) and is accepted if it raises the cumulative semblance (`InversionLayered1D.invert()`). It is meant to be replaced by a probabilistic sampler.

## CLI

```bash
qseek-inversion config --import-rundir <qseek-rundir> [--selection EventWaveformsSelection] > inversion.json
qseek-inversion modules [<name>]           # list modules, or print the JSON config of one
qseek-inversion invert inversion.json --check   # load the run, select the events, exit
qseek-inversion invert inversion.json [--rundir DIR] [--force]
qseek-inversion summary <inversion-rundir>
```

The rundir defaults to the config path without its suffix. `--force` moves an existing rundir to `<rundir>.bak-<time>`; nothing is deleted.

For scripts and agents, put `--non-interactive` before the command: the console shows only errors and `key: value` lines (`rundir`, `log`, `results`, `best_model`, `start_semblance`, `best_semblance`, `status`, and on failure `error`). The full INFO log is in `<rundir>/inversion.log`. Exit codes as in `qseek`: 0 ok, 1 error, 2 config or usage error, 130 interrupted.

An inversion rundir holds `inversion.json` (the config), `inversion.log`, `iterations.jsonl` (one `InversionIteration` per trial, 0 is the start model), `best_model.json` (`LayeredModel`), `results.json` (`InversionSummary`) and the exports of the selection (`synthetic_catalog/`, or `selected_events.csv` and `concatenated_traces/`).

## Layout

```
src/qseek_inversion/
├── inversion.py          # InversionLayered1D: loads the run, evaluates velocity models
├── velocity_model.py     # LayeredModelInversion: copy and perturb a LayeredModel
├── waveforms/            # Event and waveform selection (`event_selection` field)
│   ├── base.py           # WaveformSelection: prepare(search, import_rundir, rundir), get_images()
│   ├── event_selection.py  # EventWaveformsSelection: the N best events of the run, concatenated
│   └── synthetics.py     # SyntheticEvents: Gaussian image functions of random events
└── app/qseek_inversion.py  # CLI `qseek-inversion`: config, modules, invert, summary
```

`qseek.synthetics.synthetics` (`SyntheticEvent`, `SyntheticEventCatalog`) belongs to qseek, not to this package. It models the arrivals of synthetic events with any ray tracer and renders them as Gaussian peaks.

## Conventions

- The selections are a discriminated union on the `waveforms` field (`WaveformSelectionType`, built from `WaveformSelection.__subclasses__()`). A new selection subclasses `WaveformSelection` and is imported in `waveforms/__init__.py`.
- `InversionLayered1D.prepare()` loads the imported run once with `Search.from_config()` and hands the search to the selection. Selections don't load the run themselves. Never write into the imported run directory; exports go to the `rundir` passed to `prepare()`.
- Never mutate the start model: `LayeredModelInversion.from_layered_model()` returns an independent copy. Perturbations take a `numpy.random.Generator` so runs are reproducible.
- A trial tracer copies `phases`, `interpolation_method` and `nthreads` of the run's `FastMarchingTracer`, so the phases of the image functions keep their tracer. Other tracers of the run stay unchanged.
- The window padding is the travel time span of the start model times `PADDING_FACTOR` plus the blinding of the image function. Trial models with lower velocities shift the arrivals further.
- Synthetic events use the octree reference's frame (`lat`, `lon`, `north_shift`, `east_shift`, `depth`, `elevation`), like `Node.as_location()`.

## Testing

```bash
uv run --no-sync pytest test/test_inversion.py --no-cov
```

`test_synthetic_closed_loop` closes the loop: synthetic events, modelled in a known velocity model, are located in the same model and must come back within the smallest octree node and 0.05 s; a slower model must give a lower cumulative semblance. `test_cli_invert` inverts from a start model 10% too slow and must move towards the true model. Run them after every change to the inversion, the synthetics, `OctreeSearch` or the fast marching tracer.

For a quick look, the same synthetic inversion with 40 trials recovered Vp 3.91, 5.04 and 5.99 km/s of the true 4, 5 and 6 km/s from a start at 3.6, 4.5 and 5.4 km/s in 5 s.

Keep the synthetic geometry well constrained. With stations only outside the source volume, depth trades off against origin time: a 0.9 km depth error left residuals of at most 0.05 s, within the 0.1 s Gaussian peak. Such an event fails a location test without any bug.

## Known limitations

- The fitness sums the semblance of detected events. It jumps when an event crosses the trigger threshold, and spurious or split detections add to it.
- The resolution of the fitness is limited by the width of the image function peaks and by the network geometry.
- Only the P velocity is perturbed; Vp/Vs and the layer depths are fixed.
- Hill climbing stops at the first local maximum and gives no uncertainty.
