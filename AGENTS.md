# AGENTS.md — qseek

## Project Overview

**qseek** is an automatic, data-driven earthquake detection and localization tool for large seismic datasets. It combines neural network phase detection (SeisBench), stacking-and-migration, adaptive octree localization, magnitude calculation, and station corrections.

- **License**: GPLv3
- **Python**: >= 3.12
- **Author**: Marius Paul Isken (<marius.isken@gfz.de>)
- **Docs**: <https://pyrocko.github.io/qseek/>
- **Repo**: <https://github.com/pyrocko/qseek>

---

## Package Manager & Build

**Always use `uv`** — never `pip` directly.

```bash
uv pip install -e .          # Editable install (compiles C extensions)
uv sync --dev                # Sync all dev dependencies
uv run <command>             # Run command in the project venv
```

The project uses `setuptools` with `setuptools_scm` (versions derived from git tags), `uv` as the package manager, and a `uv.lock` lockfile. C extensions in `src/qseek/ext/` are compiled at install time via `setup.py`.

---

## Testing

```bash
uv run --no-sync pytest -m "not plot"         # Standard test run (skip plot-heavy tests)
uv run --no-sync pytest test/test_search.py   # Single file
uv run --no-sync pytest -x                    # Stop on first failure
```

- Use `--no-cov` for faster tests
- Pass `--no-sync` when plugins are installed into the venv as editable packages: a plain `uv run` syncs the environment exactly and removes them
- Tests live in `test/` with `conftest.py` providing shared fixtures
- Coverage reports written to `htmlcov/`
- Async tests use `pytest-asyncio`; default loop scope is per-function
- The `qseek.testing` module (`src/qseek/testing.py`) is a pytest plugin with shared fixtures
- Benchmark tests are in `test/test_benchmark.py`

---

## Linting & Formatting

**Ruff** is the sole linter and formatter.

```bash
uvx ruff@0.15.6 check <files> --fix   # Lint and auto-fix
uvx ruff@0.15.6 format <files>        # Format
uvx ty@0.0.85 check --python .venv <files>   # Type check
```

The venv has no ruff. Use the versions in `uv.lock` and `.pre-commit-config.yaml`: a bare `uvx ruff` is the latest release, which applies other rules. Pass explicit files rather than whole directories, so the fixes don't reach untracked files. If the prek `ty` hook can't build its environment (e.g. no torch wheel for your Python), run ty as above and commit with `SKIP=ty`.

Rules enforced: `W, N, DTZ, FA, G, RET, SIM, B, C4, I, RUF, T20, D, NPY201`
Docstring convention: **Google**
Line length: Ruff default (88 chars)

---

## Pre-commit Hooks

```bash
prek install    # Set up pre-commit hooks (required for development)
```

Hooks configured in `.pre-commit-config.yaml`:

- `ruff` linter and formatter
- `ty` type checker
- Trailing whitespace / EOF fixers
- Large-file detection
- `clang-format` for C extension sources (`src/qseek/ext/*.c`)

---

## Code Conventions

### Imports

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from heavy_module import HeavyType
```

Always add `from __future__ import annotations` at the top of every module.

### Models & Configuration

- All configuration models use **Pydantic v2** (`BaseModel`)
- Use `@field_validator`, `@model_validator`, `@computed_field` as appropriate
- Lightweight data containers use `@dataclass(slots=True)`
- Abstract interfaces (e.g., `ImageFunction`, `RayTracer`, `EventMagnitude`) are defined in subpackage `__init__.py` or base files

### Async

- Prefer `async`/`await` for all I/O (waveform loading, file I/O, HTTP)
- Use `asyncio.Semaphore` to limit concurrency (e.g., `asyncio.Semaphore(16)` for Squirrel)
- Use `asyncio.Lock` for shared-state updates
- The CLI runs plain `asyncio.run()`; there is no `nest_asyncio` (removed in #107), so `asyncio.TaskGroup`, `asyncio.timeout()` and `asyncio.current_task()` work. Never call `asyncio.run()` inside the running loop; run synchronous code that does in a thread.
- SeisBench is called with `await model.annotate_async(...)` on the event loop, not `annotate()` (which calls `asyncio.run()`).
- **Pipeline stages** (waveform loading, pre-processing, image functions) are async generators that `qseek.utils.prefetch(iterator, queue, name=...)` runs ahead of the consumer in a task with a bounded queue. Pass the queue to the stage's `QueueStats` (`qseek.stats`) for the live view. Never hand-roll a queue with a `None` sentinel: a failing producer then hangs the consumer (fixed in #106).
- A stage closes its source with `async with contextlib.aclosing(source)`, and consumers close the pipeline the same way, so a failure or cancellation cancels every stage before it.
- **Background work** of the search goes through `qseek.utils.BackgroundTasks`, an `asyncio.TaskGroup` subclass: failed tasks are logged and don't cancel the others, the block waits for tasks created while it exits, and a failing block raises its own exception, not an `ExceptionGroup`. A TaskGroup can be entered once, so create a new one per run.
- The tests of these mechanisms are in `test/test_pipeline.py` and `test/test_event_loop.py`. Assert that no tasks are left pending, and guard waits with `asyncio.timeout()` so a regression fails instead of hanging.
- On macOS, qseek's C extensions and torch each bring their own OpenMP runtime, and the second one to start aborts the process. Tests that import torch after qseek's extensions ran must use a fresh interpreter (see `test_deep_denoiser`).

### Logging

```python
import logging

logger = logging.getLogger(__name__)
```

### Type Hints

- Type hints are required throughout
- `py.typed` marker is present (PEP 561 compliant)
- Use `Literal` for constrained string types (e.g., `Literal["P", "S"]`)
- Custom type aliases live in `utils.py` (e.g., `NSL`, `PhaseDescription`, `MeasurementUnit`)

---

## Project Structure

```
src/qseek/
├── apps/               # CLI entry point (qseek command)
├── models/             # Pydantic models (detection, catalog, station, location)
├── search.py           # Core search engine
├── octree.py           # Adaptive octree spatial partitioning
├── tracers/            # Ray tracers (cake, fast_marching, constant_velocity)
├── images/             # Image functions and pickers (SeisBench, STA/LTA)
├── features/           # Feature extraction (ground motion, etc.)
├── magnitudes/         # Magnitude calculations
├── waveforms/          # Waveform providers (SDS, Squirrel, Seedlink)
├── corrections/        # Station corrections (SimpleCorrections)
├── exporters/          # Export formats (velest, simple)
├── pre_processing/     # Frequency filters, downsampling
├── synthetics/         # Synthetic data generation
├── ext/                # C extensions (array_tools, delay_sum, stack)
├── utils.py            # Common utilities and type aliases
├── server.py           # aiohttp web server
├── reduce.py           # Delay-sum reduction
├── testing.py          # Pytest plugin with shared fixtures
├── plugin_loader.py    # Loads module plugins (entry point group `qseek.modules`)
├── plugins/            # Search callbacks (Telegram alerts, callback scripts)
└── cache_lru.py        # LRU cache management
src/qseek_inversion/    # Velocity model inversion from a qseek run (`qseek-inversion`)
test/
docs/
.github/workflows/      # CI: tests.yaml, release.yaml, docs.yaml
```

`src/qseek_inversion/` has its own [AGENTS.md](src/qseek_inversion/AGENTS.md). Read it before changing the inversion, `qseek.synthetics`, or the APIs it uses (`Search.from_config`, `OctreeSearch`, `FastMarchingTracer.set_layered_model`).

---

## CLI Entry Points

```bash
qseek config              # Print default JSON config
qseek search <config.json>    # Run earthquake detection
qseek continue <rundir>   # Resume a run
qseek -q search <config.json>  # Quiet: console errors only, no live view
qseek snuffler <rundir>   # Open detections in Pyrocko Snuffler
```

`--quiet`/`-q` (global, before the subcommand) limits the console to errors and skips the `rich` live statistics view. The `qseek.log` in the rundir still receives INFO messages. Use it for non-interactive and agent runs to save tokens; add `-v` for warnings, `-vv` for INFO.

---

## Module plugins

Separate packages provide modules by subclassing qseek's module base classes (`RayTracer`, `TravelTimeCorrections`, `ImageFunction`, `WaveformProvider`, ...) and registering an entry point in the `qseek.modules` group. `qseek/__init__.py` imports them via `qseek.plugin_loader.load_plugins()` before the configuration unions (`*Type`, built from `get_subclasses()` at import time) are created. `qseek modules` marks plugin modules with the plugin name.

Plugins build on the base classes and on `qseek.models`, `qseek.cache_lru`, `qseek.stats` and `qseek.utils`. When changing these APIs, run the tests of the plugins you have installed.

---

## CI/CD

- **Tests** (`tests.yaml`): Runs on PRs and main-branch pushes; matrix of Python 3.12/3.13/3.14 on Ubuntu and macOS
- **Docs** (`docs.yaml`): Builds the docs with Zensical and publishes them to the `gh-pages` branch on tag pushes
- **Release** (`release.yaml`): Builds and tests wheels with cibuildwheel (Linux x86_64/aarch64, macOS arm64/x86_64, CPython 3.12–3.14) on main, `v*` tags and PRs changing the build; publishes to PyPI on `v*` tags after the full test matrix passed

---

## Documentation

```bash
uv sync --all-extras
uv run zensical serve   # Preview locally
uv run zensical build   # Build static site into site/
```

Zensical caches rendered pages in `.cache/`, keyed on the Markdown source. After changing Python code that the docs render (docstrings, field descriptions, markdown-exec output), run `rm -rf .cache` before building; `--clean` does not clear it.

Docs are in `docs/`, built with [Zensical](https://zensical.org) (the successor of Material for MkDocs) from `mkdocs.yml`; theme overrides live in `overrides/`. They use `mkdocstrings` with `griffe-pydantic` for auto-generated API docs from Google-style docstrings.

**Configuration pages** render the Pydantic models natively with mkdocstrings and griffe-pydantic, e.g. `::: qseek.tracers.cake.CakeTracer` with `options: heading_level: 3`, followed by a collapsible JSON example from `qseek.utils.json_example()` in a markdown-exec block. Union types of modules (e.g. `RayTracerType`) link to their page through Markdown anchors before the page title, e.g. `[](){ #qseek.tracers.tracers.RayTracers }`. Use native Zensical/mkdocstrings features; no custom templates or griffe extensions.

**Style guide:** Read [DOCS_STYLE.md](DOCS_STYLE.md) before writing or editing any page in `docs/` or the docstring or field `description` of a configuration model (rendered into the configuration pages). It defines the audience (seismologists), the voice and tone, American spelling, second person, sentence-case headings, no emoji or marketing words, and the terminology table. It lives outside `docs/` because Zensical does not support `exclude_docs`.
