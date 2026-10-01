# Qseek Docs Style Guide

How the Qseek documentation sounds and reads. Follow it for every page under `docs/` and for the docstrings and field `description`s of configuration models, which the configuration pages render. It lives outside `docs/` so that it is not published.

## Audience

The docs are written for **seismologists**: researchers and observatory staff who run Qseek on their own data sets. They know seismology: phases, travel times, velocity models, station networks, magnitudes. They use Python tools and the command line, but they are not software engineers.

- Use seismological terms without explaining them.
- Explain software and configuration details: file formats, JSON fields, CLI commands, run directories, performance settings.
- Readers come to a page with a task in mind: set up a search, tune a module, understand a result. Lead with what they need for that task.

## Voice

The voice stays the same on every page.

- **Precise.** Give the unit, the default and the valid range. Name the field, the command, the file. Prefer "within 2 s of the modeled arrival" to "close to the arrival".
- **Plain.** Short sentences, common words, one idea per sentence. Write it as you would explain it to a colleague at the next desk.
- **Direct.** Address the reader as "you" and use the imperative for steps. Qseek is the subject when the software does something. Use "we" only for the authors' design decisions and recommendations.
- **Credible.** Back claims with numbers, figures or references. Cite the method papers. Say where a module is limited or experimental.

## Tone by page type

The tone adapts to the page while the voice stays constant.

| Page type | Examples | Tone |
|---|---|---|
| Landing page | `index.md` | Confident and concrete: what Qseek does, on what data, with what result. Show, don't hype. |
| Showcase | `about/showcase.md` | Proud but factual: real data sets, real figures, real numbers, each with its source. |
| Tutorials | `getting-started/*.md` | Encouraging and step by step. Every step has a command or config snippet and says what you should see. |
| Guides | `guides/*.md` | Practical: one task per page, from the starting point to a working result, with the trade-offs on the way. |
| Concepts | `concepts/*.md` | Explanatory: how the method works, with figures and equations, so the reader can reason about their own data. |
| Configuration pages | `configuration/*.md` | Explanatory: what the module does, when to use it, which fields matter and how to tune them. Then the generated field reference. |
| Reference | `reference/**/*.md` | Terse and complete. The docstrings speak; no introduction beyond one sentence. |

## Showing results

The docs should make readers want to run Qseek on their own data. That excitement comes from results, not adjectives.

- **Lead with a result.** A detection map, a refined octree, a throughput number. Put the strongest figure near the top of the page.
- **Quantify.** "more than 30 000 earthquakes in the 2020 Reykjanes unrest" beats "many earthquakes". "Two days for 700 years of waveforms" beats "fast".
- **Only real results.** Every figure and number comes from a real run, a publication or the benchmark, and says which. Never invent or round up a result. Mark placeholders clearly until the real material exists.
- **Short time to success.** Show how few steps it takes from installation to the first detections.

## Language

- **American English**: localization, modeled, centered, visualize, color.
- **Active voice, present tense.** "Qseek refines the octree", not "The octree is refined" or "Qseek will refine".
- **No marketing words**: lightning-fast, highly-performant, powerful, seamless, cutting-edge, simply, just, easy. Show speed with the benchmark numbers instead.
- **No emoji** in headings, navigation or text.
- **Sentence case for headings**: "Station corrections", not "Station Corrections". Proper names keep their capitals: SeisBench, PhaseNet, Pyrocko Cake, NonLinLoc.
- **Units** with a space and SI symbols: `2 km`, `100 Hz`, `0.5 s`. Configuration fields use meters and seconds; say so when a field takes a unit.
- **Durations in the config** are ISO 8601 (`PT5M`). Give the readable form at first mention: "`PT5M` (5 minutes)".

## Terminology

Use one term for one thing.

| Use | Not | Notes |
|---|---|---|
| Qseek | QSeek, qseek | In prose. `qseek` only for the command and the Python package. |
| search | job | The process. `qseek search` starts it. A run is a search with its run directory. |
| run directory | rundir, project folder | `rundir` only as the CLI argument name. |
| detection | event | Qseek's output before review. "Event" for the earthquake itself. |
| localization | location | The process. "Location" for the resulting coordinates. |
| image function | annotation | `image_function` in the config. Its images are characteristic functions of the phase arrivals. |
| node | grid point, cell | A node of the octree. |
| station corrections | station delays | "Delay" for a single travel time residual. |
| phase description | phase name | E.g. `cake:P`; always in code format. |
| station code (NSL) | | Network, station and location, e.g. `GE.RUE.`. "NSL code" and "network-station code" are fine too. |

## Formatting

- **Code format** for config fields (`window_length`), values (`"MAD"`), file names (`search.json`), commands (`qseek search`) and phase descriptions.
- **Code blocks** get a `title` that says what the block does: `title="Start the search"`.
- **Config snippets** are valid JSON fragments that the reader can paste into their config.
- **Admonitions** sparingly: `tip` for tuning advice, `warning` for things that break a search or cost hours of compute, `abstract` for citations. At most one or two per page.
- **Figures** have a caption that says what is shown and where the data comes from.
- **Links**: link a module or concept at its first mention on a page, not every time.

## Examples

**Marketing → concrete**

> ~~The detector is leveraging Pyrocko and SeisBench, it is highly-performant and can search massive data sets for seismic activity efficiently.~~
>
> Qseek builds on Pyrocko and SeisBench. It scans a 600 GB data set, about 700 years of waveforms, in two days on a 64-core machine with an Nvidia A100 GPU.

**Passive → direct**

> ~~Station nodes are calculated for every node and station in the search volume.~~
>
> Qseek calculates a weight for every pair of station and node in the search volume.

**Field list → guidance**

> ~~Station weights play a crucial role for successful detection and location of events.~~
>
> By default the closest 4 stations get full weight and more distant stations are tapered over twice the mean interstation distance. If distant stations should still contribute to the stack, raise `waterlevel`: stations outside the taper are lifted by this fraction of the full weight.
