---
icon: lucide/terminal
---

# Command line

The `qseek` command runs and manages searches. Every command has a `--help`.

```bash exec='on' result='ansi'
qseek --help
```

## `qseek config`

Print a configuration with all defaults.

```bash exec='on' result='ansi'
qseek config --help
```

## `qseek search`

Start a search.

```bash exec='on' result='ansi'
qseek search --help
```

## `qseek continue`

Continue a stopped search.

```bash exec='on' result='ansi'
qseek continue --help
```

## `qseek snuffler`

Inspect detections, picks and waveforms in Pyrocko Snuffler.

```bash exec='on' result='ansi'
qseek snuffler --help
```

## `qseek explore`

Explore the results in the web UI.

```bash exec='on' result='ansi'
qseek explore --help
```

## `qseek feature-extraction`

Calculate magnitudes and features of a run again.

```bash exec='on' result='ansi'
qseek feature-extraction --help
```

## `qseek modules`

List the available modules, including plugin modules.

```bash exec='on' result='ansi'
qseek modules --help
```

## `qseek export`

Export detections to other formats.

```bash exec='on' result='ansi'
qseek export --help
```

## `qseek clear-cache`

Clear the cache directory.

```bash exec='on' result='ansi'
qseek clear-cache --help
```

## `qseek dump-schemas`

Write the JSON schemas of the data models (for development).

```bash exec='on' result='ansi'
qseek dump-schemas --help
```
