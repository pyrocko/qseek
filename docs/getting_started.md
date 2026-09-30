# Getting Started

## Installation

Qseek requires Python 3.12 or newer. Install it from [PyPI](https://pypi.org/project/qseek/) with pip or pipx.

```sh title="From PyPI"
pip install qseek
```

```sh title="Using pipx"
pipx install qseek
```

Pre-built packages (wheels) are available for:

| Platform | Architecture | Python |
|---|---|---|
| Linux | x86_64, aarch64 | 3.12–3.14 |
| macOS 15 or newer | Apple Silicon (arm64) | 3.12–3.14 |
| macOS 15 or newer | Intel (x86_64) | 3.12 |

!!! note "x86_64 CPUs without AVX2"
    The pre-built x86_64 packages require a CPU with the AVX2 and FMA instruction sets (Intel Haswell, AMD Zen or newer). On older CPUs Qseek warns at startup; install it from source instead.

### Installation from Source

Building from source requires a C compiler with OpenMP, on macOS install it with `brew install libomp`. The build is optimized for the CPU of the installing machine.

```sh title="From source"
pip install --no-binary qseek qseek
```

```sh title="From GitHub"
pip install git+https://github.com/pyrocko/qseek
```

```sh title="From GitHub the Development Branch"
pip install git+https://github.com/pyrocko/qseek@dev
```

## Running Qseek

The main entry point in the executeable is the `qseek` command. The provided command line interface (CLI) and a JSON config file is all what is needed to run the program.

```bash exec='on' result='ansi' source='above'
qseek --help
```

## Initializing a New Project

Once installed you can run the `qseek` executeable to initialize a new project.

```sh title="Initialize new Project"
qseek config > my-search.json
```

Check out the `my-search.json` config file and add your waveform data and velocity models.

??? quote "Minimal Configuration Example"
    Here is a minimal JSON configuration for Qseek.
    ```bash exec='on' result='json'
    qseek config
    ```

For more details and information about the component, head over to [details of the modules](components/configuration.md).

## Starting the Search

Once happy with the configuration, start the `qseek` CLI.

```sh title="Start the earthquake detection and localization"
qseek search my-search.json
```
