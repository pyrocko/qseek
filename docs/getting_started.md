# Getting Started

## Installation

The installation is straight-forward using pip or pipx.

```sh title="From GitHub"
pip install git+https://github.com/pyrocko/qseek
```

```sh title="From GitHub the Development Branch"
pip install git+https://github.com/pyrocko/qseek@dev
```

or

```sh title="Using pipx"
pipx install git+https://github.com/pyrocko/qseek
```

!!! note "HPC/Cluster Users"
    This package requires a C compiler to build extensions. If you don't load one, the installation will fail with a compilation error.

    **Example Error:**
    ```text
    building 'qseek.ext.array_tools' extension
    error: command 'gcc' failed: No such file or directory
    ERROR: Failed building editable for qseek
    ```

    **Solution:** Run `module load gcc` (or your cluster's equivalent) before installing.
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
