# Getting Started

## Installation

The installation is straight-forward:

```sh title="From GitHub"
pip install git+https://github.com/pyrocko/qseek
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

??? abstract "Minimal Configuration Example"
    Here is a minimal JSON configuration for Qseek
    ```bash exec='on' result='json'
    qseek config
    ```

For more details and information about the component, head over to [details of the modules](components/seismic_data.md).

## Starting the Search

Once happy, start the `qseek` CLI.

```sh title="Start earthquake detection"
qseek search my-search.json
```
