---
icon: lucide/download
---

# Installation

Qseek requires Python 3.12 or newer. Install it from [PyPI](https://pypi.org/project/qseek/):

=== "pip"

    ```sh
    pip install qseek
    ```

=== "pipx"

    ```sh
    pipx install qseek
    ```

=== "uv"

    ```sh
    uv tool install qseek
    ```

Check the installation:

```sh
qseek --version
```

## Supported platforms

Pre-built packages (wheels) are available for:

| Platform | Architecture | Python |
| --- | --- | --- |
| Linux | x86_64, aarch64 | 3.12–3.14 |
| macOS 15 or newer | Apple Silicon (arm64) | 3.12–3.14 |
| macOS 15 or newer | Intel (x86_64) | 3.12 |

!!! note "x86_64 CPUs without AVX2"
    The pre-built x86_64 packages require a CPU with the AVX2 and FMA instruction sets (Intel Haswell, AMD Zen or newer). On older CPUs Qseek warns at startup; install it from source instead.

## Install from source

Building from source requires a C compiler with OpenMP. On macOS, install OpenMP with `brew install libomp`. The build is optimized for the CPU of the installing machine.

```sh title="Build the latest release"
pip install --no-binary qseek qseek
```

```sh title="Build the development version from GitHub"
pip install git+https://github.com/pyrocko/qseek
```

## Next steps

[Quick start: Campi Flegrei](quick-start.md){ .md-button .md-button--primary }
[Set up your own search](first-search.md){ .md-button }
