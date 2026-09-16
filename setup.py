#!/usr/bin/env python
import platform

import cpuinfo
import numpy
import simde_py
from setuptools import Extension, setup

NOOP = "-Wabi"


def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_x86_64() -> bool:
    return platform.machine() == "x86_64"


def is_arm64() -> bool:
    return platform.machine() == "arm64"


def has_flag(flag: str) -> bool:
    info = cpuinfo.get_cpu_info()
    flags = info.get("flags", [])
    return flag in flags


setup(
    ext_modules=[
        Extension(
            "qseek.ext.array_tools",
            sources=["src/qseek/ext/array_tools.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=[
                "-Xpreprocessor" if is_macos() else NOOP,
                "-fopenmp",
                "-O3",
                "-flto",
            ],
            extra_link_args=[
                "-lomp" if is_macos() else "-lgomp",
            ],
        ),
        Extension(
            "qseek.ext.delay_sum",
            sources=["src/qseek/ext/delay_sum.c"],
            include_dirs=[numpy.get_include(), simde_py.get_include()],
            extra_compile_args=[
                "-Xpreprocessor" if is_macos() else NOOP,
                "-fopenmp",
                "-O3",
                "-flto",
                "-mfma" if is_x86_64() and has_flag("fma") else NOOP,
                "-mavx2" if is_x86_64() and has_flag("avx2") else NOOP,
            ],
            extra_link_args=[
                "-lomp" if is_macos() else "-lgomp",
            ],
        ),
    ]
)
