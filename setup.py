#!/usr/bin/env python
import platform

import numpy
import simde_py
from setuptools import Extension, setup


def is_macos() -> bool:
    return platform.system() == "Darwin"


def is_x86_64() -> bool:
    return platform.machine() == "x86_64"


def is_arm64() -> bool:
    return platform.machine() == "arm64"


link_args = ["-lomp"] if is_macos() else ["-lgomp"]
extra_compile_args = ["-Xpreprocessor"] if is_macos() else []

setup(
    ext_modules=[
        Extension(
            "qseek.ext.array_tools",
            sources=["src/qseek/ext/array_tools.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=[
                *extra_compile_args,
                "-fopenmp",
                "-O3",
                "-flto",
            ],
            extra_link_args=link_args,
        ),
        Extension(
            "qseek.ext.delay_sum",
            sources=["src/qseek/ext/delay_sum.c"],
            include_dirs=[numpy.get_include(), simde_py.get_include()],
            extra_compile_args=[
                *extra_compile_args,
                "-fopenmp",
                "-O3",
                "-flto",
                "-mavx2" if is_x86_64() else "",
                "-mfma" if is_x86_64() else "",
            ],
            extra_link_args=link_args,
        ),
    ]
)
