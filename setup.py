#!/usr/bin/env python
import sysconfig

import numpy
from setuptools import Extension, setup

is_mac = "macos" in sysconfig.get_platform().lower()

extra_link_args = ["-lomp"] if is_mac else ["-lgomp"]

setup(
    ext_modules=[
        Extension(
            "qseek.ext.array_tools",
            sources=["src/qseek/ext/array_tools.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-fopenmp", "-O3", "-flto"],
            extra_link_args=extra_link_args,
        )
    ]
)
