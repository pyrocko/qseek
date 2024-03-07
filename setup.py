#!/usr/bin/env python
import sysconfig

import numpy
from setuptools import Extension, setup

is_mac = "macos" in sysconfig.get_platform().lower()


extra_compile_args = ["-fopenmp"]
extra_link_args = ["-lgomp"]
if is_mac:
    extra_compile_args = ["-fopenmp=libomp"]
    extra_link_args = ["-lomp"]

setup(
    ext_modules=[
        Extension(
            "qseek.ext.array_tools",
            sources=["src/qseek/ext/array_tools.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-flto", *extra_compile_args],
            extra_link_args=extra_link_args,
        )
    ]
)
