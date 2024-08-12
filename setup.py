#!/usr/bin/env python
import numpy
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "qseek.ext.array_tools",
            sources=["src/qseek/ext/array_tools.c"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-fopenmp", "-O3", "-flto"],
            extra_link_args=["-lgomp"],
        ),
    ]
)
