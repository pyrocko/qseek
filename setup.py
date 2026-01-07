#!/usr/bin/env python
import numpy
import simde_py
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
        Extension(
            "qseek.ext.delay_sum",
            sources=["src/qseek/ext/delay_sum.c"],
            include_dirs=[numpy.get_include(), simde_py.get_include()],
            extra_compile_args=[
                "-fopenmp",
                "-O3",
                "-flto",
                "-march=native",
            ],
            extra_link_args=["-lgomp"],
        ),
    ]
)
