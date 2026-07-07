# -*- coding: utf-8 -*-
"""Build Cython extension for fast eval0."""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext = Extension(
    'algo.eval._fast_eval0',
    sources=['algo/eval/_fast_eval0.pyx'],
    include_dirs=[np.get_include()],
    language='c++',
    extra_compile_args=['-O3', '-Wno-unused-function'],
)

setup(
    name='mahjong_ai_fast_eval0',
    ext_modules=cythonize([ext], language_level=3),
    zip_safe=False,
)
