# -*- coding: utf-8 -*-
"""Build Cython extension for exact depth-2 expectimax tree builder."""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


ext = Extension(
    'algo.eval._expectimax2',
    sources=['algo/eval/_expectimax2.pyx'],
    include_dirs=[np.get_include()],
    language='c++',
    extra_compile_args=['-O3', '-Wno-unused-function'],
)

setup(
    name='mahjong_ai_expectimax2',
    ext_modules=cythonize([ext], language_level=3),
    zip_safe=False,
)
