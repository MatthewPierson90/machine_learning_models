# cython:language_level=3

from distutils.core import setup
from Cython.Build import cythonize

setup(name="cython_tree_functions", ext_modules=cythonize('cython_tree_functions.pyx'),)
