from setuptools import setup
from Cython.Build import cythonize
import numpy as np

# Might be able to remove extra_compile_args, leaving in until I have better information
setup(ext_modules=cythonize('gmz_tree.pyx'), extra_compile_args=['-O3'], include_dirs=[np.get_include()])
