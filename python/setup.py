from setuptools import setup, Extension
import os
import numpy as np

FINUFFT_INC = os.environ.get('FINUFFT_INC', '/Users/kamion/SCIENCE/FINUFFT/finufft/include')
FINUFFT_LIBDIR = os.environ.get('FINUFFT_LIBDIR', '/Users/kamion/SCIENCE/FINUFFT/finufft/lib')
GSL_INC = os.environ.get('GSL_INC', '/opt/homebrew/Cellar/gsl/2.8/include')
GSL_LIBDIR = os.environ.get('GSL_LIBDIR', '/opt/homebrew/Cellar/gsl/2.8/lib')
FFTW_LIBDIR = os.environ.get('FFTW_LIBDIR', '/opt/homebrew/opt/fftw/lib')
OMP_LIBDIR = os.environ.get('OMP_LIBDIR', '/opt/homebrew/opt/libomp/lib')

src_files = [
    '../src/_pynufht.c',
    '../src/fast_hankel_nufht.c',
    '../src/expansions.c',
    '../src/bounds.c',
]

ext = Extension(
    '_pynufht',
    sources=src_files,
    include_dirs=['../src', FINUFFT_INC, GSL_INC, np.get_include()],
    library_dirs=[FINUFFT_LIBDIR, GSL_LIBDIR, FFTW_LIBDIR, OMP_LIBDIR],
    libraries=['finufft', 'fftw3', 'gsl', 'gslcblas', 'omp', 'm'],
    extra_compile_args=['-O3', '-march=native'],
    extra_link_args=[],
)

setup(
    name='pynufht',
    version='0.1',
    description='Simple Python bindings for NUFHT',
    ext_modules=[ext],
)
