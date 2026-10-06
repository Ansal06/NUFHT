from setuptools import setup, Extension
import os, sys
import numpy as np

PREFIX = os.environ.get("CONDA_PREFIX", sys.prefix)
LIB = os.path.join(PREFIX, "lib")
INC = os.path.join(PREFIX, "include")

FINUFFT_INC    = os.environ.get("FINUFFT_INC", INC)
FINUFFT_LIBDIR = os.environ.get("FINUFFT_LIBDIR", LIB)
GSL_INC        = os.environ.get("GSL_INC", INC)
GSL_LIBDIR     = os.environ.get("GSL_LIBDIR", LIB)
FFTW_LIBDIR    = os.environ.get("FFTW_LIBDIR", LIB)
OMP_LIBDIR     = os.environ.get("OMP_LIBDIR", LIB)

libdirs = [FINUFFT_LIBDIR, GSL_LIBDIR, FFTW_LIBDIR, OMP_LIBDIR]

compile_args = ["-O3"]
if os.environ.get("NUFHT_NATIVE") == "1":      # opt-in only
    compile_args.append("-march=native")

ext = Extension(
    "_pynufht",
    sources=["../src/_pynufht.c", "../src/fast_hankel_nufht.c",
             "../src/expansions.c", "../src/bounds.c"],
    include_dirs=["../src", FINUFFT_INC, GSL_INC, np.get_include()],
    library_dirs=libdirs,
    runtime_library_dirs=libdirs,              # so the .so finds libfinufft at import time
    libraries=["finufft", "fftw3", "gsl", "gslcblas", "omp", "m"],
    extra_compile_args=compile_args,
)

setup(name="pynufht", version="0.1",
      description="Simple Python bindings for NUFHT", ext_modules=[ext])
