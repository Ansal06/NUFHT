# Batched NUFHT (C + Python)

## Overview

This project provides a C implementation of a batched Nonuniform Fast Hankel Transform (NUFHT) and a Python wrapper for convenient use in notebooks and scripts. The core C library supports running multiple transforms at once for a shared $(r,k)$ grid, and the Python extension exposes the same batched API for fast experimentation.

The NUFHT algorithm is from "A nonuniform fast Hankel transform," by Paul G. Beckman and Michael O'Neil [arXiv:2411.09583].  The C code is translated from their Julia code FastHankelTransform.jl at https://github.com/pbeckman/FastHankelTransform.jl and then generalized to allow for vectorization.

## Project Structure

- **src/** — C implementation and test harness
  - `fast_hankel_nufht.c` / `fast_hankel_nufht.h`: NUFHT core and public API
  - `expansions.c` / `expansions.h`, `bounds.c` / `bounds.h`: supporting routines
  - `test_nufht.c`: demonstration program using batched `nufht_batch()`
- **python/** — Python extension and build scripts
  - `setup.py`: builds the `_pynufht` extension
  - `run_pynufht_test.py`: example Python test
- **data/** — Gauss–Legendre points and cached tables
- **notebooks/** — example notebooks using the Python wrapper
- **Makefile** — build rules for the C demo

## Build and run the C demo

The C demo is built as `test_nufht` from [src/test_nufht.c](src/test_nufht.c) and the core library sources.

1) Build:

```
make
```

2) Run with defaults:

```
./test_nufht
```

3) Run with explicit arguments:

```
./test_nufht <gl_file> <Nk> <kmax> <rmax>
```

Example:

```
./test_nufht data/gl10000 200 10 50
```

## Python wrapper

The Python wrapper builds a `_pynufht` extension in the **python/** directory.

1) Build the extension in-place:

```
cd python
python setup.py build_ext --inplace
```

2) Use it from Python (from the repository root or notebooks/):

```
import sys, os
sys.path.insert(0, os.path.abspath("python"))
import _pynufht as pn

# Example usage: pn.nufht_batch(nu, r_nodes, cs_batch, k_grid, tol=1e-8)
```

3) Run the included Python test:

```
python python/run_pynufht_test.py
```

## Dependencies

**C:**
- FINUFFT
- FFTW (used by FINUFFT)
- GSL
- OpenMP (for parallel builds)

Set include/library paths in the Makefile as needed for your system.

**Python:**
- Python development headers
- NumPy (build and runtime)

## Notes

- The batched API `nufht_batch()` runs multiple input columns for a single Hankel order $\nu$ on a shared grid.
- For multiple $\nu$ values, create separate plans and call `nufht_batch()` per $\nu$.

## Portability notes

To build on different systems, the Makefile now prefers `pkg-config` for FINUFFT, FFTW, and GSL. If `pkg-config` is not available or the packages are not discoverable, set the paths explicitly:

```
make FINUFFT_INC=/path/to/finufft/include FINUFFT_LIBDIR=/path/to/finufft/lib \
  FFTW_LIBDIR=/path/to/fftw/lib GSL_INC=/path/to/gsl/include GSL_LIBDIR=/path/to/gsl/lib
```

OpenMP flags vary by platform:

- **Linux (GCC/Clang):** `-fopenmp` typically works.
- **macOS (Homebrew libomp):** set `OMP_LIBDIR` and use `-lomp`.
- **Windows (MSVC):** use `/openmp` and adjust the Makefile or build with a Visual Studio project/CMake.

If you prefer a cross-platform generator, consider adding a CMake build that handles compiler and OpenMP differences automatically.

## License

MIT License. See LICENSE for details.