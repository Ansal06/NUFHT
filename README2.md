# Batched NUFHT (C + Python) — Installation Guide (README2)

This document provides **detailed, user-friendly installation instructions** for the NUFHT C demo and the Python extension, including an **HPC/cluster-friendly** workflow.

It is based on common build requirements for the project (**FINUFFT + FFTW + GSL + OpenMP**) and a real interactive cluster install session that required:
- a compiler toolchain (GCC) and (sometimes) CMake,
- Conda-provided dependencies where possible,
- explicit include/library environment variables when auto-discovery failed,
- `pip` build flags such as `--no-build-isolation`,
- a NumPy version constraint (`numpy<2.4`) in that environment.

---

## Table of contents

- [Quick summary](#quick-summary)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
  - [Compilers / build tools](#compilers--build-tools)
  - [Libraries](#libraries)
  - [Python](#python)
- [Install paths](#install-paths)
  - [Path A (recommended on clusters): Conda environment + conda-forge libs](#path-a-recommended-on-clusters-conda-environment--conda-forge-libs)
  - [Path B: Build FINUFFT from source (fallback)](#path-b-build-finufft-from-source-fallback)
- [Build and run the C demo](#build-and-run-the-c-demo)
- [Build and install the Python extension](#build-and-install-the-python-extension)
  - [Important note: install from `python/`, not repo root](#important-note-install-from-python-not-repo-root)
  - [Install (preferred): `pip install` from `python/`](#install-preferred-pip-install-from-python)
  - [Build in-place (developer mode): `setup.py build_ext --inplace`](#build-in-place-developer-mode-setup.py-build_ext---inplace)
  - [Verify the install](#verify-the-install)
- [Troubleshooting](#troubleshooting)
  - [1) `pip install git+https://...` fails](#1-pip-install-githttps-fails)
  - [2) Build cannot find FINUFFT / GSL / FFTW headers or libs](#2-build-cannot-find-finufft--gsl--fftw-headers-or-libs)
  - [3) Runtime error: cannot load `libfinufft.so` / missing shared library](#3-runtime-error-cannot-load-libfinufftso--missing-shared-library)
  - [4) NumPy/ABI issues](#4-numpyabi-issues)
  - [5) OpenMP issues (`libgomp` / `libomp`)](#5-openmp-issues-libgomp--libomp)
- [Notes on portability](#notes-on-portability)

---

## Quick summary

If you are on an HPC cluster and want the Python wrapper working reliably, the most robust approach is:

1. Create a conda env (Python 3.11 recommended).
2. Install dependencies via conda-forge: `finufft`, `fftw`, `gsl`.
3. Install build tools if needed (or load modules such as `gcc` / `cmake`).
4. Pin NumPy if needed: `numpy<2.4`.
5. Clone the repo, then install from the `python/` subdirectory using pip and disable build isolation.

```bash
conda create -n nufht python=3.11 -y
conda activate nufht

conda install -c conda-forge --override-channels finufft fftw gsl -y

python -m pip install -U pip setuptools wheel
python -m pip install "numpy<2.4"

git clone https://github.com/marckamion/NUFHT.git
cd NUFHT/python

# If your build system cannot auto-detect libs, export these first:
export FINUFFT_INC="$CONDA_PREFIX/include"
export FINUFFT_LIBDIR="$CONDA_PREFIX/lib"
export GSL_INC="$CONDA_PREFIX/include"
export GSL_LIBDIR="$CONDA_PREFIX/lib"
export FFTW_LIBDIR="$CONDA_PREFIX/lib"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

python -m pip install -v . --no-build-isolation --no-cache-dir
```

Then verify:

```bash
python -c "import _pynufht; print('OK', _pynufht.__file__)"
```

---

## Project structure

- **src/** — C implementation and test harness
- **python/** — Python extension and build scripts
  - `setup.py`: builds the `_pynufht` extension
  - `run_pynufht_test.py`: example Python test
- **data/** — Gauss–Legendre points and cached tables
- **notebooks/** — example notebooks
- **Makefile** — build rules for the C demo

---

## Prerequisites

### Compilers / build tools

You need a working C/C++ toolchain and a way to compile Python extensions.

On many clusters, you may need to load compiler modules, e.g.:

```bash
module load gcc
# optionally:
module load cmake
```

If you use conda-forge for tools:

```bash
conda install -c conda-forge --override-channels cmake make compilers -y
```

### Libraries

The core dependencies are:

- FINUFFT (and its FFTW dependency)
- GSL
- OpenMP runtime (often provided by the compiler toolchain)

The easiest way on many Linux systems (including clusters) is to install via conda-forge:

```bash
conda install -c conda-forge --override-channels finufft fftw gsl -y
```

If `finufft` is not available on your system, see [Path B](#path-b-build-finufft-from-source-fallback).

### Python

You need:
- Python (this guide uses 3.11 as a good default)
- pip, setuptools, wheel
- NumPy

Upgrade packaging tools first:

```bash
python -m pip install -U pip setuptools wheel
```

If you see build or runtime issues with newer NumPy, pin to a known-good range:

```bash
python -m pip install "numpy<2.4" --force-reinstall
python -c "import numpy as np; print(np.__version__)"
```

(If the project later updates to support newer NumPy, this constraint can be relaxed.)

---

## Install paths

### Path A (recommended on clusters): Conda environment + conda-forge libs

This matches the most reliable “it just works” approach from a real cluster install.

1) Create and activate an environment:

```bash
conda create -n nufht python=3.11 -y
conda activate nufht
```

2) Install compiled dependencies:

```bash
conda install -c conda-forge --override-channels finufft fftw gsl -y
```

3) Upgrade pip tooling and install/pin NumPy:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install "numpy<2.4"
```

4) Clone NUFHT:

```bash
git clone https://github.com/marckamion/NUFHT.git
cd NUFHT
```

Proceed to [Build and install the Python extension](#build-and-install-the-python-extension).

---

### Path B: Build FINUFFT from source (fallback)

Use this if:
- you cannot install `finufft` via conda-forge,
- or your cluster module system does not provide it,
- or the build cannot locate shared libraries in the conda environment.

1) Choose an install prefix (example: in your home directory):

```bash
export FINUFFT_PREFIX="$HOME/finufft-install"
mkdir -p "$FINUFFT_PREFIX"
```

2) Clone and build FINUFFT:

```bash
git clone https://github.com/flatironinstitute/finufft.git
cd finufft
mkdir -p build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX="$FINUFFT_PREFIX" -DBUILD_SHARED_LIBS=ON
make -j
make install
```

3) Export include/lib paths for NUFHT build:

> Note: some systems install libraries to `lib64` instead of `lib`.

```bash
export FINUFFT_INC="$FINUFFT_PREFIX/include"
export FINUFFT_LIBDIR="$FINUFFT_PREFIX/lib"
if [ ! -d "$FINUFFT_LIBDIR" ] && [ -d "$FINUFFT_PREFIX/lib64" ]; then
  export FINUFFT_LIBDIR="$FINUFFT_PREFIX/lib64"
fi

# runtime loader must be able to find libfinufft.so
export LD_LIBRARY_PATH="$FINUFFT_LIBDIR:$LD_LIBRARY_PATH"
```

You will still need `fftw` and `gsl` (conda-forge is often easiest):

```bash
conda install -c conda-forge --override-channels fftw gsl -y
```

Then proceed to install NUFHT Python as below.

---

## Build and run the C demo

From the repository root:

```bash
make
./test_nufht
```

If dependencies are not discoverable automatically, you may need to pass paths (examples):

```bash
make FINUFFT_INC=/path/to/finufft/include FINUFFT_LIBDIR=/path/to/finufft/lib \
  FFTW_LIBDIR=/path/to/fftw/lib GSL_INC=/path/to/gsl/include GSL_LIBDIR=/path/to/gsl/lib
```

---

## Build and install the Python extension

### Important note: install from `python/`, not repo root

Although the repository root contains C sources and a Makefile, the **Python extension lives under `python/`**. Installing from the repo root may fail or do something unintended.

Do this instead:

```bash
cd NUFHT/python
```

### Install (preferred): `pip install` from `python/`

1) Ensure the required library include/lib paths are available.

If you installed dependencies via conda-forge into the current env:

```bash
export FINUFFT_INC="$CONDA_PREFIX/include"
export FINUFFT_LIBDIR="$CONDA_PREFIX/lib"
export GSL_INC="$CONDA_PREFIX/include"
export GSL_LIBDIR="$CONDA_PREFIX/lib"
export FFTW_LIBDIR="$CONDA_PREFIX/lib"

# Optional: if you have OpenMP runtime in your environment:
# export OMP_LIBDIR="$CONDA_PREFIX/lib"
# If OpenMP causes issues, you can also try:
# unset OMP_LIBDIR

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

2) Install (verbose) and disable build isolation:

```bash
python -m pip install -v . --no-build-isolation --no-cache-dir
```

Why these flags?
- `-v` helps diagnose build failures.
- `--no-build-isolation` ensures the build uses your environment’s NumPy and library locations.
- `--no-cache-dir` avoids stale wheels or cached build artifacts masking changes.

If you need to reinstall:

```bash
python -m pip uninstall -y pynufht
python -m pip install -v . --no-build-isolation --no-cache-dir
```

### Build in-place (developer mode): `setup.py build_ext --inplace`

If you do not want to install into site-packages and instead want the `.so` built in the `python/` directory:

```bash
cd NUFHT/python
python setup.py build_ext --inplace
```

Then in a script/notebook (from repo root):

```python
import os, sys
sys.path.insert(0, os.path.abspath("python"))
import _pynufht as pn
```

### Verify the install

Basic import test:

```bash
python -c "import _pynufht; print('OK', _pynufht.__file__)"
```

Run the included test:

```bash
python run_pynufht_test.py
# or from repo root:
# python python/run_pynufht_test.py
```

On Linux clusters, you can confirm which shared libraries the extension is linked against:

```bash
python - <<'PY'
import _pynufht, sys
print(_pynufht.__file__)
PY

ldd "$(python -c 'import _pynufht; print(_pynufht.__file__)')" | egrep "finufft|gsl|fftw|libstdc\+\+|libgcc_s|libgomp|libomp" || true
```

---

## Troubleshooting

### 1) `pip install git+https://...` fails

Some users try:

```bash
python -m pip install --no-cache-dir --no-binary :all: git+https://github.com/marckamion/NUFHT.git
```

This may fail because the repo root is not necessarily a standard Python package and/or because build dependencies aren’t available during pip’s isolated build step.

**Fix:** clone the repo and install from the `python/` subdirectory:

```bash
git clone https://github.com/marckamion/NUFHT.git
cd NUFHT/python
python -m pip install -v . --no-build-isolation
```

### 2) Build cannot find FINUFFT / GSL / FFTW headers or libs

Symptoms include compiler errors such as:
- `finufft.h: No such file or directory`
- `gsl/gsl_...: No such file or directory`
- linker errors `cannot find -lfinufft` / `-lgsl` / `-lfftw3`

**Fix:** export include/lib paths before building:

```bash
export FINUFFT_INC="/path/to/include"
export FINUFFT_LIBDIR="/path/to/lib"
export GSL_INC="/path/to/include"
export GSL_LIBDIR="/path/to/lib"
export FFTW_LIBDIR="/path/to/lib"
```

If you used conda-forge:

```bash
export FINUFFT_INC="$CONDA_PREFIX/include"
export FINUFFT_LIBDIR="$CONDA_PREFIX/lib"
export GSL_INC="$CONDA_PREFIX/include"
export GSL_LIBDIR="$CONDA_PREFIX/lib"
export FFTW_LIBDIR="$CONDA_PREFIX/lib"
```

Then reinstall:

```bash
python -m pip install -v . --no-build-isolation --no-cache-dir
```

### 3) Runtime error: cannot load `libfinufft.so` / missing shared library

Symptoms:
- `ImportError: libfinufft.so: cannot open shared object file: No such file or directory`

**Fix:** ensure the runtime loader can find shared libs:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

If FINUFFT was built/installed to a custom prefix:

```bash
export LD_LIBRARY_PATH="$FINUFFT_LIBDIR:$LD_LIBRARY_PATH"
```

Then retry:

```bash
python -c "import _pynufht; print('OK')"
```

### 4) NumPy/ABI issues

If you see errors around NumPy during build, try pinning NumPy before building:

```bash
python -m pip install "numpy<2.4" --force-reinstall
python -m pip install -v . --no-build-isolation --no-cache-dir
```

### 5) OpenMP issues (`libgomp` / `libomp`)

OpenMP runtime availability is system-specific.

- On Linux with GCC, OpenMP often works out of the box.
- On macOS, you may need `libomp` (Homebrew) and to set `OMP_LIBDIR`.
- On some clusters, mixing compiler toolchains and conda runtimes can cause missing `libgomp`/`libstdc++` at runtime.

If you explicitly set `OMP_LIBDIR` and see linking problems, try:

```bash
unset OMP_LIBDIR
```

Then rebuild/reinstall.

---

## Notes on portability

- The **Makefile** prefers `pkg-config` for FINUFFT/FFTW/GSL, but on clusters those packages may not be discoverable via `pkg-config`. In that case, explicit paths (or conda-forge) are recommended.
- For a truly cross-platform build, a future enhancement would be adding a CMake build for both the C demo and Python extension to standardize detection of OpenMP and third-party libraries.

---

## License

MIT License. See `LICENSE` for details.