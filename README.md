# Fast Hankel Transform (NUFHT) - Fresnel Integral Evaluation

## Overview

This C code implements a Nonuniform Fast Hankel Transform (NUFHT) for the evaluation of Fresnel-type integrals F(w,y) on a w-y grid. The NUFHT algorithm is translated from Julia (FastTransforms.jl) to C, generalizing to a batch mode to allow several NUFHTs with the same grid to be evaluated simultaneously, and also to allow for parallel processing.

## Project Structure

The project consists of the following files and directories:

- **src/**: Contains the source code files for the NUFHT implementation.
  - `axifresnel_jan13.c`: Implements the parallel evaluation of the Fresnel-type integral using NUFHT.
  - `fast_hankel_nufht.c`: Contains the main algorithm for the NUFHT, including setup and execution functions.
  - `fast_hankel_nufht.h`: Header file declaring functions and data structures for NUFHT operations.
  - `expansions.c`: Implements various expansion techniques used in the NUFHT.
  - `expansions.h`: Header file declaring functions and data structures for expansion techniques.
  - `bounds.c`: Manages tolerance and nu lookup tables for the NUFHT.
  - `bounds.h`: Header file declaring functions and data structures for bounds management.

- **data/**: Contains Gauss-Legendre point files used for numerical integration.
  - `gl1000`, `gl2000`, `gl3000`, `gl4000`, `gl5000`, `gl6000`, `gl7000`, `gl8000`, `gl9000`, `gl10000`: Each file contains corresponding Gauss-Legendre points and weights.
  - `nufht_tables.cache`: contains numbers used by the nufht.  The code will re-generate it if it is not found.

- **Makefile**: Contains build instructions for compiling the C files into an executable, specifying compiler options and dependencies.

## Usage Instructions

1. **Build the Project**: Navigate to the project directory and run the following command to compile the source code:
   ```
   make
   ```

   You will probably need to specify the proper include/library paths for fftw, gsl, openmp, and finufft

   The potential is given as psi_x in compute_cj() om axifresnel_jan13.c.

   The executable file is currently axifresnel

2. **Run the Program**: After building, execute the program to compute the Fresnel integral:
   ```
   make run
   ```

3. **Output**: The results will be written to `axifresnel.out`, containing the computed values of the Fresnel integral.

## Dependencies

This project relies on the following libraries:
- **GSL**: For Bessel function evaluations.
- **FINUFFT**: For efficient nonuniform FFT computations.
- **FFTW**:  called by FINUFFT

Ensure that these libraries are installed and properly linked in the Makefile.

## Contribution

Contributions to this project are welcome. Please fork the repository and submit a pull request for any changes or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.