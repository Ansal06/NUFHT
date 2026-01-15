/**
 * \file axifresnel_jan13.c
 * \brief Parallel evaluation of Fresnel-type integral on w×y grid via NUFHT.
 *
 * Computes F(w,y) = 1 + (e^{iwy²/2}/iw) Σ c_j(w) J_0(u_j y) for each (w,y).
 * The sum is evaluated via NUFHT for vectorization over y.
 * OpenMP parallelizes over w values on multi-core systems.
 */

#include "fast_hankel_nufht.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <omp.h>


/**
 * \brief Load Gauss–Legendre points from file.
 *  Reads n_gl pairs of (abscissa, weight) from filename.
 *  Returns number of points read.
 */
static int load_gl_points(const char *filename, int max_n,
                          double *absc, double *wght)
{
    FILE *f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "Cannot open GL file: %s\n", filename);
        return 0;
    }

    int n = 0;
    while (n < max_n && fscanf(f, "%lf %lf", &absc[n], &wght[n]) == 2) {
        n++;
    }
    fclose(f);
    return n;
}

/**
 * \brief Rescale GL points from [-1,1] to [0, u_max].
 *  Transforms abscissas and weights accordingly.
 */
static void rescale_gl_points(double *absc, double *wght, int n,
                              double u_max)
{
    for (int k = 0; k < n; ++k) {
        /* Map [-1,1] to [0, u_max]: u = (u_max/2)*(x+1) */
        absc[k] = (u_max / 2.0) * (absc[k] + 1.0);
        wght[k] *= (u_max / 2.0);
    }
}

/**
 * \brief Compute coefficient c_j(w) for a given w and u_j.
 *  c_j(w) = ξ_j u_j exp{ i u_j² / (2w) [e^{-i w ψ(u_j/w)} - 1] }
 * xi_j is the weight associated with u_j
 */
static double complex compute_cj(double w, double u_j, double xi_j,double Rmax)
{
    double x = u_j / w;
    double arg = x - 0.75*Rmax;
    double numerator = 0.5 * (1.0 - tanh(arg));   // windowing for potential
    double xc= 0.05;
    double psi_x = sqrt(x*x + xc*xc) + xc * log((2.0*xc/(sqrt(x*x+xc*xc)+xc)));  // cored isothermal sphere from 2210.05658 Tambalo et al.
    // double psi_x = x;   //  SIS profile
    // double psi_x = log(0.0001+x); //  point mass
    psi_x *= numerator;  // psi(x) windowed to zero at large x
    

    // Phase terms
    double complex phaseA  = cexp(I * (u_j * u_j) / (2.0 * w));   // exp(i u^2 / 2w)
    double complex phasePsi = cexp(-I * w * psi_x);              // exp(-i w ψ)
    // Correct coefficient
    return xi_j * u_j * phaseA * (phasePsi - 1.0);
}

/**
 * \brief Select n_gl based on w value.
 *  Returns n_gl: 1000 for w<10, 2000 for 10≤w<20, ..., 10000 for 90≤w<100
 */
static int select_n_gl(double w)
{
    if (w < 10.0) return 1000;
    if (w < 20.0) return 2000;
    if (w < 30.0) return 3000;
    if (w < 40.0) return 4000;
    if (w < 50.0) return 5000;
    if (w < 60.0) return 6000;
    if (w < 70.0) return 7000;
    if (w < 80.0) return 8000;
    if (w < 90.0) return 9000;
    return 10000;  /* w >= 90 */
}

int main(void)
{
    /* Parameters */
    int n_w = 120;    /* number of w values */
    int n_y = 150;    /* number of y values */   //  set this to >=150; code is slower at smaller n_y

    double w_min = 0.1, w_max = 100.0;    /* frequency grid */
    double y_min = 0.0, y_max = 5.0;    /* spatial grid */

    /* Allocate grids */
    double *w_grid = malloc((size_t)n_w * sizeof(double));
    double *y_grid = malloc((size_t)n_y * sizeof(double));

    if (!w_grid || !y_grid) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    /* Build w and y grids */
    for (int iw = 0; iw < n_w; ++iw) {
        w_grid[iw] = w_min + (w_max - w_min) * (double)iw / (double)(n_w - 1);
    }
    for (int iy = 0; iy < n_y; ++iy) {
        y_grid[iy] = 0.1 + (y_max - 0.1) * (double)iy / (double)(n_y - 1);
    }

    /* Load 10 sets of GL points from files gl1000 through gl10000 */
    typedef struct {
        int n_gl;
        double *absc;
        double *wght;
    } GLSet;
    
    GLSet gl_sets[10];
    int gl_sizes[10] = {1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};
    
    for (int i = 0; i < 10; ++i) {
        int size = gl_sizes[i];
        gl_sets[i].absc = malloc((size_t)size * sizeof(double));
        gl_sets[i].wght = malloc((size_t)size * sizeof(double));
        
        if (!gl_sets[i].absc || !gl_sets[i].wght) {
            fprintf(stderr, "Allocation failed for GL set %d\n", i);
            return 1;
        }
        
        char gl_file[256];
        snprintf(gl_file, sizeof(gl_file), "data/gl%d", size);
        int n_actual = load_gl_points(gl_file, size, gl_sets[i].absc, gl_sets[i].wght);
        if (n_actual != size) {
            fprintf(stderr, "Warning: GL file %s: requested %d, got %d\n", gl_file, size, n_actual);
        }
        gl_sets[i].n_gl = n_actual;
    }

    /* Output file */
    FILE *out = fopen("axifresnel.out", "w");
    if (!out) {
        fprintf(stderr, "Cannot open output file axifresnel.out\n");
        return 1;
    }

    fprintf(out, "# w          y          Re(F)       Im(F)\n");

    /* Error flag for parallel region */
    int error_flag = 0;

    /* NUFHT options for plan creation */
    NufhtOptions opt_plan = nufht_default_options();
    opt_plan.tol = 1.0e-8;

    /* Buffer to collect all output lines from all threads */
    /* Allocate enough for 48 w x 100 y lines */
    typedef struct {
        double w, y, re, im;
    } OutputLine;
    OutputLine *output_buffer = malloc((size_t)n_w * (size_t)n_y * sizeof(OutputLine));
    if (!output_buffer) {
        fprintf(stderr, "output buffer allocation failed\n");
        fclose(out);
        return 1;
    }
    
    int output_count = 0;

    /* Parallel region: each thread creates its own plan and executes batch */
    /* Using #pragma omp parallel (structured block) like test_nufht.c */
    #pragma omp parallel num_threads(12)
    {
    #pragma omp for collapse(1)
    for (int iw = 0; iw < n_w; ++iw) {
        fprintf(stderr, "Thread starting iw=%d\n", iw);
        double w = w_grid[iw];
        
        /* Select GL set based on w */
        int n_gl = select_n_gl(w);
        int gl_idx = (n_gl / 1000) - 1;  /* Maps 1000->0, 2000->1, ..., 10000->9 */
        
        double *gl_absc = gl_sets[gl_idx].absc;
        double *gl_wght = gl_sets[gl_idx].wght;
        n_gl = gl_sets[gl_idx].n_gl;  /* Use actual loaded count */
        
        double u_max = w * sqrt((double)n_gl/(2.0*w));   //  R = sqrt(n_gl /(2w)),

        /* Allocate per-thread GL arrays and rescale for this w */
        double *u_j = malloc((size_t)n_gl * sizeof(double));
        double *xi_j = malloc((size_t)n_gl * sizeof(double));
        if (!u_j || !xi_j) {
            fprintf(stderr, "allocation failed\n");
            error_flag = 1;
            continue;
        }

        /* Copy and rescale GL points for this w */
        memcpy(u_j, gl_absc, (size_t)n_gl * sizeof(double));
        memcpy(xi_j, gl_wght, (size_t)n_gl * sizeof(double));
        rescale_gl_points(u_j, xi_j, n_gl, u_max);

        /* Compute coefficients c_j(w) for this w and pack for batch (real, imag) */
        double *cs_batch = malloc((size_t)n_gl * 2 * sizeof(double));
        if (!cs_batch) {
            fprintf(stderr, "cs_batch allocation failed\n");
            free(u_j);
            free(xi_j);
            error_flag = 1;
            continue;
        }

        for (int j = 0; j < n_gl; ++j) {
            double complex cj = compute_cj(w, u_j[j], xi_j[j],u_max/w);
            cs_batch[j] = creal(cj);                 /* column 0 */
            cs_batch[n_gl + j] = cimag(cj);          /* column 1 */
        }

     
        fprintf(stderr, "Processing w=%d/%d (w=%.6e)...\n", iw+1, n_w, w);
     

        /* Each thread creates its own plan and scratch - thread-safe pattern */
        NufhtPlan local_plan;
        NufhtScratch scratch;

        fprintf(stderr, "iw=%d: About to call nufht_plan_init with u_j=%p, n_gl=%d, y_grid=%p, n_y=%d\n", 
                iw, (void*)u_j, n_gl, (void*)y_grid, n_y);

        if (nufht_plan_init(&local_plan, 0.0, u_j, n_gl, y_grid, n_y, &opt_plan) != 0) {
            fprintf(stderr, "[Thread %d] plan_init failed for w=%.6e\n", omp_get_thread_num(), w);
            free(u_j);
            free(xi_j);
            free(cs_batch);
            #pragma omp critical
            {
                error_flag = 1;
            }
            continue;
        }

        if (nufht_scratch_init(&scratch, &local_plan) != 0) {
            fprintf(stderr, "[Thread %d] scratch_init failed for w=%.6e\n", omp_get_thread_num(), w);
            nufht_plan_free(&local_plan);
            free(u_j);
            free(xi_j);
            free(cs_batch);
            #pragma omp critical
            {
                error_flag = 1;
            }
            continue;
        }

        /* Allocate output buffers for batch=2 (real and imaginary) */
        double *gs_batch = malloc((size_t)n_y * 2 * sizeof(double));
        if (!gs_batch) {
            fprintf(stderr, "[Thread %d] output buffer allocation failed\n", omp_get_thread_num());
            nufht_scratch_free(&scratch);
            nufht_plan_free(&local_plan);
            free(u_j);
            free(xi_j);
            free(cs_batch);
            #pragma omp critical
            {
                error_flag = 1;
            }
            continue;
        }
        memset(gs_batch, 0, (size_t)n_y * 2 * sizeof(double));

        /* Execute NUFHT for real and imaginary parts in one batch (batch=2) */
        int ret_batch = nufht_batch(&local_plan, cs_batch, n_gl, gs_batch, n_y, 2, &scratch);
        if (ret_batch != 0) {
            fprintf(stderr, "[Thread %d] nufht_batch(batch=2) failed for w=%.6e\n", omp_get_thread_num(), w);
            nufht_scratch_free(&scratch);
            nufht_plan_free(&local_plan);
            free(u_j);
            free(xi_j);
            free(cs_batch);
            free(gs_batch);
            #pragma omp critical
            {
                error_flag = 1;
            }
            continue;
        }

        /* Combine results: F(w,y) = 1 + (e^{iwy²/2} / iw) * (G_re + i*G_im) */
        for (int iy = 0; iy < n_y; ++iy) {
            double y = y_grid[iy];
            double complex prefac_y = cexp(I * w * y * y / 2.0) / (I * w);
            double G_re = gs_batch[iy];
            double G_im = gs_batch[n_y + iy];
            double complex G_sum = G_re + I * G_im;
            double complex F_y = 1.0 + prefac_y * G_sum;
            
            /* Store in thread-safe buffer position */
            #pragma omp critical(buffer_write)
            {
                output_buffer[output_count].w = w;
                output_buffer[output_count].y = y;
                output_buffer[output_count].re = creal(F_y);
                output_buffer[output_count].im = cimag(F_y);
                output_count++;
            }
        }

        /* Cleanup for this thread's iteration */
        nufht_scratch_free(&scratch);
        nufht_plan_free(&local_plan);
        free(u_j);
        free(xi_j);
        free(cs_batch);
        free(gs_batch);
    }  /* end for loop */
    }  /* end parallel region */

    if (error_flag) {
        free(output_buffer);
        fclose(out);
        return 1;
    }

    /* Write all collected results to file */
    printf("outputcount=%d\n",output_count);
    for (int i = 0; i < output_count; ++i) {
        fprintf(out, "  %12.6e  %12.6e  %+15.8e  %+15.8e\n",
                output_buffer[i].w, output_buffer[i].y, 
                output_buffer[i].re, output_buffer[i].im);
    }

    fclose(out);

    /* Cleanup */
    free(output_buffer);
    free(w_grid);
    free(y_grid);
    
    /* Free all 10 GL sets */
    for (int i = 0; i < 10; ++i) {
        free(gl_sets[i].absc);
        free(gl_sets[i].wght);
    }

    printf("Results written to axifresnel.out\n");
    return 0;
}
