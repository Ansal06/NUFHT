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
static double complex compute_cj(double w, double u_j, double xi_j,
                                 double u_max_w)
{
    if (u_max_w <= 0.0) return 0.0;
    /* Smoothly taper contributions beyond the per-w cutoff */
    double du = 0.02 * u_max_w;
    if (du <= 0.0) du = 1e-12;
    double window_u = 0.5 * (1.0 - tanh((u_j - u_max_w) / du));

    double x = u_j / w;
    double Rmax = u_max_w / w;
    double arg = x - 0.75 * Rmax;
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
    return window_u * xi_j * u_j * phaseA * (phasePsi - 1.0);
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
    double y_min = 0.0, y_max = 1.0;    /* spatial grid */

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

    /* Buffer to collect all output lines (indexed by iw*n_y + iy) */
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

    /* Process in groups by n_gl, using shared u_j grid per group */
    #pragma omp parallel for schedule(dynamic, 1) num_threads(10)
    for (int gi = 0; gi < 10; ++gi) {
        if (error_flag) continue;
        int n_gl = gl_sizes[gi];
        int gl_idx = gi;

        /* Count how many w fall in this n_gl bin */
        int group_count = 0;
        for (int iw = 0; iw < n_w; ++iw) {
            if (select_n_gl(w_grid[iw]) == n_gl) {
                group_count++;
            }
        }
        if (group_count == 0) continue;

        /* Max w in this bin -> common u_max to share u_j grid */
        double w_max_group = 0.0;
        for (int iw = 0; iw < n_w; ++iw) {
            double w = w_grid[iw];
            if (select_n_gl(w) == n_gl && w > w_max_group) {
                w_max_group = w;
            }
        }
        double u_max = w_max_group * sqrt((double)n_gl / (2.0 * w_max_group));

        double *gl_absc = gl_sets[gl_idx].absc;
        double *gl_wght = gl_sets[gl_idx].wght;
        n_gl = gl_sets[gl_idx].n_gl;

        double *u_j = malloc((size_t)n_gl * sizeof(double));
        double *xi_j = malloc((size_t)n_gl * sizeof(double));
        if (!u_j || !xi_j) {
            fprintf(stderr, "allocation failed\n");
            #pragma omp critical
            { error_flag = 1; }
            free(u_j);
            free(xi_j);
            continue;
        }

        memcpy(u_j, gl_absc, (size_t)n_gl * sizeof(double));
        memcpy(xi_j, gl_wght, (size_t)n_gl * sizeof(double));
        rescale_gl_points(u_j, xi_j, n_gl, u_max);

        /* Batch size = 2 * group_count (real+imag for each w in group) */
        int batch = 2 * group_count;
        double *cs_batch = malloc((size_t)n_gl * (size_t)batch * sizeof(double));
        double *gs_batch = malloc((size_t)n_y * (size_t)batch * sizeof(double));
        if (!cs_batch || !gs_batch) {
            fprintf(stderr, "batch allocation failed\n");
            free(u_j);
            free(xi_j);
            free(cs_batch);
            free(gs_batch);
            #pragma omp critical
            { error_flag = 1; }
            continue;
        }

        /* Pack coefficients for each w in the group */
        int t = 0;
        for (int iw = 0; iw < n_w; ++iw) {
            double w = w_grid[iw];
            if (select_n_gl(w) != n_gl) continue;

            double u_max_w = w * sqrt((double)n_gl / (2.0 * w));

            for (int j = 0; j < n_gl; ++j) {
                double complex cj = compute_cj(w, u_j[j], xi_j[j], u_max_w);
                cs_batch[(size_t)t * (size_t)n_gl + (size_t)j] = creal(cj);
                cs_batch[(size_t)(t + 1) * (size_t)n_gl + (size_t)j] = cimag(cj);
            }
            t += 2;
        }

        memset(gs_batch, 0, (size_t)n_y * (size_t)batch * sizeof(double));

        NufhtPlan local_plan;
        NufhtScratch scratch;
        if (nufht_plan_init(&local_plan, 0.0, u_j, n_gl, y_grid, n_y, &opt_plan) != 0) {
            fprintf(stderr, "plan_init failed for n_gl=%d\n", n_gl);
            free(u_j);
            free(xi_j);
            free(cs_batch);
            free(gs_batch);
            #pragma omp critical
            { error_flag = 1; }
            continue;
        }
        if (nufht_scratch_init(&scratch, &local_plan) != 0) {
            fprintf(stderr, "scratch_init failed for n_gl=%d\n", n_gl);
            nufht_plan_free(&local_plan);
            free(u_j);
            free(xi_j);
            free(cs_batch);
            free(gs_batch);
            #pragma omp critical
            { error_flag = 1; }
            continue;
        }

        int ret_batch = nufht_batch(&local_plan, cs_batch, n_gl, gs_batch, n_y, batch, &scratch);
        if (ret_batch != 0) {
            fprintf(stderr, "nufht_batch failed for n_gl=%d\n", n_gl);
            nufht_scratch_free(&scratch);
            nufht_plan_free(&local_plan);
            free(u_j);
            free(xi_j);
            free(cs_batch);
            free(gs_batch);
            #pragma omp critical
            { error_flag = 1; }
            continue;
        }

        /* Unpack results into output_buffer */
        t = 0;
        for (int iw = 0; iw < n_w; ++iw) {
            double w = w_grid[iw];
            if (select_n_gl(w) != n_gl) continue;
            for (int iy = 0; iy < n_y; ++iy) {
                double y = y_grid[iy];
                double complex prefac_y = cexp(I * w * y * y / 2.0) / (I * w);
                double G_re = gs_batch[(size_t)t * (size_t)n_y + (size_t)iy];
                double G_im = gs_batch[(size_t)(t + 1) * (size_t)n_y + (size_t)iy];
                double complex G_sum = G_re + I * G_im;
                double complex F_y = 1.0 + prefac_y * G_sum;

                size_t out_idx = (size_t)iw * (size_t)n_y + (size_t)iy;
                output_buffer[out_idx].w = w;
                output_buffer[out_idx].y = y;
                output_buffer[out_idx].re = creal(F_y);
                output_buffer[out_idx].im = cimag(F_y);
            }
            t += 2;
        }

        nufht_scratch_free(&scratch);
        nufht_plan_free(&local_plan);
        free(u_j);
        free(xi_j);
        free(cs_batch);
        free(gs_batch);
    }

    if (error_flag) {
        free(output_buffer);
        fclose(out);
        return 1;
    }

    /* Write all collected results to file */
    output_count = n_w * n_y;
    printf("outputcount=%d\n", output_count);
    for (int iw = 0; iw < n_w; ++iw) {
        for (int iy = 0; iy < n_y; ++iy) {
            size_t out_idx = (size_t)iw * (size_t)n_y + (size_t)iy;
            fprintf(out, "  %12.6e  %12.6e  %+15.8e  %+15.8e\n",
                    output_buffer[out_idx].w, output_buffer[out_idx].y,
                    output_buffer[out_idx].re, output_buffer[out_idx].im);
        }
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
