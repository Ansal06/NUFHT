/*
 * test_nufht.c
 *
 * Standalone test that checks nufht_batch() against analytic spherical
 * Bessel transforms g(k) = int_0^inf dr r^2 j_0(k r) f(r)
 * for f(r) = (1 + r^2)^(-n) with n = 1 and n = 3.
 *
 * Using j_0(x) = sqrt(pi/(2x)) J_{1/2}(x), we compute
 *   G_{1/2}(k) = int_0^inf dr r^{3/2} f(r) J_{1/2}(k r)
 * and then g(k) = sqrt(pi/(2k)) * G_{1/2}(k).
 * Analytic solutions:
 *   n=2: g(k) = (pi/4 )e^{-k}
 *   n=3: g(k) = (pi/16) (1 + k) e^{-k}
 */

#include "fast_hankel_nufht.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

/* Load Gauss-Legendre points from a two-column file: abscissa weight */
static int load_gl_points(const char *filename, int max_n,
                          double *absc, double *wght)
{
    FILE *f = fopen(filename, "r");
    if (!f) return 0;
    int n = 0;
    while (n < max_n && fscanf(f, "%lf %lf", &absc[n], &wght[n]) == 2) n++;
    fclose(f);
    return n;
}

/* Rescale abscissas from [-1,1] to [0,x_max], adjust weights accordingly */
static void rescale_gl_points(double *absc, double *wght, int n,
                              double x_max)
{
    for (int k = 0; k < n; ++k) {
        absc[k] = (x_max / 2.0) * (absc[k] + 1.0);
        wght[k] *= (x_max / 2.0);
    }
}

int main(int argc, char **argv)
{
    const char *gl_file = (argc > 1) ? argv[1] : "data/gl10000";
    int Nk = (argc > 2) ? atoi(argv[2]) : 100;
    double kmax = (argc > 3) ? atof(argv[3]) : 10.0;
    double rmax = (argc > 4) ? atof(argv[4]) : 10.0;

    const int MAX_GL = 10000;
    double *absc = malloc(MAX_GL * sizeof(double));
    double *wght = malloc(MAX_GL * sizeof(double));
    if (!absc || !wght) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    int n_gl = load_gl_points(gl_file, MAX_GL, absc, wght);
    if (n_gl <= 0) {
        fprintf(stderr, "Failed to load GL points from %s\n", gl_file);
        free(absc); free(wght);
        return 1;
    }

    /* choose radial domain [0, rmax] and rescale GL points */
    rescale_gl_points(absc, wght, n_gl, rmax);

    /* Prepare cs (sampled integrand * weights) for batch n={2,3} */
    const int batch = 2;
    int n_vals[2] = {2, 3};
    double mu = 0.5;
    double *cs = malloc((size_t)n_gl * batch * sizeof(double));
    if (!cs) { fprintf(stderr, "cs alloc failed\n"); return 1; }
    for (int j = 0; j < n_gl; ++j) {
        double r = absc[j];
        for (int b = 0; b < batch; ++b) {
            int n = n_vals[b];
            double f = pow(1.0 + r * r, -n);
            cs[j + (size_t)n_gl * b] = pow(r, mu + 1.0) * f * wght[j];
        }
    }

    /* k grid */
    double *k_grid = malloc(Nk * sizeof(double));
    if (!k_grid) { fprintf(stderr, "k_grid alloc failed\n"); return 1; }
    for (int ik = 0; ik < Nk; ++ik) k_grid[ik] = kmax * (double)(ik + 1) / (double)Nk;

    /* NUFHT plan and scratch */
    NufhtOptions opt = nufht_default_options();
    opt.tol = 1e-8;

    NufhtPlan plan;
    if (nufht_plan_init(&plan, mu, absc, n_gl, k_grid, Nk, &opt) != 0) {
        fprintf(stderr, "nufht_plan_init failed\n");
        free(absc); free(wght); free(cs); free(k_grid);
        return 1;
    }

    NufhtScratch scratch;
    if (nufht_scratch_init(&scratch, &plan) != 0) {
        fprintf(stderr, "nufht_scratch_init failed\n");
        nufht_plan_free(&plan);
        free(absc); free(wght); free(cs); free(k_grid);
        return 1;
    }

    double *gs = calloc((size_t)Nk * batch, sizeof(double));
    if (!gs) { fprintf(stderr, "gs alloc failed\n"); return 1; }

    int ret = nufht_batch(&plan, cs, n_gl, gs, Nk, batch, &scratch);
    if (ret != 0) {
        fprintf(stderr, "nufht_batch returned %d\n", ret);
        nufht_scratch_free(&scratch);
        nufht_plan_free(&plan);
        free(absc); free(wght); free(cs); free(k_grid); free(gs);
        return 1;
    }

    /* analytic spherical Bessel transform g(k) for each n */
    double pi = acos(-1.0);
    for (int b = 0; b < batch; ++b) {
        int n = n_vals[b];
        double max_abs_err = 0.0;
        double max_rel_err = 0.0;
        printf("\n# Batch column n=%d\n", n);
        printf("# %12s  %15s  %15s  %12s\n", "k", "analytic", "numeric", "rel_err");
        for (int ik = 0; ik < Nk; ++ik) {
            double k = k_grid[ik];
            double G = gs[ik + (size_t)Nk * b];
            double numeric = sqrt(pi / (2.0 * k)) * G;
            double analytic = 0.0;
            if (n == 2) {
                analytic = (pi / 4.0) * exp(-k);
            } else if (n == 3) {
                analytic = (pi / 16.0) * (1.0 + k) * exp(-k);
            }
            double abs_err = fabs(numeric - analytic);
            double rel_err = abs_err / (fabs(analytic) + 1e-16);
            if (abs_err > max_abs_err) max_abs_err = abs_err;
            if (rel_err > max_rel_err) max_rel_err = rel_err;
            printf("%12.6e  %15.8e  %15.8e  %12.3e\n", k, analytic, numeric, rel_err);
        }

        fprintf(stderr, "max abs err = %.3e, max rel err = %.3e\n", max_abs_err, max_rel_err);
    }

    nufht_scratch_free(&scratch);
    nufht_plan_free(&plan);
    free(absc); free(wght); free(cs); free(k_grid); free(gs);
    return 0;
}
