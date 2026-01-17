#if 0  /* legacy corrupted content disabled */
/**
 * \file expansions.c
 * \brief Expansion regimes for NUFHT: direct, local (Wimp), and asymptotic.
 *
 * Implements the three complementary strategies used by the transform:
 *  - add_dir: direct Bessel summation for small problems
 *  - add_loc: local Wimp expansion with Chebyshev acceleration (integer nu)
 *  - add_asy: asymptotic expansion accelerated via FINUFFT (large arguments)
 */
#include "fast_hankel_nufht.h"

#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <gsl/gsl_sf_bessel.h>
#include <stdio.h>

#include <finufft.h>   // make sure FINUFFT headers are in your include path

static const double SQRT_2_OVER_PI = 0.0;  /* will init lazily */

/* Helper to lazily initialize sqrt(2/pi) */
static double get_sqrt_2_over_pi(void) {
    static int init = 0;
    static double val = 0.0;
    if (!init) {
        val = sqrt(2.0 / M_PI);
        init = 1;
    }
    return val;
}
#else /* clean implementation */

/**
 * \file expansions.c
 * \brief Expansion regimes for NUFHT: direct, local (Wimp), and asymptotic.
 *
 * Clean, batch-enabled implementation.
 */
#include "fast_hankel_nufht.h"

#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <gsl/gsl_sf_bessel.h>
#include <stdio.h>

#include <finufft.h>

static double get_sqrt_2_over_pi_clean(void) {
    static int init = 0;
    static double val = 0.0;
    if (!init) {
        val = sqrt(2.0 / M_PI);
        init = 1;
    }
    return val;
}

void add_dir(double *gs, double nu,
             const double *rs, const double *cs,
             const double *ws,
             int m, int n)
{
    for (int j = 0; j < n; ++j) {
        double wj = ws[j];
        double sum = gs[j];
        for (int k = 0; k < m; ++k) {
            double arg = wj * rs[k];
            double J = gsl_sf_bessel_Jnu(nu, arg);
            sum += cs[k] * J;
        }
        gs[j] = sum;
    }
}

void add_dir_batch(double *gs, double nu,
                   const double *rs, const double *cs,
                   const double *ws,
                   int m, int n,
                   int batch,
                   int ld_cs, int ld_gs)
{
    for (int j = 0; j < n; ++j) {
        double wj = ws[j];
        for (int k = 0; k < m; ++k) {
            double arg = wj * rs[k];
            double J = gsl_sf_bessel_Jnu(nu, arg);
            for (int t = 0; t < batch; ++t) {
                size_t cs_idx = (size_t)t * (size_t)ld_cs + (size_t)k;
                size_t gs_idx = (size_t)t * (size_t)ld_gs + (size_t)j;
                gs[gs_idx] += cs[cs_idx] * J;
            }
        }
    }
}

void add_loc(double *gs, double nu,
             const double *rs, const double *cs,
             const double *ws,
             int m, int n,
             int K,
             double *cheb_buffer,
             double *bessel_buffer_1,
             double *bessel_buffer_2)
{
    int nu_int = (int)llround(nu);
    if (fabs(nu - (double)nu_int) > 1e-12) {
        fprintf(stderr, "add_loc: nu must be integer\n");
        return;
    }

    int l0 = nu_int / 2;
    int is_odd_nu = (nu_int & 1);
    double r_end = rs[m - 1];

    for (int l = 0; l <= K; ++l) {
        cheb_buffer[l] = 0.0;
    }

    for (int l = 0; l <= K; ++l) {
        int l_is0 = (l == 0);
        double coeff0 = (nu_int == 0) ? (l_is0 ? 1.0 : 2.0) : 0.0;
        double coeff_even = (!(nu_int & 1)) ? (l_is0 ? 1.0 : 2.0) : 0.0;
        double coeff_odd = (nu_int & 1) ? 2.0 : 0.0;

        for (int k = 0; k < m; ++k) {
            double x = rs[k] / r_end;
            if (x > 1.0) x = 1.0;
            if (x < -1.0) x = -1.0;
            double theta = acos(x);

            double val = 0.0;
            if (nu_int == 0) {
                double cl = cos((double)l * theta);
                double tmp = 2.0 * cl * cl - 1.0;
                val = coeff0 * tmp;
            } else if ((nu_int & 1) == 0) {
                double c2l = cos(2.0 * (double)l * theta);
                val = coeff_even * c2l;
            } else {
                double c = cos(((double)(2*l + 1)) * theta);
                val = coeff_odd * c;
            }

            cheb_buffer[l] += val * cs[k];
        }
    }

    for (int j = 0; j < n; ++j) {
        double w = ws[j];
        double z = w * r_end / 2.0;

        for (int l = 0; l <= K; ++l) {
            bessel_buffer_1[l] = 0.0;
        }

        if (nu_int == 0) {
            for (int l = 0; l <= K; ++l) {
                bessel_buffer_1[l] = gsl_sf_bessel_Jn(l, z);
            }
            for (int l = 0; l <= K; ++l) {
                bessel_buffer_1[l] *= bessel_buffer_1[l];
            }
            for (int l = 1; l <= K; l += 2) {
                bessel_buffer_1[l] *= -1.0;
            }
        } else {
            int max_order = l0 + K + is_odd_nu;
            int len_b2 = max_order + 1;

            for (int t = 0; t < len_b2; ++t) {
                bessel_buffer_2[t] = 0.0;
            }
            for (int k = 0; k <= max_order; ++k) {
                bessel_buffer_2[k] = gsl_sf_bessel_Jn(k, z);
            }
            int start_idx = l0 + is_odd_nu;
            for (int l = 0; l <= K; ++l) {
                int idx = start_idx + l;
                bessel_buffer_1[l] = (idx <= max_order) ? bessel_buffer_2[idx] : 0.0;
            }
            for (int t = 0; t <= l0; ++t) {
                int idx_b1 = l0 - t;
                if (idx_b1 >= 0 && idx_b1 <= K) {
                    bessel_buffer_1[idx_b1] *= bessel_buffer_2[t];
                }
            }
            int limit = K - l0;
            for (int q = 0; q < limit; ++q) {
                int idx_b1 = l0 + 1 + q;
                int idx_b2 = 1 + q;
                if (idx_b1 <= K && idx_b2 <= max_order) {
                    bessel_buffer_1[idx_b1] *= bessel_buffer_2[idx_b2];
                }
            }
            for (int l = l0 + 1; l <= K; l += 2) {
                bessel_buffer_1[l] *= -1.0;
            }
        }

        double acc = 0.0;
        for (int l = 0; l <= K; ++l) {
            acc += bessel_buffer_1[l] * cheb_buffer[l];
        }

        gs[j] += acc;
    }
}

void add_loc_batch(double *gs, double nu,
                   const double *rs, const double *cs,
                   const double *ws,
                   int m, int n,
                   int K,
                   int batch,
                   int ld_cs, int ld_gs,
                   double *cheb_buffer_batch,
                   double *bessel_buffer_1,
                   double *bessel_buffer_2)
{
    int nu_int = (int)llround(nu);
    if (fabs(nu - (double)nu_int) > 1e-12) {
        fprintf(stderr, "add_loc_batch: nu must be integer\n");
        return;
    }

    int l0 = nu_int / 2;
    int is_odd_nu = (nu_int & 1);
    double r_end = rs[m - 1];

    int stride = K + 1;

    for (int t = 0; t < batch; ++t) {
        double *cheb_t = cheb_buffer_batch + (size_t)t * (size_t)stride;
        for (int l = 0; l <= K; ++l) {
            cheb_t[l] = 0.0;
        }
    }

    for (int l = 0; l <= K; ++l) {
        int l_is0 = (l == 0);
        double coeff0 = (nu_int == 0) ? (l_is0 ? 1.0 : 2.0) : 0.0;
        double coeff_even = (!(nu_int & 1)) ? (l_is0 ? 1.0 : 2.0) : 0.0;
        double coeff_odd = (nu_int & 1) ? 2.0 : 0.0;

        for (int k = 0; k < m; ++k) {
            double x = rs[k] / r_end;
            if (x > 1.0) x = 1.0;
            if (x < -1.0) x = -1.0;
            double theta = acos(x);

            double val = 0.0;
            if (nu_int == 0) {
                double cl = cos((double)l * theta);
                double tmp = 2.0 * cl * cl - 1.0;
                val = coeff0 * tmp;
            } else if ((nu_int & 1) == 0) {
                double c2l = cos(2.0 * (double)l * theta);
                val = coeff_even * c2l;
            } else {
                double c = cos(((double)(2 * l + 1)) * theta);
                val = coeff_odd * c;
            }

            for (int t = 0; t < batch; ++t) {
                const double *cs_t = cs + (size_t)t * (size_t)ld_cs;
                double *cheb_t = cheb_buffer_batch + (size_t)t * (size_t)stride;
                cheb_t[l] += val * cs_t[k];
            }
        }
    }

    for (int j = 0; j < n; ++j) {
        double w = ws[j];
        double z = w * r_end / 2.0;

        for (int l = 0; l <= K; ++l) {
            bessel_buffer_1[l] = 0.0;
        }

        if (nu_int == 0) {
            for (int l = 0; l <= K; ++l) {
                bessel_buffer_1[l] = gsl_sf_bessel_Jn(l, z);
            }
            for (int l = 0; l <= K; ++l) {
                bessel_buffer_1[l] *= bessel_buffer_1[l];
            }
            for (int l = 1; l <= K; l += 2) {
                bessel_buffer_1[l] *= -1.0;
            }
        } else {
            int max_order = l0 + K + is_odd_nu;
            int len_b2 = max_order + 1;

            for (int t = 0; t < len_b2; ++t) {
                bessel_buffer_2[t] = 0.0;
            }
            for (int k = 0; k <= max_order; ++k) {
                bessel_buffer_2[k] = gsl_sf_bessel_Jn(k, z);
            }
            int start_idx = l0 + is_odd_nu;
            for (int l = 0; l <= K; ++l) {
                int idx = start_idx + l;
                bessel_buffer_1[l] = (idx <= max_order) ? bessel_buffer_2[idx] : 0.0;
            }
            for (int t = 0; t <= l0; ++t) {
                int idx_b1 = l0 - t;
                if (idx_b1 >= 0 && idx_b1 <= K) {
                    bessel_buffer_1[idx_b1] *= bessel_buffer_2[t];
                }
            }
            int limit = K - l0;
            for (int q = 0; q < limit; ++q) {
                int idx_b1 = l0 + 1 + q;
                int idx_b2 = 1 + q;
                if (idx_b1 <= K && idx_b2 <= max_order) {
                    bessel_buffer_1[idx_b1] *= bessel_buffer_2[idx_b2];
                }
            }
            for (int l = l0 + 1; l <= K; l += 2) {
                bessel_buffer_1[l] *= -1.0;
            }
        }

        for (int t = 0; t < batch; ++t) {
            double *gs_t = gs + (size_t)t * (size_t)ld_gs;
            const double *cheb_t = cheb_buffer_batch + (size_t)t * (size_t)stride;
            double acc = 0.0;
            for (int l = 0; l <= K; ++l) {
                acc += bessel_buffer_1[l] * cheb_t[l];
            }
            gs_t[j] += acc;
        }
    }
}

int nufft1d3_wrapper(int nj, const double *xj, const double complex *cj,
                     int iflag, double eps,
                     int nk, const double *sk, double complex *fk)
{
    int ier;
    finufft_opts opts;
    finufft_default_opts(&opts);

    opts.nthreads = 1;

    ier = finufft1d3(
        (int64_t)nj,
        (double*)xj,
        (double _Complex*)cj,
        iflag,
        eps,
        (int64_t)nk,
        (double*)sk,
        (double _Complex*)fk,
        &opts
    );

    if (ier != 0) {
        fprintf(stderr, "FINUFFT finufft1d3 error code %d\n", ier);
    }

    return ier;
}

int nufft1d3many_wrapper(int ntr,
                         int nj, const double *xj, const double complex *cj,
                         int iflag, double eps,
                         int nk, const double *sk, double complex *fk)
{
    int ier;
    finufft_opts opts;
    finufft_default_opts(&opts);

    opts.nthreads = 1;

    ier = finufft1d3many(
        (int)ntr,
        (int64_t)nj,
        (double*)xj,
        (double _Complex*)cj,
        iflag,
        eps,
        (int64_t)nk,
        (double*)sk,
        (double _Complex*)fk,
        &opts
    );

    if (ier != 0) {
        fprintf(stderr, "FINUFFT finufft1d3many error code %d\n", ier);
    }

    return ier;
}

void add_asy(double *gs, double nu,
             const double *rs, const double *cs,
             const double *ws,
             int m, int n,
             int K,
             double tol,
             const double *asy_coef,
             double *real_buffer_1,
             double *real_buffer_2,
             double *in_buffer,
             double *out_buffer)
{
    double complex *c_in  = (double complex *)in_buffer;
    double complex *c_out = (double complex *)out_buffer;

    double sqrt2pi = get_sqrt_2_over_pi_clean();
    double phase_arg = M_PI * (-nu / 2.0 - 0.25);
    double complex phase = cos(phase_arg) + I * sin(phase_arg);

    for (int l = 0; l <= K; ++l) {
        int int_exp1 = -2 * l - 1;
        int int_exp2 = -2 * l - 2;

        for (int k = 0; k < m; ++k) {
            double rk = rs[k];
            double re = cs[k] * pow(rk, (double)int_exp1) * sqrt(rk);
            c_in[k] = re + 0.0 * I;
        }

        if (nufft1d3_wrapper(m, rs, c_in, +1, tol, n, ws, c_out) != 0) {
            fprintf(stderr, "add_asy: nufft1d3_wrapper failed (first half)\n");
            return;
        }

        for (int j = 0; j < n; ++j) {
            c_out[j] *= phase;
        }

        for (int j = 0; j < n; ++j) {
            double wj = ws[j];
            real_buffer_1[j] = creal(c_out[j]);
            real_buffer_2[j] = pow(wj, (double)int_exp1) * sqrt(wj);
        }

        double sign_l = (l & 1) ? -1.0 : 1.0;
        double coeff1 = sqrt2pi * sign_l * asy_coef[2 * l];

        for (int j = 0; j < n; ++j) {
            real_buffer_1[j] *= coeff1;
            real_buffer_1[j] *= real_buffer_2[j];
            gs[j] += real_buffer_1[j];
        }

        for (int k = 0; k < m; ++k) {
            double rk = rs[k];
            double re = cs[k] * pow(rk, (double)int_exp2) * sqrt(rk);
            c_in[k] = re + 0.0 * I;
        }

        if (nufft1d3_wrapper(m, rs, c_in, +1, tol, n, ws, c_out) != 0) {
            fprintf(stderr, "add_asy: nufft1d3_wrapper failed (second half)\n");
            return;
        }

        for (int j = 0; j < n; ++j) {
            c_out[j] *= phase;
        }

        for (int j = 0; j < n; ++j) {
            double wj = ws[j];
            real_buffer_1[j] = cimag(c_out[j]);
            real_buffer_2[j] = pow(wj, (double)int_exp2) * sqrt(wj);
        }

        double coeff2 = sqrt2pi * sign_l * asy_coef[2 * l + 1];

        for (int j = 0; j < n; ++j) {
            real_buffer_1[j] *= coeff2;
            real_buffer_1[j] *= real_buffer_2[j];
            gs[j] -= real_buffer_1[j];
        }
    }
}

void add_asy_batch(double *gs, double nu,
                   const double *rs, const double *cs,
                   const double *ws,
                   int m, int n,
                   int K,
                   double tol,
                   const double *asy_coef,
                   int batch,
                   int ld_cs, int ld_gs,
                   double *in_buffer,
                   double *out_buffer)
{
    double complex *c_in  = (double complex *)in_buffer;
    double complex *c_out = (double complex *)out_buffer;

    double sqrt2pi = get_sqrt_2_over_pi_clean();
    double phase_arg = M_PI * (-nu / 2.0 - 0.25);
    double complex phase = cos(phase_arg) + I * sin(phase_arg);

    for (int l = 0; l <= K; ++l) {
        int int_exp1 = -2 * l - 1;
        int int_exp2 = -2 * l - 2;

        for (int t = 0; t < batch; ++t) {
            const double *cs_t = cs + (size_t)t * (size_t)ld_cs;
            for (int k = 0; k < m; ++k) {
                double rk = rs[k];
                double re = cs_t[k] * pow(rk, (double)int_exp1) * sqrt(rk);
                c_in[(size_t)k + (size_t)m * (size_t)t] = re + 0.0 * I;
            }
        }

        if (nufft1d3many_wrapper(batch, m, rs, c_in, +1, tol, n, ws, c_out) != 0) {
            fprintf(stderr, "add_asy_batch: nufft1d3many_wrapper failed (first half)\n");
            return;
        }

        double sign_l = (l & 1) ? -1.0 : 1.0;
        double coeff1 = sqrt2pi * sign_l * asy_coef[2 * l];

        for (int t = 0; t < batch; ++t) {
            double *gs_t = gs + (size_t)t * (size_t)ld_gs;
            for (int j = 0; j < n; ++j) {
                double wj = ws[j];
                double scale = coeff1 * pow(wj, (double)int_exp1) * sqrt(wj);
                double complex val = c_out[(size_t)j + (size_t)n * (size_t)t] * phase;
                gs_t[j] += creal(val) * scale;
            }
        }

        for (int t = 0; t < batch; ++t) {
            const double *cs_t = cs + (size_t)t * (size_t)ld_cs;
            for (int k = 0; k < m; ++k) {
                double rk = rs[k];
                double re = cs_t[k] * pow(rk, (double)int_exp2) * sqrt(rk);
                c_in[(size_t)k + (size_t)m * (size_t)t] = re + 0.0 * I;
            }
        }

        if (nufft1d3many_wrapper(batch, m, rs, c_in, +1, tol, n, ws, c_out) != 0) {
            fprintf(stderr, "add_asy_batch: nufft1d3many_wrapper failed (second half)\n");
            return;
        }

        double coeff2 = sqrt2pi * sign_l * asy_coef[2 * l + 1];

        for (int t = 0; t < batch; ++t) {
            double *gs_t = gs + (size_t)t * (size_t)ld_gs;
            for (int j = 0; j < n; ++j) {
                double wj = ws[j];
                double scale = coeff2 * pow(wj, (double)int_exp2) * sqrt(wj);
                double complex val = c_out[(size_t)j + (size_t)n * (size_t)t] * phase;
                gs_t[j] -= cimag(val) * scale;
            }
        }
    }
}
#endif /* legacy corrupted content disabled */
