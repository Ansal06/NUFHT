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

/* ============================================================
 * add_dir!  (direct summation)
 * Julia:
 *   for j
 *     for k
 *       gs[j] += cs[k] * besselj(nu, wj*rs[k])
 * ============================================================ */

/** \brief Direct summation over Bessel evaluations (no buffers required). */
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

/* ============================================================
 * add_loc!  (local Wimp expansion)
 * Julia signature:
 *   add_loc!(gs, nu, rs, cs, ws; K, cheb_buffer, bessel_buffer_1, bessel_buffer_2)
 *
 * Assumptions (matching Julia use):
 *  - nu is a non-negative integer.
 *  - cheb_buffer has length >= K+1
 *  - bessel_buffer_1 has length >= K+1
 *  - bessel_buffer_2 has length >= l0+K+1+isodd(nu)
 * ============================================================ */

/** \brief Local Wimp expansion for integer nu with Chebyshev acceleration.
 *  \param gs Output array of length n (accumulated in-place).
 *  \param nu Integer Bessel order.
 *  \param rs Source radii grid (length m).
 *  \param cs Source coefficients (length m).
 *  \param ws Target frequencies grid (length n).
 *  \param K Local polynomial degree.
 *  \param cheb_buffer Scratch buffer of len >= K+1.
 *  \param bessel_buffer_1 Scratch buffer of len >= K+1.
 *  \param bessel_buffer_2 Scratch buffer of len >= l0+K+1+isodd(nu).
 */
void add_loc(double *gs, double nu,
             const double *rs, const double *cs,
             const double *ws,
             int m, int n,
             int K,
             double *cheb_buffer,
             double *bessel_buffer_1,
             double *bessel_buffer_2)
{
    /* Wimp expansion only works for integer nu */
    int nu_int = (int)llround(nu);
    if (fabs(nu - (double)nu_int) > 1e-12) {
        fprintf(stderr, "add_loc: nu must be integer\n");
        return;
    }

    /* center index of Wimp expansion: l0 = div(nu,2) */
    int l0 = nu_int / 2;
    int is_odd_nu = (nu_int & 1);

    /* rs[end] in Julia -> rs[m-1] here */
    double r_end = rs[m - 1];

    /* ---- Build Chebyshev-based coefficients (cheb_buffer) ---- */

    /* Initialize cheb_buffer to zeros */
    for (int l = 0; l <= K; ++l) {
        cheb_buffer[l] = 0.0;
    }

    for (int l = 0; l <= K; ++l) {
        int l_is0 = (l == 0);
        double coeff0 = (nu_int == 0) ? (l_is0 ? 1.0 : 2.0) : 0.0;
        double coeff_even = (! (nu_int & 1)) ? (l_is0 ? 1.0 : 2.0) : 0.0; /* for even nu>0 */
        double coeff_odd = (nu_int & 1) ? 2.0 : 0.0;

        for (int k = 0; k < m; ++k) {
            double x = rs[k] / r_end;
            if (x > 1.0) x = 1.0;
            if (x < -1.0) x = -1.0;
            double theta = acos(x);

            double val = 0.0;
            if (nu_int == 0) {
                /* (l==0?1:2) * (2cos(lθ)^2 - 1) */
                double cl = cos((double)l * theta);
                double tmp = 2.0 * cl * cl - 1.0;
                val = coeff0 * tmp;
            } else if ((nu_int & 1) == 0) {
                /* even nu: (l==0?1:2) * cos(2lθ) */
                double c2l = cos(2.0 * (double)l * theta);
                val = coeff_even * c2l;
            } else {
                /* odd nu: 2 * cos((2l+1)θ) */
                double c = cos(((double)(2*l + 1)) * theta);
                val = coeff_odd * c;
            }

            cheb_buffer[l] += val * cs[k];
        }
    }

    /* ---- For each ws[j], build Bessel combinations and dot ---- */

    for (int j = 0; j < n; ++j) {
        double w = ws[j];
        double z = w * r_end / 2.0;

        /* Zero bessel_buffer_1 each time */
        for (int l = 0; l <= K; ++l) {
            bessel_buffer_1[l] = 0.0;
        }

        if (nu_int == 0) {
            /* nu == 0 case */

            /* bessel_buffer_1[l] = J_l(z) */
            for (int l = 0; l <= K; ++l) {
                bessel_buffer_1[l] = gsl_sf_bessel_Jn(l, z);
            }

            /* square: J_l^2 */
            for (int l = 0; l <= K; ++l) {
                bessel_buffer_1[l] *= bessel_buffer_1[l];
            }

            /* use J_{-l} = (-1)^l J_l, so odd l get a minus */
            for (int l = 1; l <= K; l += 2) {
                bessel_buffer_1[l] *= -1.0;
            }

        } else {
            /* nu > 0 integer case */

            /* max order in bessel_buffer_2: 0..(l0+K+isodd(nu)) */
            int max_order = l0 + K + is_odd_nu;
            int len_b2 = max_order + 1;

            /* Zero bessel_buffer_2 */
            for (int t = 0; t < len_b2; ++t) {
                bessel_buffer_2[t] = 0.0;
            }

            /* fill bessel_buffer_2[k] = J_k(z), k = 0..max_order */
            for (int k = 0; k <= max_order; ++k) {
                bessel_buffer_2[k] = gsl_sf_bessel_Jn(k, z);
            }

            /* 1) copy orders l0+isodd .. l0+isodd+K into bessel_buffer_1[0..K] */
            int start_idx = l0 + is_odd_nu;
            for (int l = 0; l <= K; ++l) {
                int idx = start_idx + l;
                if (idx <= max_order) {
                    bessel_buffer_1[l] = bessel_buffer_2[idx];
                } else {
                    bessel_buffer_1[l] = 0.0;
                }
            }

            /* 2) view(b1, l0+1:-1:1) .*= view(b2,1:l0+1)
             *    => for t=0..l0: b1[l0 - t] *= b2[t]
             */
            for (int t = 0; t <= l0; ++t) {
                int idx_b1 = l0 - t;
                if (idx_b1 >= 0 && idx_b1 <= K) {
                    bessel_buffer_1[idx_b1] *= bessel_buffer_2[t];
                }
            }

            /* 3) view(b1, l0+2:(K+1)) .*= view(b2,2:K-l0+1)
             *    => for q=0..K-l0-1: b1[l0+1+q] *= b2[1+q]
             */
            int limit = K - l0;
            for (int q = 0; q < limit; ++q) {
                int idx_b1 = l0 + 1 + q;
                int idx_b2 = 1 + q;
                if (idx_b1 <= K && idx_b2 <= max_order) {
                    bessel_buffer_1[idx_b1] *= bessel_buffer_2[idx_b2];
                }
            }

            /* 4) use J_{-n} = (-1)^n J_n for some entries:
             *    view(b1, l0+2:2:(K+1)) .*= -1
             *    => l = l0+1, l0+3, ... <= K
             */
            for (int l = l0 + 1; l <= K; l += 2) {
                bessel_buffer_1[l] *= -1.0;
            }
        }

        /* dot(bessel_buffer_1, cheb_buffer) */
        double acc = 0.0;
        for (int l = 0; l <= K; ++l) {
            acc += bessel_buffer_1[l] * cheb_buffer[l];
        }

        gs[j] += acc;
    }
}

/* ============================================================
 * add_asy!  (asymptotic expansion + NUFFT)
 *
 * Julia logic:
 *   for l = 0:K
 *     in_buffer  = cs .* rs.^(-2l-1) .* sqrt.(rs)
 *     nufft1d3!(rs, in_buffer, +1, tol, ws, out_buffer)
 *     out_buffer *= cispi(-nu/2 - 1/4)
 *     real_buffer_1[j] = real(out_buffer[j])
 *     real_buffer_2[j] = ws[j]^(-2l-1)*sqrt(ws[j])
 *     real_buffer_1 *= sqrt(2/pi)*(-1)^l*ASY_COEF[2l+1]
 *     real_buffer_1 *= real_buffer_2
 *     gs += real_buffer_1
 *
 *     then with exponent -2l-2, using imag(out_buffer) and ASY_COEF[2l+2], subtracting.
 *
 * Here we treat in_buffer/out_buffer as arrays of double complex
 * stored in double[2*m], double[2*n] (as in nufht_inplace.c).
 * ============================================================ */

/* You must provide a NUFFT 1D type-3 implementation.
 * This wrapper signature is a simplified stand-in for FINUFFT's finufft1d3.
 *
 * nj : number of source points (m)
 * xj : source locations (rs)
 * cj : source strengths (in_buffer, complex)
 * iflag : +1 or -1
 * eps : tolerance
 * nk : number of target points (n)
 * sk : target locations (ws)
 * fk : output strengths (out_buffer, complex)
 */

int nufft1d3_wrapper(int nj, const double *xj, const double complex *cj,
                     int iflag, double eps,
                     int nk, const double *sk, double complex *fk)
{
    int ier;
    finufft_opts opts;
    finufft_default_opts(&opts);

    opts.nthreads = 1;  /* single-threaded for now */

    /* You can tune opts here if you like, but defaults usually work well */

    /* FINUFFT expects 'double *' and 'double _Complex *'. Cast away const. */
    ier = finufft1d3(
        (int64_t)nj,
        (double*)xj,              /* xj */
        (double _Complex*)cj,     /* cj */
        iflag,
        eps,
        (int64_t)nk,
        (double*)sk,              /* target points */
        (double _Complex*)fk,     /* output */
        &opts
    );

    if (ier != 0) {
        fprintf(stderr, "FINUFFT finufft1d3 error code %d\n", ier);
    }

    return ier;
}



/** \brief Asymptotic expansion accelerated by FINUFFT (large arguments).
 *  Uses two NUFFT calls per degree to accumulate real/imag parts with
 *  appropriate scaling and phase factors.
 */
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
    double complex *c_in  = (double complex *)in_buffer;   /* length >= m */
    double complex *c_out = (double complex *)out_buffer;  /* length >= n */

    double sqrt2pi = get_sqrt_2_over_pi();

    /* phase factor: cispi(-nu/2 - 1/4) = exp(i*pi*(-nu/2-1/4)) */
    double phase_arg = M_PI * (-nu / 2.0 - 0.25);
    double complex phase = cos(phase_arg) + I * sin(phase_arg);

    for (int l = 0; l <= K; ++l) {
        int int_exp1 = -2 * l - 1;   /* exponent for ws in first half */
        int int_exp2 = -2 * l - 2;   /* exponent for ws in second half */

        /* -------------------- First NUFFT: real part -------------------- */

        /* in_buffer = cs .* rs.^(-2l-1) .* sqrt(rs) */
        for (int k = 0; k < m; ++k) {
            double rk = rs[k];
            double re = cs[k] * pow(rk, (double)int_exp1) * sqrt(rk);
            c_in[k] = re + 0.0 * I;
        }

        if (nufft1d3_wrapper(m, rs, c_in,
                             +1, tol,
                             n, ws, c_out) != 0) {
            fprintf(stderr, "add_asy: nufft1d3_wrapper failed (first half)\n");
            return;
        }

        /* out_buffer *= cispi(-nu/2 - 1/4) */
        for (int j = 0; j < n; ++j) {
            c_out[j] *= phase;
        }

        /* real_buffer_1 = real(out_buffer), real_buffer_2 = ws.^int_exp1 * sqrt(ws) */
        for (int j = 0; j < n; ++j) {
            double wj = ws[j];
            real_buffer_1[j] = creal(c_out[j]);
            real_buffer_2[j] = pow(wj, (double)int_exp1) * sqrt(wj);
        }

        /* Multiply by coefficient and do diagonal scaling:
         * real_buffer_1 *= sqrt(2/pi) * (-1)^l * ASY_COEF[2l+1]
         * ASY_COEF index: 2l+1 (1-based) => [2l] in 0-based
         */
        double sign_l = (l & 1) ? -1.0 : 1.0;
        double coeff1 = sqrt2pi * sign_l * asy_coef[2 * l];

        for (int j = 0; j < n; ++j) {
            real_buffer_1[j] *= coeff1;
            real_buffer_1[j] *= real_buffer_2[j];
            gs[j] += real_buffer_1[j];
        }

        /* -------------------- Second NUFFT: imag part -------------------- */

        /* in_buffer = cs .* rs.^(-2l-2) .* sqrt(rs) */
        for (int k = 0; k < m; ++k) {
            double rk = rs[k];
            double re = cs[k] * pow(rk, (double)int_exp2) * sqrt(rk);
            c_in[k] = re + 0.0 * I;
        }

        if (nufft1d3_wrapper(m, rs, c_in,
                     +1, tol,
                             n, ws, c_out) != 0) {
            fprintf(stderr, "add_asy: nufft1d3_wrapper failed (second half)\n");
            return;
        }

        /* Apply phase again */
        for (int j = 0; j < n; ++j) {
            c_out[j] *= phase;
        }

        /* real_buffer_1 = imag(out_buffer), real_buffer_2 = ws.^int_exp2 * sqrt(ws) */
        for (int j = 0; j < n; ++j) {
            double wj = ws[j];
            real_buffer_1[j] = cimag(c_out[j]);
            real_buffer_2[j] = pow(wj, (double)int_exp2) * sqrt(wj);
        }

        /* real_buffer_1 *= sqrt(2/pi) * (-1)^l * ASY_COEF[2l+2]
         * ASY_COEF index: 2l+2 (1-based) => [2l+1] in 0-based
         */
        double coeff2 = sqrt2pi * sign_l * asy_coef[2 * l + 1];

        for (int j = 0; j < n; ++j) {
            real_buffer_1[j] *= coeff2;
            real_buffer_1[j] *= real_buffer_2[j];
            gs[j] -= real_buffer_1[j];
        }
    }
}
