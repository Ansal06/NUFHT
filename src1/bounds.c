/**
 * \file bounds.c
 * \brief Table generation for asymptotic coefficients and regime bounds.
 *
 * Populates ASY_COEF_TABLE, ASY_Z_TABLE, and WIMP_K_TABLE used to plan the
 * NUFHT box partition and expansion degrees for given tolerances and orders.
 */
#include "fast_hankel_nufht.h"

#include <math.h>
#include <stdio.h>
#include <time.h>

/* ------------------------------------------------------------------
   These must match the definitions in fast_hankel_nufht.c
   ------------------------------------------------------------------ */

#ifndef MAX_NU_INDEX
#define MAX_NU_INDEX 401   /* for nus_all = 0:0.5:200 => 401 values */
#endif

#ifndef MAX_ASY_COEF
#define MAX_ASY_COEF 64    /* >= 2*max_asy_K+2 (22) */
#endif

#ifndef MAX_I
#define MAX_I  401         /* index in nu for zs/Ks; we use 0..200 */
#endif

#ifndef MAX_J
#define MAX_J  32          /* number of tol “bins” (we use 12) */
#endif

#ifndef MAX_K
#define MAX_K  16          /* >= max_asy_K (10) */
#endif

/* These arrays are actually defined in fast_hankel_nufht.c */
extern double ASY_COEF_TABLE[MAX_NU_INDEX][MAX_ASY_COEF];
extern double ASY_Z_TABLE[MAX_I][MAX_J][MAX_K];
extern int    WIMP_K_TABLE[MAX_I][MAX_J][MAX_K];

/* ------------------------------------------------------------------
   Parameters that mirror bounds.jl
   ------------------------------------------------------------------ */

static const int   NU_MAX    = 200;
static const int   MAX_ASY_K = 10;  /* max_asy_K */
static const int   NTOLS     = 12;  /* tols = 10.^(-4:-1:-15) */

/* tols[j] = 10^(-4-j) for j=0..11 => 1e-4 .. 1e-15 */
static const double TOLS[12] = {       // THIS SHOULD MATCH NTOLS
    1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9,
    1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15
};

/* ------------------------------------------------------------------
   a(k, nu): coefficients in Hankel's expansion
   Julia: a(k,nu) = k==0 ? 1 : prod(4nu^2 - (odd)^2)/(k! 8^k)
   ------------------------------------------------------------------ */

/** \brief Hankel asymptotic coefficient a(k,nu) used in expansion. */
static double a_coef(int k, double nu)
{
    if (k == 0) {
        return 1.0;
    }
    /* For the ranges we care about (k <= 21, nu <= 200),
       double precision is fine. */
    double prod = 1.0;
    for (int j = 1; j <= 2*k - 1; j += 2) {
        double term = 4.0*nu*nu - (double)(j*j);
        prod *= term;
    }
    /* denominator: k! * 8^k */
    double fact = 1.0;
    for (int j = 2; j <= k; ++j) {
        fact *= (double)j;
    }
    double denom = fact * pow(8.0, (double)k);
    return prod / denom;
}

/* ------------------------------------------------------------------
   generate_asy_a_table(nus_all, max_asy_K)
   Julia:
     nus_all = 0:0.5:nu_max
     Js = 0:(2max_asy_K+1)
     as[i,j] = a(J,nu)
   We fill ASY_COEF_TABLE[row][J] with row = 2*nu (0-based).
   ------------------------------------------------------------------ */

static void generate_asy_a_table(void)
{
    int Jmax = 2*MAX_ASY_K + 1;  /* J goes 0..Jmax */

    for (int i = 0; i <= 2*NU_MAX; ++i) {
        double nu = 0.5 * (double)i;  /* 0,0.5,1,...,200 */
        int row = i;                  /* 0-based row index */

        for (int J = 0; J <= Jmax; ++J) {
            printf("computing asymptotic coefficient for nu = %.1f, j = %d...\n",
                   nu, J);
            ASY_COEF_TABLE[row][J] = a_coef(J, nu);
        }

        /* zero out remaining columns if MAX_ASY_COEF > Jmax+1 */
        for (int J = Jmax+1; J < MAX_ASY_COEF; ++J) {
            ASY_COEF_TABLE[row][J] = 0.0;
        }
    }
}

/* ------------------------------------------------------------------
   asy_error_bound(nu, K, z)
   Julia:
     sqrt(2/(pi*z)) * ( |a(2K,nu)|/z^(2K) + |a(2K+1,nu)|/z^(2K+1) )
   ------------------------------------------------------------------ */

static double asy_error_bound(double nu, int K, double z)
{
    double k1 = (double)(2*K);
    double k2 = (double)(2*K + 1);
    double a1 = fabs(a_coef(2*K, nu));
    double a2 = fabs(a_coef(2*K + 1, nu));
    double z1 = pow(z, k1);
    double z2 = pow(z, k2);

    double pref = sqrt(2.0 / (M_PI * z));
    return pref * (a1 / z1 + a2 / z2);
}

/* ------------------------------------------------------------------
   Numeric Newton solver for asy_error_bound(nu,K,z) - tol = 0
   Julia newton(f,x0,tol; maxiter=1000, verbose=false)
   We'll use:
     x0 = 1.0, rel_tol = 1e-8, maxiter = 1000
   No bounds.
   ------------------------------------------------------------------ */

static double newton_asy(double nu, int K, double tol,
                         double x0, double rel_tol, int maxiter)
{
    double x = x0;
    double dx = 0.0;

    for (int iter = 0; iter < maxiter; ++iter) {
        double f = asy_error_bound(nu, K, x) - tol;

        /* central difference derivative */
        double h = 1e-5 * (fabs(x) > 1.0 ? fabs(x) : 1.0);
        double xph = x + h;
        double xmh = x - h;
        if (xph <= 0.0) xph = x + h;
        if (xmh <= 0.0) xmh = x - h;

        double fp = asy_error_bound(nu, K, xph) - tol;
        double fm = asy_error_bound(nu, K, xmh) - tol;
        double df = (fp - fm) / (2.0*h);

        if (fabs(df) < 1e-16) {
            /* derivative too small; bail out */
            break;
        }

        dx = -f / df;
        x += dx;
        if (x <= 0.0) {
            x = fabs(x);
        }

        if (fabs(dx / x) <= rel_tol) {
            break;
        }
    }

    return x;
}

/* ------------------------------------------------------------------
   generate_asy_z_table(nus_int, asy_Ks, tols)
   Julia:
     nus = 0:nu_max (integer)
     asy_Ks = 1:max_asy_K
     tols = 10.^(-4:-1:-15) (12 values)
     zs[i,j,k] = newton( z -> asy_error_bound(nu,K,z)-tol, 1.0, 1e-8 )
   We store:
     ASY_Z_TABLE[i_idx][j_idx][k_idx] with
       i_idx = nu (0-based, nu integer)
       j_idx = 0..NTOLS-1
       k_idx = asy_K-1 (0..MAX_ASY_K-1)
   ------------------------------------------------------------------ */

static void generate_asy_z_table(void)
{
    for (int nu = 0; nu <= NU_MAX; ++nu) {
        int i_idx = nu;  /* 0..200 */

        for (int j = 0; j < NTOLS; ++j) {
            double tol = TOLS[j];

            for (int asy_K = 1; asy_K <= MAX_ASY_K; ++asy_K) {
                int k_idx = asy_K - 1;

                printf("computing asymptotic z for nu = %d, tol = %.0e, K = %d...\n",
                       nu, tol, asy_K);

                double z = newton_asy((double)nu, asy_K, tol,
                                      1.0, 1e-8, 1000);
                ASY_Z_TABLE[i_idx][j][k_idx] = z;
            }
        }
    }
}

/* ------------------------------------------------------------------
   psi(p) and wimp_error_bound(nu,K,z)
   Julia:
     psi(p) = log(p) + sqrt(1-p^2) - log(1 + sqrt(1-p^2))
     wimp_error_bound(nu,K,z) as in your code.
   ------------------------------------------------------------------ */

static double psi_func(double p)
{
    /* assume 0 < p <= 1 */
    double r = sqrt(fmax(0.0, 1.0 - p*p));
    return log(p) + r - log(1.0 + r);
}

static double wimp_error_bound(double nu, int K, double z)
{
    /* assert: z <= 2K + nu */
    double denom1 = 2.0 * K + 2.0 + nu;
    double bK = psi_func(z / denom1);

    double twoK_minus_nu = 2.0 * K - nu;
    if (twoK_minus_nu > 0.0 && z / twoK_minus_nu <= 1.0) {
        double denom2 = 2.0 * K + 2.0 - nu;
        double cK = psi_func(z / denom2);

        double num = 2.0 * exp(bK*(nu/2.0 + K + 1.0)
                              + cK*(-nu/2.0 + K + 1.0));
        double den = 1.0 - exp(bK + cK);
        return num / den;
    } else {
        double num = 2.0 * exp(bK*(nu/2.0 + K + 1.0));
        double den = 1.0 - exp(bK);
        return num / den;
    }
}

/* ------------------------------------------------------------------
   generate_wimp_K_table(nus_int, asy_Ks, zs, tols)
   Julia:
     wimp_K = ceil((z - nu)/2)
     while !conv
       conv = wimp_error_bound(nu, wimp_K, z) < tol
       wimp_K += 1
     end
     wimp_Ks[i,j,k] = wimp_K - 1
   We store:
     WIMP_K_TABLE[i_idx][j_idx][k_idx]
   ------------------------------------------------------------------ */

static void generate_wimp_K_table(void)
{
    for (int nu = 0; nu <= NU_MAX; ++nu) {
        int i_idx = nu;

        for (int j = 0; j < NTOLS; ++j) {
            double tol = TOLS[j];

            for (int asy_K = 1; asy_K <= MAX_ASY_K; ++asy_K) {
                int k_idx = asy_K - 1;
                double z = ASY_Z_TABLE[i_idx][j][k_idx];

                printf("computing Wimp K for nu = %d, tol = %.0e, asymptotic K = %d...\n",
                       nu, tol, asy_K);

                int wimp_K = (int)ceil( (z - (double)nu)/2.0 );
                if (wimp_K < 1) wimp_K = 1;

                int conv = 0;
                while (!conv) {
                    double err = wimp_error_bound((double)nu, wimp_K, z);
                    if (err < tol) {
                        conv = 1;
                    } else {
                        wimp_K++;
                    }
                }

                WIMP_K_TABLE[i_idx][j][k_idx] = wimp_K;
            }
        }
    }
}

/* ------------------------------------------------------------------
   generate_tables()
   Top-level driver, analogous to Julia generate_tables().
   Fills the global tables in memory.
   ------------------------------------------------------------------ */

/** \brief Build global tables: asymptotic coefficients, z-bounds, and Wimp K. */
void generate_tables(void)
{
    printf("\n--------------------\n");
    printf("Generating tables\n");
    printf("--------------------\n");

    clock_t t0 = clock();

    generate_asy_a_table();
    generate_asy_z_table();
    generate_wimp_K_table();

    clock_t t1 = clock();
    double elapsed = (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("\n--------------------\n");
    printf("Tables successfully generated! (%.1f s)\n", elapsed);
    printf("--------------------\n");
}
