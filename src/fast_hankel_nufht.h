/**
 * \file fast_hankel_nufht.h
 * \brief Public API for Nonuniform Fast Hankel Transform (NUFHT).
 *
 * Declares options, planning structs, box generation, and execution APIs for
 * evaluating Hankel integrals on arbitrary grids using local/asymptotic/direct
 * regimes. See fast_hankel_nufht.c for implementation details.
 */
#ifndef FAST_HANKEL_NUFHT_H
#define FAST_HANKEL_NUFHT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------ Options ------------------ */
/** \brief Runtime options controlling partitioning and expansion degrees. */

typedef struct {
    double tol;         /* default: 1e-8 */
    int    max_levels;  /* <=0 => auto */
    int    min_dim_prod;/* default: 10000 */
    double z_split;     /* NaN => use table */
    int    K_asy;       /* <=0 => auto/table */
    int    K_loc;       /* <=0 => table */
} NufhtOptions;

/** \brief Return defaults for \ref NufhtOptions. */
NufhtOptions nufht_default_options(void);

/* ------------------ Box types ------------------ */

/** \brief Rectangular subregion in (w,r) grid by index bounds. */
typedef struct {
    int i0, i1;   /* row indices (w-index), inclusive, 0-based */
    int j0, j1;   /* col indices (r-index), inclusive, 0-based */
} Box;

/** \brief Dynamic array of \ref Box entries. */
typedef struct {
    Box   *data;
    size_t length;
    size_t capacity;
} BoxVector;

/** \brief Partition of (w,r) into local/asymptotic/direct work regions. */
typedef struct {
    BoxVector loc;  /* local-expansion boxes */
    BoxVector asy;  /* asymptotic-expansion boxes */
    BoxVector dir;  /* direct boxes */
} Boxes;

/* Free all allocations inside Boxes */
/** \brief Free internal allocations inside a \ref Boxes container. */
void boxes_free(Boxes *boxes);

/* ------------------ Planning structs (thread-safe) ------------------ */

/** \brief Immutable plan with precomputed parameters and box partition. */
typedef struct {
    double nu;
    double tol;
    double z_split;
    int    K_asy;
    int    K_loc;
    double asy_coef[64]; /* copy of ASY_COEF_TABLE row for this nu */

    int m;               /* number of source points (rs, cs) */
    int n;               /* number of target points (ws) */
    const double *rs;    /* not owned; user-provided grid */
    const double *ws;    /* not owned; user-provided grid */

    int max_levels;      /* store options used to build boxes */
    int min_dim_prod;

    Boxes boxes;         /* precomputed loc/asy/dir boxes for rs/ws */

    int tables_loaded;   /* 1 if tables were loaded/generated */
} NufhtPlan;

/** \brief Thread-local buffers used during NUFHT evaluation. */
typedef struct {
    double *in_buffer;       /* len >= 2*m (complex) */
    double *out_buffer;      /* len >= 2*n (complex) */
    double *real_buffer_1;   /* len >= n */
    double *real_buffer_2;   /* len >= n */
    double *cheb_buffer;     /* len >= K_loc+1 */
    double *bessel_buffer_1; /* len >= K_loc+1 */
    double *bessel_buffer_2; /* len >= K_loc+1+extra */

    int m_alloc;
    int n_alloc;
    int Kloc_alloc;
    int extra_alloc;
} NufhtScratch;

/** \brief Initialize a \ref NufhtPlan for given grids and options.
 *  \return 0 on success, nonzero on invalid parameters.
 */
int nufht_plan_init(NufhtPlan *plan,
                    double nu,
                    const double *rs, int m,
                    const double *ws, int n,
                    const NufhtOptions *opt);

/** \brief Free resources owned by a \ref NufhtPlan. */
void nufht_plan_free(NufhtPlan *plan);

/** \brief Allocate thread-local scratch buffers sized to a plan. */
int nufht_scratch_init(NufhtScratch *scratch, const NufhtPlan *plan);
/** \brief Free buffers owned by \ref NufhtScratch. */
void nufht_scratch_free(NufhtScratch *scratch);

/* ------------------ Box generation ------------------ */

/* Generate loc/asy/dir boxes for:
 *   rs[0..m-1], ws[0..n-1]
 * with given z_split, max_levels (<=0 => auto), and min_dim_prod.
 */
/** \brief Partition rs×ws into local/asymptotic/direct boxes. */
Boxes generate_boxes(const double *rs, int m,
                     const double *ws, int n,
                     double z_split,
                     int max_levels,
                     int min_dim_prod,
                     int K_loc_threshold);

/* Split a direct box into two subregions (as in Julia split_box). */
/** \brief Split a direct box into two subregions using boundary z. */
void split_box(const double *rs, int m,
               const double *ws, int n,
               const Box    *box,
               double        z,
               int          *ispl_out,
               int          *jspl_out);

/* ------------------ Main NUFHT API ------------------ */

/* Batched transform:
 * - cs layout: column-major by transform, stride ld_cs (>= m)
 * - gs layout: column-major by transform, stride ld_gs (>= n)
 * - batch: number of transforms
 * If scratch is NULL, temporary scratch is allocated per call.
 */
/** \brief Batched NUFHT for multiple RHS (column-major stride). */
int nufht_batch(const NufhtPlan *plan,
                const double *cs, int ld_cs,
                double *gs, int ld_gs,
                int batch,
                NufhtScratch *scratch);

/* Single-transform convenience wrappers (legacy compatible). */
/** \brief Single-transform convenience wrapper writing directly to gs. */
void nufht_inplace(double       *gs,
                   double        nu,
                   const double *rs, int m,
                   const double *cs,
                   const double *ws, int n,
                   const NufhtOptions *opt);

/** \brief Allocate output and execute a single NUFHT transform. */
double *nufht(double        nu,
              const double *rs, int m,
              const double *cs,
              const double *ws, int n,
              const NufhtOptions *opt);

#ifdef __cplusplus
}
#endif

/** \brief Build global tolerance/order-dependent tables used by planning. */
void generate_tables(void);


/** \brief Direct summation over Bessel evaluations (small problems). */
void add_dir(double *gs, double nu,
             const double *rs, const double *cs,
             const double *ws,
             int m, int n);

/** \brief Asymptotic expansion accelerated via NUFFT (large arguments). */
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
             double *out_buffer);

/** \brief Local (Wimp) expansion with Chebyshev acceleration (integer nu). */
void add_loc(double *gs, double nu,
             const double *rs, const double *cs,
             const double *ws,
             int m, int n,
             int K,
             double *cheb_buffer,
             double *bessel_buffer_1,
             double *bessel_buffer_2);



#endif /* FAST_HANKEL_NUFHT_H */



