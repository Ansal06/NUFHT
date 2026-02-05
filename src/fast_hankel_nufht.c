/**
 * \file fast_hankel_nufht.c
 * \brief Core implementation of the Nonuniform Fast Hankel Transform (NUFHT).
 *
 * Provides planning, box partitioning, expansion selection, and batched
 * execution routines. Also includes table generation and persistence logic
 * for startup efficiency.
 */
#include "fast_hankel_nufht.h"

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <limits.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

/* ================== Global NUFHT state ================== */

/* Legacy globals retained for backward compatibility only. */
double NUFHT_NU      = NAN;
double NUFHT_TOL     = NAN;
double NUFHT_Z_SPLIT = NAN;
int    NUFHT_ASY_K   = 0;
int    NUFHT_LOC_K   = -1;

/* ============ Tables (placeholders, adjust sizes) ============ */

#define MAX_NU_INDEX  401   /* for |nu| <= 200 => 2|nu|+1 <= 401 */
#define MAX_ASY_COEF  64    /* adjust to your real table size */

double ASY_COEF_TABLE[MAX_NU_INDEX][MAX_ASY_COEF]; /* "as" in Julia */

#define MAX_I  401   /* i index (|nu|) */
#define MAX_J   32   /* j index (tolerance) */
#define MAX_K   16   /* k index (K_asy up to ~10) */

double ASY_Z_TABLE[MAX_I][MAX_J][MAX_K];  /* "zs" in Julia */
int    WIMP_K_TABLE[MAX_I][MAX_J][MAX_K]; /* "Ks" in Julia */

static int tables_initialized = 0;

/* Forward declarations for table persistence */
static int save_tables(const char *filename);
static int load_tables(const char *filename);

/* Helper: build default cache file path in same dir as the executable. */
static int get_default_cache_path(char *out, size_t outlen)
{
#ifdef __APPLE__
    char exec_path[PATH_MAX];
    uint32_t sz = (uint32_t)sizeof(exec_path);
    if (_NSGetExecutablePath(exec_path, &sz) != 0) {
        return -1;
    }
    char real_exec[PATH_MAX];
    char *resolved = realpath(exec_path, real_exec);
    const char *use = resolved ? real_exec : exec_path;
    const char *slash = strrchr(use, '/');
    size_t dirlen = slash ? (size_t)(slash - use) : 0;
    const char *fname = "data/nufht_tables.cache";
    if (dirlen == 0) return -1;
    if (dirlen + 1 + strlen(fname) + 1 > outlen) return -1;
    memcpy(out, use, dirlen);
    out[dirlen] = '/';
    strcpy(out + dirlen + 1, fname);
    return 0;
#elif defined(__linux__)
    char exe_link[PATH_MAX];
    ssize_t r = readlink("/proc/self/exe", exe_link, sizeof(exe_link) - 1);
    if (r <= 0) return -1;
    exe_link[r] = '\0';
    const char *slash = strrchr(exe_link, '/');
    size_t dirlen = slash ? (size_t)(slash - exe_link) : 0;
    const char *fname = "data/nufht_tables.cache";
    if (dirlen == 0) return -1;
    if (dirlen + 1 + strlen(fname) + 1 > outlen) return -1;
    memcpy(out, exe_link, dirlen);
    out[dirlen] = '/';
    strcpy(out + dirlen + 1, fname);
    return 0;
#else
    (void)out; (void)outlen;
    return -1;
#endif
}

/** \brief Ensure global tables are available, loading or generating as needed. */
static void ensure_tables(void) {
    if (tables_initialized) return;
    /* Simple one-time init; for threaded callers protect generation. */
    #pragma omp critical (nufht_tables_init)
    {
        if (!tables_initialized) {
            /* Try to load from cache first */
            const char *cache_env = getenv("NUFHT_TABLE_CACHE");
            char default_path[PATH_MAX];
            const char *cache_file = cache_env;
            if (!cache_file) {
                /* Prefer a writable project data/ directory (cwd or parent) */
                char cwd_path[PATH_MAX];
                struct stat st;
                if (getcwd(cwd_path, sizeof(cwd_path)) != NULL) {
                    const char *fname = "data/nufht_tables.cache";

                    /* Check cwd/data directory exists */
                    char data_dir[PATH_MAX];
                    size_t cwdlen = strlen(cwd_path);
                    if (cwdlen + 1 + strlen("data") + 1 < sizeof(data_dir)) {
                        memcpy(data_dir, cwd_path, cwdlen);
                        data_dir[cwdlen] = '/';
                        strcpy(data_dir + cwdlen + 1, "data");
                        if (stat(data_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
                            if (cwdlen + 1 + strlen(fname) + 1 < sizeof(default_path)) {
                                memcpy(default_path, cwd_path, cwdlen);
                                default_path[cwdlen] = '/';
                                strcpy(default_path + cwdlen + 1, fname);
                                cache_file = default_path;
                            }
                        }
                    }

                    /* If cwd/data doesn't exist, try parent/data (project root) */
                    if (!cache_file) {
                        const char *slash = strrchr(cwd_path, '/');
                        if (slash) {
                            size_t parent_len = (size_t)(slash - cwd_path);
                            if (parent_len + 1 + strlen("data") + 1 < sizeof(data_dir)) {
                                memcpy(data_dir, cwd_path, parent_len);
                                data_dir[parent_len] = '/';
                                strcpy(data_dir + parent_len + 1, "data");
                                if (stat(data_dir, &st) == 0 && S_ISDIR(st.st_mode)) {
                                    if (parent_len + 1 + strlen(fname) + 1 < sizeof(default_path)) {
                                        memcpy(default_path, cwd_path, parent_len);
                                        default_path[parent_len] = '/';
                                        strcpy(default_path + parent_len + 1, fname);
                                        cache_file = default_path;
                                    }
                                }
                            }
                        }
                    }
                }

                /* Fallback to executable directory if cwd-based paths unavailable */
                if (!cache_file) {
                    if (get_default_cache_path(default_path, sizeof(default_path)) == 0) {
                        cache_file = default_path;
                    } else {
                        cache_file = "data/nufht_tables.cache";
                    }
                }
            }
            int load_ok = (cache_file != NULL) && (load_tables(cache_file) == 0);
            
            if (!load_ok) {
                /* Generate tables if load failed or no cache specified */
                generate_tables();
                
                /* Try to save to cache if a path is available */
                if (cache_file != NULL && save_tables(cache_file) == 0) {
                    printf("Saved tables to %s\n", cache_file);
                }
                else {
                    fprintf(stderr, "Warning: could not save tables to cache file (%s). They will be regenerated next run.\n", cache_file ? cache_file : "(null)");
                }
            } else {
                printf("Loaded tables from %s\n", cache_file);
            }
            tables_initialized = 1;
        }
    }
}

/* ================== Table persistence (save/load) ================== */

#define NUFHT_TABLE_MAGIC   0x4E5546u    /* "NUF" in hex, 24-bit magic */
#define NUFHT_TABLE_VERSION 1u

/* Save tables to a binary cache file.
 * Format: magic (4) | version (4) | MAX_NU_INDEX (4) | MAX_I (4) | MAX_J (4) | MAX_K (4) |
 *         ASY_COEF_TABLE | ASY_Z_TABLE | WIMP_K_TABLE
 */
/** \brief Persist global tables to a binary cache file.
 *  \return 0 on success, nonzero on I/O error.
 */
static int save_tables(const char *filename)
{
    if (!filename) return -1;
    
    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "nufht_save_tables: cannot open %s for writing\n", filename);
        return -2;
    }

    uint32_t magic = NUFHT_TABLE_MAGIC;
    uint32_t version = NUFHT_TABLE_VERSION;
    uint32_t nu_index = (uint32_t)MAX_NU_INDEX;
    uint32_t i_dim = (uint32_t)MAX_I;
    uint32_t j_dim = (uint32_t)MAX_J;
    uint32_t k_dim = (uint32_t)MAX_K;

    if (fwrite(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&version, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&nu_index, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&i_dim, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&j_dim, sizeof(uint32_t), 1, f) != 1 ||
        fwrite(&k_dim, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "nufht_save_tables: header write failed\n");
        fclose(f);
        return -3;
    }

    /* Write ASY_COEF_TABLE */
    if (fwrite(ASY_COEF_TABLE, sizeof(double), MAX_NU_INDEX * MAX_ASY_COEF, f)
        != (size_t)(MAX_NU_INDEX * MAX_ASY_COEF)) {
        fprintf(stderr, "nufht_save_tables: ASY_COEF_TABLE write failed\n");
        fclose(f);
        return -4;
    }

    /* Write ASY_Z_TABLE */
    if (fwrite(ASY_Z_TABLE, sizeof(double), MAX_I * MAX_J * MAX_K, f)
        != (size_t)(MAX_I * MAX_J * MAX_K)) {
        fprintf(stderr, "nufht_save_tables: ASY_Z_TABLE write failed\n");
        fclose(f);
        return -5;
    }

    /* Write WIMP_K_TABLE */
    if (fwrite(WIMP_K_TABLE, sizeof(int), MAX_I * MAX_J * MAX_K, f)
        != (size_t)(MAX_I * MAX_J * MAX_K)) {
        fprintf(stderr, "nufht_save_tables: WIMP_K_TABLE write failed\n");
        fclose(f);
        return -6;
    }

    fclose(f);
    return 0;
}

/* Load tables from a binary cache file.
 * Returns 0 on success, nonzero on failure (file not found, version mismatch, etc).
 */
/** \brief Load global tables from a binary cache file.
 *  \return 0 on success, nonzero on mismatch or I/O error.
 */
static int load_tables(const char *filename)
{
    if (!filename) return -1;

    FILE *f = fopen(filename, "rb");
    if (!f) {
        return -2;  /* File not found or unreadable (expected in first run) */
    }

    uint32_t magic, version, nu_index, i_dim, j_dim, k_dim;

    if (fread(&magic, sizeof(uint32_t), 1, f) != 1 ||
        fread(&version, sizeof(uint32_t), 1, f) != 1 ||
        fread(&nu_index, sizeof(uint32_t), 1, f) != 1 ||
        fread(&i_dim, sizeof(uint32_t), 1, f) != 1 ||
        fread(&j_dim, sizeof(uint32_t), 1, f) != 1 ||
        fread(&k_dim, sizeof(uint32_t), 1, f) != 1) {
        fprintf(stderr, "nufht_load_tables: header read failed\n");
        fclose(f);
        return -3;
    }

    /* Validate header */
    if (magic != NUFHT_TABLE_MAGIC || version != NUFHT_TABLE_VERSION ||
        (int)nu_index != MAX_NU_INDEX || (int)i_dim != MAX_I ||
        (int)j_dim != MAX_J || (int)k_dim != MAX_K) {
        fprintf(stderr, "nufht_load_tables: version/dimension mismatch\n");
        fclose(f);
        return -4;
    }

    /* Read ASY_COEF_TABLE */
    if (fread(ASY_COEF_TABLE, sizeof(double), MAX_NU_INDEX * MAX_ASY_COEF, f)
        != (size_t)(MAX_NU_INDEX * MAX_ASY_COEF)) {
        fprintf(stderr, "nufht_load_tables: ASY_COEF_TABLE read failed\n");
        fclose(f);
        return -5;
    }

    /* Read ASY_Z_TABLE */
    if (fread(ASY_Z_TABLE, sizeof(double), MAX_I * MAX_J * MAX_K, f)
        != (size_t)(MAX_I * MAX_J * MAX_K)) {
        fprintf(stderr, "nufht_load_tables: ASY_Z_TABLE read failed\n");
        fclose(f);
        return -6;
    }

    /* Read WIMP_K_TABLE */
    if (fread(WIMP_K_TABLE, sizeof(int), MAX_I * MAX_J * MAX_K, f)
        != (size_t)(MAX_I * MAX_J * MAX_K)) {
        fprintf(stderr, "nufht_load_tables: WIMP_K_TABLE read failed\n");
        fclose(f);
        return -7;
    }

    fclose(f);
    return 0;
}


/* ================== Utilities ================== */

static int is_integer_double(double x) {
    double r = nearbyint(x);
    return fabs(x - r) < 1e-12;
}

static int fill_plan_parameters(NufhtPlan *plan,
                                double nu,
                                const NufhtOptions *opt)
{
    if (!plan || !opt) return -100;

    if (opt->tol < 1e-15) {
        fprintf(stderr, "setup_nufht: cannot set NUFHT tolerance below 1e-15\n");
        return -1;
    }

    int is_2nu_int = is_integer_double(2.0 * nu);
    int is_nu_int  = is_integer_double(nu);

    if (!is_2nu_int ||
        (is_nu_int && fabs(nu) > 200.0) ||
        (!is_nu_int && is_2nu_int && fabs(nu) > 19.0 / 2.0))
    {
        fprintf(stderr,
           "setup_nufht: only integer orders nu = 0,±1,...,±200 and "
           "half-integer nu = 1/2,3/2,...,19/2 are implemented\n");
        return -2;
    }

    ensure_tables();

    int i_idx = (int)llround(fabs(nu));
    if (i_idx < 0 || i_idx >= MAX_I) {
        fprintf(stderr, "setup_nufht: i index out of range\n");
        return -3;
    }

    int row_idx = (int)llround(2.0 * fabs(nu) + 1.0);  /* Julia index */
    int row = row_idx - 1;                             /* C index */
    if (row < 0 || row >= MAX_NU_INDEX) {
        fprintf(stderr, "setup_nufht: ASY_COEF_TABLE row out of range\n");
        return -4;
    }

    for (int k = 0; k < MAX_ASY_COEF; ++k) {
        plan->asy_coef[k] = ASY_COEF_TABLE[row][k];
    }

    double j_real = -log10(opt->tol) - 3.0;
    int j = (int)ceil(j_real);
    if (j < 1) j = 1;
    int j_idx = j - 1;
    if (j_idx < 0 || j_idx >= MAX_J) {
        fprintf(stderr, "setup_nufht: j index out of range\n");
        return -5;
    }

    plan->nu  = nu;
    plan->tol = opt->tol;

    if (is_nu_int) {
        int k_val;
        if (opt->K_asy <= 0) {
            double expr = fabs(nu)/5.0 + log10(1.0/opt->tol)/4.0 + 1.0;
            int k0 = (int)floor(expr);
            k_val = (k0 < 10) ? k0 : 10;
        } else {
            k_val = opt->K_asy;
        }
        if (k_val < 0 || k_val >= MAX_K) {
            fprintf(stderr, "setup_nufht: asymptotic K index out of range\n");
            return -6;
        }
        plan->K_asy = k_val;

        if (isnan(opt->z_split)) {
            plan->z_split = ASY_Z_TABLE[i_idx][j_idx][k_val];
        } else {
            plan->z_split = opt->z_split;
        }

        if (opt->K_loc <= 0) {
            plan->K_loc = WIMP_K_TABLE[i_idx][j_idx][k_val];
        } else {
            plan->K_loc = opt->K_loc;
        }

    } else {
        double tmp = fabs(nu) - 0.5;
        int k_val = (int)llround(tmp);
        plan->K_asy   = k_val;
        plan->z_split = NAN;
        plan->K_loc   = -1;
    }

    plan->tables_loaded = 1;
    return 0;
}

/* ------------------ BoxVector helpers ------------------ */

static void boxvec_init(BoxVector *v) {
    v->data = NULL;
    v->length = 0;
    v->capacity = 0;
}

static void boxvec_push(BoxVector *v, int i0, int i1, int j0, int j1) {
    if (v->length == v->capacity) {
        size_t new_cap = (v->capacity == 0) ? 8 : 2 * v->capacity;
        Box *tmp = (Box *)realloc(v->data, new_cap * sizeof(Box));
        if (!tmp) {
            fprintf(stderr, "boxvec_push: allocation failed\n");
            return;
        }
        v->data = tmp;
        v->capacity = new_cap;
    }
    v->data[v->length].i0 = i0;
    v->data[v->length].i1 = i1;
    v->data[v->length].j0 = j0;
    v->data[v->length].j1 = j1;
    v->length++;
}

void boxes_free(Boxes *boxes) {
    if (!boxes) return;
    free(boxes->loc.data);
    free(boxes->asy.data);
    free(boxes->dir.data);
    boxes->loc.data = boxes->asy.data = boxes->dir.data = NULL;
    boxes->loc.length = boxes->asy.length = boxes->dir.length = 0;
    boxes->loc.capacity = boxes->asy.capacity = boxes->dir.capacity = 0;
}

/* ------------------ findfirst helpers ------------------ */

static int find_first_ws_greater(const double *ws, int n, double thresh) {
    for (int i = 0; i < n; ++i) {
        if (ws[i] > thresh) return i;
    }
    return -1;
}

static int find_first_rs_greater(const double *rs, int m, double thresh) {
    for (int j = 0; j < m; ++j) {
        if (rs[j] > thresh) return j;
    }
    return -1;
}

/* ================== setup_nufht (setup_nufht!) ================== */

int setup_nufht(double nu, double tol,
                double z_split,
                int    K_asy,
                int    K_loc)
{
    NufhtOptions opt = nufht_default_options();
    opt.tol     = tol;
    opt.z_split = z_split;
    opt.K_asy   = K_asy;
    opt.K_loc   = K_loc;

    NufhtPlan tmp;
    memset(&tmp, 0, sizeof(tmp));
    int rc = fill_plan_parameters(&tmp, nu, &opt);
    if (rc) return rc;

    /* Legacy globals populated for backward compatibility */
    NUFHT_NU      = tmp.nu;
    NUFHT_TOL     = tmp.tol;
    NUFHT_Z_SPLIT = tmp.z_split;
    NUFHT_ASY_K   = tmp.K_asy;
    NUFHT_LOC_K   = tmp.K_loc;

    return 0;
}

/* ================== splitting_objective & split_box ================== */

static long long splitting_objective(const double *rs, int m,
                                     const double *ws,
                                     const Box *box,
                                     double z,
                                     int i)
{
    int j = find_first_rs_greater(rs, m, z / ws[i]);
    if (j < 0) {
        /* Should not happen in valid configurations; treat as zero weight. */
        return 0;
    }

    int i0 = box->i0;
    int i1 = box->i1;
    int j0 = box->j0;
    int j1 = box->j1;

    long long d1a = (long long)(i - i0 + 1);
    long long d2a = (long long)(j - j0 + 1);
    long long d1b = (long long)(i1 - i + 1);
    long long d2b = (long long)(j1 - j + 1);

    return d1a * d2a + d1b * d2b;
}

void split_box(const double *rs, int m,
               const double *ws, int n,
               const Box    *box,
               double        z,
               int          *ispl_out,
               int          *jspl_out)
{
    (void)n; /* unused but kept for symmetry */

    const int num_check = 10;
    int i0 = box->i0;
    int i1 = box->i1;

    int candidates[10];   //  THIS SHOULD MATCH num_check
    int ncand = 0;

    if (i1 <= i0) {
        candidates[0] = i0;
        ncand = 1;
    } else {
        for (int k = 0; k < num_check; ++k) {
            double t = (num_check == 1) ? 0.0 : (double)k / (double)(num_check - 1);
            double idx_d = i0 + t * (double)(i1 - i0);
            int idx = (int)llround(idx_d);
            if (idx < i0) idx = i0;
            if (idx > i1) idx = i1;
            if (ncand == 0 || idx != candidates[ncand - 1]) {
                candidates[ncand++] = idx;
            }
        }
    }

    long long best_val = -1;
    int best_i = i0;

    for (int c = 0; c < ncand; ++c) {
        int i = candidates[c];
        long long val = splitting_objective(rs, m, ws, box, z, i);
        if (val > best_val) {
            best_val = val;
            best_i = i;
        }
    }

    int j = find_first_rs_greater(rs, m, z / ws[best_i]);
    if (j < 0) {
        j = box->j0; /* fallback */
    }

    *ispl_out = best_i;
    *jspl_out = j;
}

/* ================== generate_boxes ================== */

Boxes generate_boxes(const double *rs, int m,
                     const double *ws, int n,
                     double z_split,
                     int max_levels,
                     int min_dim_prod,
                     int K_loc_threshold)
{
    Boxes boxes;
    boxvec_init(&boxes.loc);
    boxvec_init(&boxes.asy);
    boxvec_init(&boxes.dir);

    if (n <= 0 || m <= 0) {
        return boxes;
    }

    /* max_levels default: floor(log2(min(n,m)^2 / min_dim_prod)) */
    if (max_levels <= 0) {
        int mn = (n < m) ? n : m;
        double ratio = (double)mn * (double)mn / (double)min_dim_prod;
        if (ratio < 1.0) {
            max_levels = 0;
        } else {
            max_levels = (int)floor(log2(ratio));
            if (max_levels < 0) max_levels = 0;
        }
    }

    /* Find i0, i1, j0, j1 (0-based) */

    int i0 = find_first_ws_greater(ws, n, z_split / rs[m - 1]);
    int i1;
    if (m == 1) {
        i1 = (i0 == -1) ? -1 : i0 - 1;
    } else {
        int idx = find_first_ws_greater(ws, n, z_split / rs[0]);
        if (idx == -1) {
            i1 = n - 1;
        } else {
            i1 = idx - 1;
        }
    }

    int j0 = find_first_rs_greater(rs, m, z_split / ws[n - 1]);
    int j1;
    if (n == 1) {
        j1 = (j0 == -1) ? -1 : j0 - 1;
    } else {
        int idx = find_first_rs_greater(rs, m, z_split / ws[0]);
        if (idx == -1) {
            j1 = m - 1;
        } else {
            j1 = idx - 1;
        }
    }

    /* All local or all asymptotic? */

    if (i0 == -1) {
        boxvec_push(&boxes.loc, 0, n - 1, 0, m - 1);
        return boxes;
    } else if (i1 < 0) {
        boxvec_push(&boxes.asy, 0, n - 1, 0, m - 1);
        return boxes;
    }

    /* Local-only strips */

    if (i0 > 0) {
        boxvec_push(&boxes.loc, 0, i0 - 1, 0, m - 1);
    }
    if (j0 > 0) {
        boxvec_push(&boxes.loc, i0, n - 1, 0, j0 - 1);
    }

    /* Asymptotic-only strips */

    if (i1 < n - 1) {
        boxvec_push(&boxes.asy, i1 + 1, n - 1, 0, m - 1);
    }
    if (j1 < m - 1) {
        boxvec_push(&boxes.asy, 0, i1, j1 + 1, m - 1);
    }

    if (m == 1 || n == 1) {
        return boxes;
    }

    /* Initial direct box (i0, i1, j0, j1) */

    boxvec_push(&boxes.dir, i0, i1, j0, j1);

    /* Hierarchical splitting */

    for (int level = 0; level < max_levels; ++level) {
        BoxVector new_dir;
        boxvec_init(&new_dir);

        for (size_t bi = 0; bi < boxes.dir.length; ++bi) {
            Box box = boxes.dir.data[bi];
            int i0b = box.i0, i1b = box.i1;
            int j0b = box.j0, j1b = box.j1;

            int d1 = i1b - i0b + 1;
            int d2 = j1b - j0b + 1;

            if ((long long)d1 * (long long)d2 > 4LL * (long long)min_dim_prod) {
                int ispl, jspl;
                split_box(rs, m, ws, n, &box, z_split, &ispl, &jspl);

                if (ispl > i0b) {
                    boxvec_push(&boxes.loc, i0b, ispl - 1, j0b, jspl - 1);
                    boxvec_push(&new_dir, i0b, ispl - 1, jspl, j1b);
                }

                boxvec_push(&boxes.asy, ispl, i1b, jspl, j1b);
                boxvec_push(&new_dir, ispl, i1b, j0b, jspl - 1);
            } else {
                boxvec_push(&new_dir, i0b, i1b, j0b, j1b);
            }
        }

        free(boxes.dir.data);
        boxes.dir = new_dir;
    }

    /* Move sufficiently small boxes to direct */

    BoxVector new_loc, new_asy;
    boxvec_init(&new_loc);
    boxvec_init(&new_asy);

    /* loc: min(d1,d2) < NUFHT_LOC_K -> direct */
    for (size_t bi = 0; bi < boxes.loc.length; ++bi) {
        Box box = boxes.loc.data[bi];
        int d1 = box.i1 - box.i0 + 1;
        int d2 = box.j1 - box.j0 + 1;
        int min_d = (d1 < d2) ? d1 : d2;

        if (min_d < K_loc_threshold) {
            boxvec_push(&boxes.dir, box.i0, box.i1, box.j0, box.j1);
        } else {
            boxvec_push(&new_loc, box.i0, box.i1, box.j0, box.j1);
        }
    }

    /* asy: d1*d2 < min_dim_prod -> direct */
    for (size_t bi = 0; bi < boxes.asy.length; ++bi) {
        Box box = boxes.asy.data[bi];
        int d1 = box.i1 - box.i0 + 1;
        int d2 = box.j1 - box.j0 + 1;
        long long area = (long long)d1 * (long long)d2;

        if (area < (long long)min_dim_prod) {
            boxvec_push(&boxes.dir, box.i0, box.i1, box.j0, box.j1);
        } else {
            boxvec_push(&new_asy, box.i0, box.i1, box.j0, box.j1);
        }
    }

    free(boxes.loc.data);
    free(boxes.asy.data);
    boxes.loc = new_loc;
    boxes.asy = new_asy;

    return boxes;
}

/* ================== NufhtOptions helper ================== */

NufhtOptions nufht_default_options(void) {
    NufhtOptions opt;
    opt.tol          = 1e-8;
    opt.max_levels   = 0;       /* auto */
    opt.min_dim_prod = 10000;
    opt.z_split      = NAN;     /* use table */
    opt.K_asy        = 0;       /* auto/table */
    opt.K_loc        = 0;       /* table */
    return opt;
}

int nufht_plan_init(NufhtPlan *plan,
                    double nu,
                    const double *rs, int m,
                    const double *ws, int n,
                    const NufhtOptions *opt_in)
{
    if (!plan || !rs || !ws || m <= 0 || n <= 0) return -200;

    NufhtOptions opt = opt_in ? *opt_in : nufht_default_options();
    memset(plan, 0, sizeof(*plan));

    plan->rs = rs;
    plan->ws = ws;
    plan->m  = m;
    plan->n  = n;
    plan->max_levels   = opt.max_levels;
    plan->min_dim_prod = opt.min_dim_prod;

    int rc = fill_plan_parameters(plan, nu, &opt);
    if (rc) return rc;

    plan->boxes = generate_boxes(rs, m, ws, n,
                                 plan->z_split,
                                 opt.max_levels,
                                 opt.min_dim_prod,
                                 plan->K_loc < 0 ? 0 : plan->K_loc);
    return 0;
}

void nufht_plan_free(NufhtPlan *plan)
{
    if (!plan) return;
    boxes_free(&plan->boxes);
    memset(plan, 0, sizeof(*plan));
}

static int compute_extra_bessel_len(double nu) {
    long n_int = (long)llround(nu);
    int extra = (int)(labs(n_int) / 2);
    if ((n_int & 1L) != 0L) extra += 1;
    return extra;
}

int nufht_scratch_init(NufhtScratch *scratch, const NufhtPlan *plan)
{
    if (!scratch || !plan) return -300;
    memset(scratch, 0, sizeof(*scratch));

    int m = plan->m;
    int n = plan->n;
    int Kloc = (plan->K_loc < 0) ? 0 : plan->K_loc;
    int extra = (plan->K_loc < 0) ? 0 : compute_extra_bessel_len(plan->nu);

    scratch->in_buffer  = (double *)calloc((size_t)m * 2, sizeof(double));
    scratch->out_buffer = (double *)calloc((size_t)n * 2, sizeof(double));
    scratch->real_buffer_1 = (double *)calloc((size_t)n, sizeof(double));
    scratch->real_buffer_2 = (double *)calloc((size_t)n, sizeof(double));
    scratch->cheb_buffer     = (double *)calloc((size_t)Kloc + 1, sizeof(double));
    scratch->bessel_buffer_1 = (double *)calloc((size_t)Kloc + 1, sizeof(double));
    scratch->bessel_buffer_2 = (double *)calloc((size_t)Kloc + 1 + (size_t)extra, sizeof(double));

    if (!scratch->in_buffer || !scratch->out_buffer ||
        !scratch->real_buffer_1 || !scratch->real_buffer_2 ||
        !scratch->cheb_buffer || !scratch->bessel_buffer_1 || !scratch->bessel_buffer_2) {
        nufht_scratch_free(scratch);
        return -301;
    }

    scratch->m_alloc = m;
    scratch->n_alloc = n;
    scratch->Kloc_alloc = Kloc;
    scratch->extra_alloc = extra;
    return 0;
}

void nufht_scratch_free(NufhtScratch *scratch)
{
    if (!scratch) return;
    free(scratch->in_buffer);
    free(scratch->out_buffer);
    free(scratch->real_buffer_1);
    free(scratch->real_buffer_2);
    free(scratch->cheb_buffer);
    free(scratch->bessel_buffer_1);
    free(scratch->bessel_buffer_2);
    memset(scratch, 0, sizeof(*scratch));
}

/* Helper: is nu an odd integer? */
static int is_odd_integer_double(double x) {
    long n = (long)llround(x);
    if (!is_integer_double(x)) return 0;
    return (n & 1L) != 0;
}

static void execute_boxes_single(const NufhtPlan *plan,
                                 const double *cs,
                                 double *gs,
                                 NufhtScratch *scratch)
{
    double absnu = fabs(plan->nu);
    int Kloc = plan->K_loc;
    int Kasy = plan->K_asy;

    /* Macros to loop over boxes */
#define FOR_EACH_BOX(BV, BODY)                                      \
    do {                                                            \
        for (size_t bi = 0; bi < (BV).length; ++bi) {               \
            Box b = (BV).data[bi];                                  \
            int i0b = b.i0, i1b = b.i1;                             \
            int j0b = b.j0, j1b = b.j1;                             \
            int ni = i1b - i0b + 1;                                 \
            int nj = j1b - j0b + 1;                                 \
            BODY;                                                   \
        }                                                           \
    } while (0)

    /* Local boxes */
    if (Kloc >= 0) {
        FOR_EACH_BOX(plan->boxes.loc, {
            add_loc(&gs[i0b], absnu,
                    &plan->rs[j0b], &cs[j0b],
                    &plan->ws[i0b],
                    nj, ni,
                    Kloc,
                    scratch->cheb_buffer,
                    scratch->bessel_buffer_1,
                    scratch->bessel_buffer_2);
        });
    }

    /* Asymptotic boxes */
    FOR_EACH_BOX(plan->boxes.asy, {
        add_asy(&gs[i0b], absnu,
                &plan->rs[j0b], &cs[j0b],
                &plan->ws[i0b],
                nj, ni,
                Kasy,
                plan->tol,
                plan->asy_coef,
                scratch->real_buffer_1,
                scratch->real_buffer_2,
                scratch->in_buffer,
                scratch->out_buffer);
    });

    /* Direct boxes */
    FOR_EACH_BOX(plan->boxes.dir, {
        add_dir(&gs[i0b], absnu,
                &plan->rs[j0b], &cs[j0b],
                &plan->ws[i0b],
                nj, ni);
    });

#undef FOR_EACH_BOX

    /* Negative odd nu: gs *= -1 */
    if (plan->nu < 0.0 && is_odd_integer_double(plan->nu)) {
        for (int i = 0; i < plan->n; ++i) {
            gs[i] = -gs[i];
        }
    }
}

static void execute_boxes_batch(const NufhtPlan *plan,
                                const double *cs, int ld_cs,
                                double *gs, int ld_gs,
                                int batch,
                                NufhtScratch *scratch,
                                double *cheb_batch,
                                double *asy_in,
                                double *asy_out)
{
    double absnu = fabs(plan->nu);
    int Kloc = plan->K_loc;
    int Kasy = plan->K_asy;

#define FOR_EACH_BOX(BV, BODY)                                      \
    do {                                                            \
        for (size_t bi = 0; bi < (BV).length; ++bi) {               \
            Box b = (BV).data[bi];                                  \
            int i0b = b.i0, i1b = b.i1;                             \
            int j0b = b.j0, j1b = b.j1;                             \
            int ni = i1b - i0b + 1;                                 \
            int nj = j1b - j0b + 1;                                 \
            BODY;                                                   \
        }                                                           \
    } while (0)

    if (Kloc >= 0) {
        FOR_EACH_BOX(plan->boxes.loc, {
            add_loc_batch(&gs[i0b], absnu,
                          &plan->rs[j0b], &cs[j0b],
                          &plan->ws[i0b],
                          nj, ni,
                          Kloc,
                          batch,
                          ld_cs, ld_gs,
                          cheb_batch,
                          scratch->bessel_buffer_1,
                          scratch->bessel_buffer_2);
        });
    }

    FOR_EACH_BOX(plan->boxes.asy, {
        add_asy_batch(&gs[i0b], absnu,
                      &plan->rs[j0b], &cs[j0b],
                      &plan->ws[i0b],
                      nj, ni,
                      Kasy,
                      plan->tol,
                      plan->asy_coef,
                      batch,
                      ld_cs, ld_gs,
                      asy_in,
                      asy_out);
    });

    FOR_EACH_BOX(plan->boxes.dir, {
        add_dir_batch(&gs[i0b], absnu,
                      &plan->rs[j0b], &cs[j0b],
                      &plan->ws[i0b],
                      nj, ni,
                      batch,
                      ld_cs, ld_gs);
    });

#undef FOR_EACH_BOX
}

int nufht_batch(const NufhtPlan *plan,
                const double *cs, int ld_cs,
                double *gs, int ld_gs,
                int batch,
                NufhtScratch *scratch_opt)
{
    if (!plan || !cs || !gs || batch <= 0) return -400;
    if (ld_cs < plan->m || ld_gs < plan->n) return -401;

    NufhtScratch local_scratch;
    NufhtScratch *scratch = scratch_opt ? scratch_opt : &local_scratch;
    int owns_scratch = (scratch_opt == NULL);

    if (owns_scratch) {
        if (nufht_scratch_init(scratch, plan) != 0) {
            fprintf(stderr, "nufht_batch: scratch alloc failed\n");
            return -402;
        }
    }

    long long dim_prod = (long long)plan->m * (long long)plan->n;
    int use_direct = (dim_prod < (long long)plan->min_dim_prod);
    int nu_is_int = is_integer_double(plan->nu);

    int need_asy = (!use_direct) && (!nu_is_int || plan->boxes.asy.length > 0);
    int need_loc = (!use_direct) && (nu_is_int && plan->boxes.loc.length > 0) && (plan->K_loc >= 0);

    double *asy_in = scratch->in_buffer;
    double *asy_out = scratch->out_buffer;
    double *cheb_batch = scratch->cheb_buffer;
    int owns_batch_buffers = 0;

    if (batch > 1 && (need_asy || need_loc)) {
        size_t in_len = (size_t)plan->m * (size_t)batch * 2;
        size_t out_len = (size_t)plan->n * (size_t)batch * 2;
        size_t cheb_len = need_loc ? (size_t)(plan->K_loc + 1) * (size_t)batch : 0;

        asy_in = need_asy ? (double *)calloc(in_len, sizeof(double)) : NULL;
        asy_out = need_asy ? (double *)calloc(out_len, sizeof(double)) : NULL;
        cheb_batch = need_loc ? (double *)calloc(cheb_len, sizeof(double)) : NULL;

        if ((need_asy && (!asy_in || !asy_out)) || (need_loc && !cheb_batch)) {
            fprintf(stderr, "nufht_batch: batch buffer alloc failed\n");
            free(asy_in);
            free(asy_out);
            free(cheb_batch);
            if (owns_scratch) {
                nufht_scratch_free(scratch);
            }
            return -403;
        }
        owns_batch_buffers = 1;
    }

    for (int t = 0; t < batch; ++t) {
        double *gs_t = gs + (size_t)t * (size_t)ld_gs;
        memset(gs_t, 0, (size_t)plan->n * sizeof(double));
    }

    if (use_direct) {
        add_dir_batch(gs, plan->nu, plan->rs, cs, plan->ws,
                      plan->m, plan->n,
                      batch, ld_cs, ld_gs);
    } else if (!nu_is_int) {
        if (batch > 1 && need_asy) {
            add_asy_batch(gs, fabs(plan->nu),
                          plan->rs, cs, plan->ws,
                          plan->m, plan->n,
                          plan->K_asy,
                          plan->tol,
                          plan->asy_coef,
                          batch,
                          ld_cs, ld_gs,
                          asy_in, asy_out);
        } else {
            for (int t = 0; t < batch; ++t) {
                double *gs_t = gs + (size_t)t * (size_t)ld_gs;
                const double *cs_t = cs + (size_t)t * (size_t)ld_cs;
                add_asy(gs_t, fabs(plan->nu),
                        plan->rs, cs_t, plan->ws,
                        plan->m, plan->n,
                        plan->K_asy,
                        plan->tol,
                        plan->asy_coef,
                        scratch->real_buffer_1,
                        scratch->real_buffer_2,
                        scratch->in_buffer,
                        scratch->out_buffer);
            }
        }
    } else {
        if (batch > 1) {
            execute_boxes_batch(plan, cs, ld_cs, gs, ld_gs, batch, scratch,
                                need_loc ? cheb_batch : scratch->cheb_buffer,
                                need_asy ? asy_in : scratch->in_buffer,
                                need_asy ? asy_out : scratch->out_buffer);
        } else {
            execute_boxes_single(plan, cs, gs, scratch);
        }
    }

    if (plan->nu < 0.0 && is_odd_integer_double(plan->nu)) {
        for (int t = 0; t < batch; ++t) {
            double *gs_t = gs + (size_t)t * (size_t)ld_gs;
            for (int i = 0; i < plan->n; ++i) gs_t[i] = -gs_t[i];
        }
    }

    if (owns_batch_buffers) {
        free(asy_in);
        free(asy_out);
        free(cheb_batch);
    }

    if (owns_scratch) {
        nufht_scratch_free(scratch);
    }
    return 0;
}

/* ================== nufht_inplace (nufht!) ================== */

void nufht_inplace(double       *gs,
                   double        nu,
                   const double *rs, int m,
                   const double *cs,
                   const double *ws, int n,
                   const NufhtOptions *opt_in)
{
    assert(gs && rs && cs && ws);
    assert(m > 0 && n > 0);

    NufhtOptions opt = opt_in ? *opt_in : nufht_default_options();

    /* Plan + batch wrapper for backward compatibility */
    NufhtPlan plan;
    if (nufht_plan_init(&plan, nu, rs, m, ws, n, &opt) != 0) {
        fprintf(stderr, "nufht_inplace: plan init failed\n");
        return;
    }

    NufhtScratch scratch;
    if (nufht_scratch_init(&scratch, &plan) != 0) {
        fprintf(stderr, "nufht_inplace: scratch alloc failed\n");
        nufht_plan_free(&plan);
        return;
    }

    memset(gs, 0, (size_t)n * sizeof(double));
    if (nufht_batch(&plan, cs, m, gs, n, 1, &scratch) != 0) {
        fprintf(stderr, "nufht_inplace: batch execution failed\n");
    }

    nufht_scratch_free(&scratch);
    nufht_plan_free(&plan);
}

/* ================== nufht wrapper ================== */

double *nufht(double        nu,
              const double *rs, int m,
              const double *cs,
              const double *ws, int n,
              const NufhtOptions *opt)
{
    double *gs = (double *)calloc((size_t)n, sizeof(double));
    if (!gs) return NULL;

    nufht_inplace(gs, nu, rs, m, cs, ws, n, opt);
    return gs;  /* caller must free(gs) */
}
