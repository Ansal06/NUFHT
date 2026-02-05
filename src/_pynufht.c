#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include "fast_hankel_nufht.h"
#include <stdlib.h>
#include <string.h>

/*
 * Minimal Python bindings for convenience:
 * - nufht(nu, rs, cs, ws, tol=1e-8) -> numpy array (length n)
 * - nufht_batch_loop(nu, rs, cs_matrix, ws, tol=1e-8) -> numpy array (n, batch)
 *   (loops over columns and calls `nufht()` per column)
 * - nufht_batch(nu, rs, cs_matrix, ws, tol=1e-8) -> numpy array (n, batch)
 *   (uses the native `nufht_batch()` with a plan/scratch)
 */

static PyObject *py_nufht_single(PyObject *self, PyObject *args, PyObject *kwargs)
{
    double nu;
    PyObject *rs_obj = NULL, *cs_obj = NULL, *ws_obj = NULL;
    double tol = 1e-8;
    static char *kwlist[] = {"nu", "rs", "cs", "ws", "tol", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dOOO|d", kwlist,
                                     &nu, &rs_obj, &cs_obj, &ws_obj, &tol))
        return NULL;

    PyObject *rs_arr = PyArray_FROM_OTF(rs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *cs_arr = PyArray_FROM_OTF(cs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *ws_arr = PyArray_FROM_OTF(ws_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!rs_arr || !cs_arr || !ws_arr) {
        Py_XDECREF(rs_arr); Py_XDECREF(cs_arr); Py_XDECREF(ws_arr);
        return NULL;
    }

    int m = (int)PyArray_DIM((PyArrayObject*)rs_arr, 0);
    int n = (int)PyArray_DIM((PyArrayObject*)ws_arr, 0);
    double *rs = (double*)PyArray_DATA((PyArrayObject*)rs_arr);
    double *cs = (double*)PyArray_DATA((PyArrayObject*)cs_arr);
    double *ws = (double*)PyArray_DATA((PyArrayObject*)ws_arr);

    NufhtOptions opt = nufht_default_options();
    opt.tol = tol;

    double *out = nufht(nu, rs, m, cs, ws, n, &opt);
    if (!out) {
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
        PyErr_SetString(PyExc_RuntimeError, "nufht returned NULL");
        return NULL;
    }

    npy_intp dims[1] = {n};
    PyObject *res = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    if (!res) {
        free(out);
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
        return PyErr_NoMemory();
    }
    memcpy(PyArray_DATA((PyArrayObject*)res), out, n * sizeof(double));
    free(out);

    Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
    return res;
}

static PyObject *py_nufht_batch_loop(PyObject *self, PyObject *args, PyObject *kwargs)
{
    double nu;
    PyObject *rs_obj = NULL, *cs_obj = NULL, *ws_obj = NULL;
    double tol = 1e-8;
    static char *kwlist[] = {"nu", "rs", "cs", "ws", "tol", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dOOO|d", kwlist,
                                     &nu, &rs_obj, &cs_obj, &ws_obj, &tol))
        return NULL;

    PyObject *rs_arr = PyArray_FROM_OTF(rs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *cs_arr = PyArray_FROM_OTF(cs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyObject *ws_arr = PyArray_FROM_OTF(ws_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!rs_arr || !cs_arr || !ws_arr) {
        Py_XDECREF(rs_arr); Py_XDECREF(cs_arr); Py_XDECREF(ws_arr);
        return NULL;
    }

    int m = (int)PyArray_DIM((PyArrayObject*)rs_arr, 0);
    int n = (int)PyArray_DIM((PyArrayObject*)ws_arr, 0);
    int nd = PyArray_NDIM((PyArrayObject*)cs_arr);
    if (nd != 2) {
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
        PyErr_SetString(PyExc_ValueError, "cs must be 2-D array with shape (m, batch)");
        return NULL;
    }
    int m_cs = (int)PyArray_DIM((PyArrayObject*)cs_arr, 0);
    int batch = (int)PyArray_DIM((PyArrayObject*)cs_arr, 1);
    if (m_cs != m) {
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
        PyErr_SetString(PyExc_ValueError, "cs first dimension must match rs length");
        return NULL;
    }

    double *rs = (double*)PyArray_DATA((PyArrayObject*)rs_arr);
    double *cs_data = (double*)PyArray_DATA((PyArrayObject*)cs_arr);
    double *ws = (double*)PyArray_DATA((PyArrayObject*)ws_arr);

    NufhtOptions opt = nufht_default_options();
    opt.tol = tol;

    npy_intp dims[2] = {n, batch};
    PyObject *res = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!res) {
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
        return PyErr_NoMemory();
    }
    double *res_data = (double*)PyArray_DATA((PyArrayObject*)res);

    /* temporary column vector */
    double *cs_col = (double*)malloc(m * sizeof(double));
    if (!cs_col) {
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr); Py_DECREF(res);
        return PyErr_NoMemory();
    }

    for (int b = 0; b < batch; ++b) {
        /* extract column b from cs_data which is C-contiguous row-major with shape (m,batch)
         * element (i,b) is at cs_data[i*batch + b]
         */
        for (int i = 0; i < m; ++i) cs_col[i] = cs_data[i * batch + b];

        double *out = nufht(nu, rs, m, cs_col, ws, n, &opt);
        if (!out) {
            free(cs_col);
            Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr); Py_DECREF(res);
            PyErr_SetString(PyExc_RuntimeError, "nufht returned NULL inside batch loop");
            return NULL;
        }
        /* store into result column b: res_data[i*batch + b]? shape (n,batch): element (i,b) index i*batch + b */
        for (int i = 0; i < n; ++i) res_data[i * batch + b] = out[i];
        free(out);
    }

    free(cs_col);
    Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
    return res;
}

static PyObject *py_nufht_batch(PyObject *self, PyObject *args, PyObject *kwargs)
{
    double nu;
    PyObject *rs_obj = NULL, *cs_obj = NULL, *ws_obj = NULL;
    double tol = 1e-8;
    static char *kwlist[] = {"nu", "rs", "cs", "ws", "tol", NULL};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "dOOO|d", kwlist,
                                     &nu, &rs_obj, &cs_obj, &ws_obj, &tol))
        return NULL;

    PyObject *rs_arr = PyArray_FROM_OTF(rs_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    /* Ensure cs is Fortran-contiguous (column-major) for nufht_batch */
    PyObject *cs_arr = PyArray_FROM_OTF(cs_obj, NPY_DOUBLE, NPY_ARRAY_F_CONTIGUOUS | NPY_ARRAY_ALIGNED);
    PyObject *ws_arr = PyArray_FROM_OTF(ws_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    if (!rs_arr || !cs_arr || !ws_arr) {
        Py_XDECREF(rs_arr); Py_XDECREF(cs_arr); Py_XDECREF(ws_arr);
        return NULL;
    }

    int m = (int)PyArray_DIM((PyArrayObject*)rs_arr, 0);
    int n = (int)PyArray_DIM((PyArrayObject*)ws_arr, 0);
    int nd = PyArray_NDIM((PyArrayObject*)cs_arr);
    if (nd != 2) {
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
        PyErr_SetString(PyExc_ValueError, "cs must be 2-D array with shape (m, batch)");
        return NULL;
    }
    int m_cs = (int)PyArray_DIM((PyArrayObject*)cs_arr, 0);
    int batch = (int)PyArray_DIM((PyArrayObject*)cs_arr, 1);
    if (m_cs != m) {
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
        PyErr_SetString(PyExc_ValueError, "cs first dimension must match rs length");
        return NULL;
    }

    double *rs = (double*)PyArray_DATA((PyArrayObject*)rs_arr);
    double *cs_data = (double*)PyArray_DATA((PyArrayObject*)cs_arr); /* column-major */
    double *ws = (double*)PyArray_DATA((PyArrayObject*)ws_arr);

    NufhtOptions opt = nufht_default_options();
    opt.tol = tol;

    NufhtPlan plan;
    if (nufht_plan_init(&plan, nu, rs, m, ws, n, &opt) != 0) {
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
        PyErr_SetString(PyExc_RuntimeError, "nufht_plan_init failed");
        return NULL;
    }
    NufhtScratch scratch;
    if (nufht_scratch_init(&scratch, &plan) != 0) {
        nufht_plan_free(&plan);
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
        PyErr_SetString(PyExc_RuntimeError, "nufht_scratch_init failed");
        return NULL;
    }

    /* allocate temporary column-major output buffer */
    double *gs_buf = (double*)malloc((size_t)n * (size_t)batch * sizeof(double));
    if (!gs_buf) {
        nufht_scratch_free(&scratch);
        nufht_plan_free(&plan);
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
        return PyErr_NoMemory();
    }

    int ld_cs = m; /* Fortran stride */
    int ld_gs = n;
    int ret = nufht_batch(&plan, cs_data, ld_cs, gs_buf, ld_gs, batch, &scratch);
    if (ret != 0) {
        free(gs_buf);
        nufht_scratch_free(&scratch);
        nufht_plan_free(&plan);
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
        PyErr_Format(PyExc_RuntimeError, "nufht_batch returned %d", ret);
        return NULL;
    }

    /* create C-order numpy array and copy data (convert column-major -> row-major) */
    npy_intp dims[2] = {n, batch};
    PyObject *res = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    if (!res) {
        free(gs_buf);
        nufht_scratch_free(&scratch);
        nufht_plan_free(&plan);
        Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
        return PyErr_NoMemory();
    }
    double *res_data = (double*)PyArray_DATA((PyArrayObject*)res);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < n; ++i) {
            /* res_data is C-order: index = i*batch + b */
            res_data[i * batch + b] = gs_buf[i + (size_t)b * (size_t)n];
        }
    }

    free(gs_buf);
    nufht_scratch_free(&scratch);
    nufht_plan_free(&plan);
    Py_DECREF(rs_arr); Py_DECREF(cs_arr); Py_DECREF(ws_arr);
    return res;
}

static PyMethodDef PynufhtMethods[] = {
    {"nufht", (PyCFunction)py_nufht_single, METH_VARARGS | METH_KEYWORDS,
     "Compute single NUFHT: nufht(nu, rs, cs, ws, tol=1e-8) -> array(n)"},
    {"nufht_batch_loop", (PyCFunction)py_nufht_batch_loop, METH_VARARGS | METH_KEYWORDS,
     "Compute batch NUFHT by looping: nufht_batch_loop(nu, rs, cs_matrix, ws, tol=1e-8) -> array(n,batch)"},
    {"nufht_batch", (PyCFunction)py_nufht_batch, METH_VARARGS | METH_KEYWORDS,
     "Compute batched NUFHT using plan/scratch: nufht_batch(nu, rs, cs_matrix, ws, tol=1e-8) -> array(n,batch)"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef pynufhtmodule = {
    PyModuleDef_HEAD_INIT,
    "_pynufht",
    "Python bindings for NUFHT (simple)",
    -1,
    PynufhtMethods
};

PyMODINIT_FUNC PyInit__pynufht(void)
{
    PyObject *m;
    m = PyModule_Create(&pynufhtmodule);
    if (m == NULL) return NULL;
    import_array();
    return m;
}
