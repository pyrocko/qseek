#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN /* Make "s#" use Py_ssize_t rather than int. */
#define BLOCK_SIZE 64

#include "numpy/arrayobject.h"
#include <Python.h>
#include <float.h>
#include <omp.h>

static inline npy_intp min_intp(npy_intp a, npy_intp b) {
  return a < b ? a : b;
}

static PyObject *fill_zero_bytes(PyObject *module, PyObject *args,
                                 PyObject *kwds) {
  PyObject *array;
  int n_threads = 8;
  int thread_num;
  npy_intp n_bytes, start, size;

  static char *kwlist[] = {"array", "n_threads", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|i", kwlist, &array,
                                   &n_threads))
    return NULL;

  if (!PyArray_Check(array)) {
    PyErr_SetString(PyExc_ValueError, "object is not a NumPy array");
    return NULL;
  }
  if (n_threads < 0) {
    PyErr_SetString(PyExc_ValueError, "n_threads must be greater than 0");
    return NULL;
  }

  Py_BEGIN_ALLOW_THREADS;
  n_bytes = PyArray_NBYTES((PyArrayObject *)array);
#pragma omp parallel num_threads(n_threads) private(thread_num, start, size)
  {
    thread_num = omp_get_thread_num();
    start = (thread_num * n_bytes) / n_threads;
    size = ((thread_num + 1) * n_bytes) / n_threads - start;

    memset(PyArray_DATA((PyArrayObject *)array) + start, 0, size);
  }
  Py_END_ALLOW_THREADS;
  Py_RETURN_NONE;
}

static PyObject *argmax(PyObject *module, PyObject *args, PyObject *kwds) {
  PyObject *obj, *result_max_idx, *result_max_values;
  PyObject *node_mask = Py_None;
  PyArrayObject *data_arr, *node_mask_arr;

  npy_intp *shape, shapeout[1], block_length, *result_max_idx_data;
  npy_intp ix, i_node, n_nodes, n_samples, ix_offset, idx_max[BLOCK_SIZE];
  float val_max[BLOCK_SIZE], *result_max_values_data, value;

  int n_threads = 8;

  static char *kwlist[] = {"array", "mask", "n_threads", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|Oi", kwlist, &obj, &node_mask,
                                   &n_threads))
    return NULL;

  if (!PyArray_Check(obj)) {
    PyErr_SetString(PyExc_ValueError, "array is not a NumPy array");
    return NULL;
  }

  data_arr = (PyArrayObject *)obj;
  if (PyArray_TYPE(data_arr) != NPY_FLOAT) {
    PyErr_SetString(PyExc_ValueError, "Bad dtype, only float32 is supported.");
    return NULL;
  }
  if (PyArray_NDIM(data_arr) != 2) {
    PyErr_SetString(PyExc_ValueError, "array is not 2D");
    return NULL;
  }
  if (!PyArray_IS_C_CONTIGUOUS(data_arr)) {
    PyErr_SetString(PyExc_ValueError, "array is not C contiguous");
    return NULL;
  }

  shape = PyArray_DIMS(data_arr);
  n_nodes = shape[0];
  n_samples = shape[1];
  shapeout[0] = n_samples;

  if (node_mask != Py_None) {
    if (!PyArray_Check(node_mask)) {
      PyErr_SetString(PyExc_ValueError, "mask is not a NumPy array");
      return NULL;
    }
    node_mask_arr = (PyArrayObject *)node_mask;
    if (PyArray_NDIM(node_mask_arr) != 1) {
      PyErr_SetString(PyExc_ValueError, "mask is not a 1D NumPy array");
      return NULL;
    }
    if (PyArray_SIZE(node_mask_arr) != n_nodes) {
      PyErr_SetString(PyExc_ValueError, "mask size does not match array");
      return NULL;
    }
    if (PyArray_TYPE(node_mask_arr) != NPY_BOOL) {
      PyErr_SetString(PyExc_ValueError, "mask is not of type np.bool");
      return NULL;
    }
  }

  result_max_idx = PyArray_ZEROS(1, shapeout, NPY_INTP, 0);
  result_max_values = PyArray_ZEROS(1, shapeout, NPY_FLOAT32, 0);
  result_max_idx_data =
      (npy_intp *)PyArray_DATA((PyArrayObject *)result_max_idx);
  result_max_values_data =
      (float *)PyArray_DATA((PyArrayObject *)result_max_values);

  Py_BEGIN_ALLOW_THREADS;

#pragma omp parallel for private(i_node, ix_offset, idx_max, val_max,          \
                                     block_length, value)                      \
    num_threads(n_threads) schedule(dynamic, 1)
  for (ix = 0; ix < n_samples; ix += BLOCK_SIZE) {
    block_length = min_intp(BLOCK_SIZE, n_samples - ix);

#pragma omp simd
    for (ix_offset = 0; ix_offset < block_length; ix_offset++) {
      idx_max[ix_offset] = 0;
      val_max[ix_offset] = FLT_MIN;
    }

    for (i_node = 0; i_node < n_nodes; i_node++) {
      if (node_mask != Py_None &&
          !*(npy_bool *)PyArray_GETPTR1(node_mask_arr, i_node)) {
        continue;
      }
      for (ix_offset = 0; ix_offset < block_length; ix_offset++) {
        value = *(float *)PyArray_GETPTR2(data_arr, i_node, ix + ix_offset);
        if (value > val_max[ix_offset]) {
          val_max[ix_offset] = value;
          idx_max[ix_offset] = i_node;
        }
      }
    }

#pragma omp simd
    for (ix_offset = 0; ix_offset < block_length; ix_offset++) {
      result_max_idx_data[ix + ix_offset] = idx_max[ix_offset];
      result_max_values_data[ix + ix_offset] = val_max[ix_offset];
    }
  }
  Py_END_ALLOW_THREADS;

  return Py_BuildValue("NN", (PyObject *)result_max_idx,
                       (PyObject *)result_max_values);
}

static PyMethodDef methods[] = {
    {"fill_zero_bytes", (PyCFunction)(void (*)(void))fill_zero_bytes,
     METH_VARARGS | METH_KEYWORDS, "Fill a numpy array with zero bytes."},
    {"argmax_masked", (PyCFunction)(void (*)(void))argmax,
     METH_VARARGS | METH_KEYWORDS,
     "Find the argmax of a 2D numpy array on axis 0."},
    {NULL, NULL, 0, NULL} /* sentinel */
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "array_tools", NULL, -1, methods,
};

PyMODINIT_FUNC PyInit_array_tools(void) {
  import_array();
  return PyModule_Create(&module);
}
