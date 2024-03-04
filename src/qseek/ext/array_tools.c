#define PY_SSIZE_T_CLEAN /* Make "s#" use Py_ssize_t rather than int. */
#include "numpy/arrayobject.h"
#include <Python.h>
#include <omp.h>

static PyObject *fill_zero_bytes(PyObject *module, PyObject *args,
                                 PyObject *kwds) {
  PyObject *array;
  static char *kwlist[] = {"array", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &array))
    return NULL;

  if (!PyArray_Check(array)) {
    PyErr_SetString(PyExc_ValueError, "weights is not a NumPy array");
    return NULL;
  }
  Py_BEGIN_ALLOW_THREADS;
  memset(PyArray_DATA((PyArrayObject *)array), 0,
         PyArray_NBYTES((PyArrayObject *)array));
  Py_END_ALLOW_THREADS;
  Py_RETURN_NONE;
}

static PyObject *apply_cache(PyObject *module, PyObject *args, PyObject *kwds) {
  PyObject *array, *cache, *mask;
  int nthreads = 4;
  npy_intp *array_shape;
  npy_intp n_nodes, n_samples;

  npy_int *cumsum_mask, mask_value;
  npy_int idx_sum = 0;
  npy_bool *mask_data;
  PyArrayObject *cached_array;

  static char *kwlist[] = {"array", "cache", "mask", "nthreads", NULL};

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOO|i", kwlist, &array, &cache,
                                   &mask, &nthreads))
    return NULL;

  if (!PyArray_Check(array)) {
    PyErr_SetString(PyExc_ValueError, "array is not a NumPy array");
    return NULL;
  }
  if (PyArray_NDIM(array) != 2) {
    PyErr_SetString(PyExc_ValueError, "array is not a 2D NumPy array");
    return NULL;
  }
  if (!PyArray_IS_C_CONTIGUOUS(array)) {
    PyErr_SetString(PyExc_ValueError, "array is not C contiguous");
    return NULL;
  }
  if (PyArray_TYPE(array) != NPY_FLOAT) {
    fprintf(stderr, "array type: %d %d\n", PyArray_TYPE(array), NPY_FLOAT);
    PyErr_SetString(PyExc_ValueError, "array is not of type np.float32");
    return NULL;
  }
  array_shape = PyArray_SHAPE((PyArrayObject *)array);
  n_nodes = array_shape[0];
  n_samples = array_shape[1];

  if (!PyArray_Check(mask)) {
    PyErr_SetString(PyExc_ValueError, "mask is not a NumPy array");
    return NULL;
  }
  if (PyArray_NDIM(mask) != 1) {
    PyErr_SetString(PyExc_ValueError, "mask is not a 2D NumPy array");
    return NULL;
  }
  if (PyArray_SIZE(mask) != n_nodes) {
    PyErr_SetString(PyExc_ValueError, "mask size does not match array");
    return NULL;
  }

  if (!PyList_Check(cache)) {
    PyErr_SetString(PyExc_ValueError, "cache is not a list");
    return NULL;
  }

  for (int i_node = 0; i_node < PyList_Size(cache); i_node++) {
    PyObject *item = PyList_GetItem(cache, i_node);
    if (!PyArray_Check(item)) {
      PyErr_SetString(PyExc_ValueError, "cache item is not a NumPy array");
      return NULL;
    }
    if (PyArray_TYPE(item) != NPY_FLOAT) {
      PyErr_SetString(PyExc_ValueError, "cache item is not of type np.float32");
      return NULL;
    }
    if (PyArray_NDIM(item) != 1) {
      PyErr_SetString(PyExc_ValueError, "cache item is not a 1D NumPy array");
      return NULL;
    }
    if (!PyArray_IS_C_CONTIGUOUS(item)) {
      PyErr_SetString(PyExc_ValueError, "cache item is not C contiguous");
      return NULL;
    }
    if (PyArray_SIZE(item) != n_samples) {
      PyErr_SetString(PyExc_ValueError, "cache item size does not match array");
      return NULL;
    }
  }

  // cumsum mask

  cumsum_mask = (npy_int *)malloc(n_nodes * sizeof(npy_int));
  mask_data = PyArray_DATA((PyArrayObject *)mask);
  for (int i_node = 0; i_node < n_nodes; i_node++) {
    mask_value = mask_data[i_node];
    if (!mask_value) {
      cumsum_mask[i_node] = -1;
    } else {
      cumsum_mask[i_node] = idx_sum;
      idx_sum += 1;
    }
  }
  Py_BEGIN_ALLOW_THREADS;
  for (int i_node = 0; i_node < n_nodes; i_node++) {
    if (cumsum_mask[i_node] == -1) {
      continue;
    }
    cached_array = PyList_GET_ITEM(cache, (Py_ssize_t)cumsum_mask[i_node]);
    memcpy(
        PyArray_GETPTR2((PyArrayObject *)array, (npy_intp)i_node, (npy_intp)0),
        PyArray_DATA((PyArrayObject *)cached_array),
        n_samples * sizeof(npy_float32));
  }
  Py_END_ALLOW_THREADS;

  Py_RETURN_NONE;
}

static PyMethodDef methods[] = {
    /* The cast of the function is necessary since PyCFunction values
     * only take two PyObject* parameters, and fill_zero_bytes() takes
     * three.
     */
    {"fill_zero_bytes", (PyCFunction)(void (*)(void))fill_zero_bytes,
     METH_VARARGS | METH_KEYWORDS, "Fill a numpy array with zero bytes."},
    {"apply_cache", (PyCFunction)(void (*)(void))apply_cache,
     METH_VARARGS | METH_KEYWORDS,
     "Apply a cache to a 2D numpy array of type float32."},
    {NULL, NULL, 0, NULL} /* sentinel */
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "array_tools", NULL, -1, methods,
};

PyMODINIT_FUNC PyInit_array_tools(void) {
  import_array();
  return PyModule_Create(&module);
}
