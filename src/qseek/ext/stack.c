#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <immintrin.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

// Structure definitions equivalent to Mojo's structs
typedef struct {
  int32_t *shifts;
  float *weights;
} Node;

typedef struct {
  int32_t *shifts;
  float *weights;
  float *stack;
} NodeWithStack;

typedef struct {
  float *data;
  Py_ssize_t size;
  int32_t offset;
} Trace;

// Function to get thread count
static inline int get_thread_count(int n_threads) {
  if (n_threads <= 0) {
    return omp_get_max_threads();
  }
  return n_threads;
}

static inline Py_ssize_t imax(Py_ssize_t a, Py_ssize_t b) {
  return a > b ? a : b;
}
static inline Py_ssize_t imin(Py_ssize_t a, Py_ssize_t b) {
  return a < b ? a : b;
}

// Function to check NumPy array dtype
static inline int check_array_dtype(PyArrayObject *arr, int expected_type) {
  if (PyArray_TYPE(arr) != expected_type) {
    PyErr_Format(PyExc_TypeError, "Input array must be of type %s",
                 expected_type == NPY_FLOAT32 ? "float32" : "unknown");
    return 0;
  }
  return 1;
}

// Prepare function equivalent to Mojo's prepare
static PyObject *prepare(PyObject *traces, PyObject *offsets, PyObject *shifts,
                         PyObject *weights, Trace **traces_list,
                         Node **nodes_list, int32_t *min_shift,
                         int32_t *max_shift) {
  Py_ssize_t n_traces = PyList_Size(traces);
  PyArrayObject *shifts_arr =
      (PyArrayObject *)PyArray_ContiguousFromObject(shifts, NPY_INT32, 2, 2);
  PyArrayObject *weights_arr =
      (PyArrayObject *)PyArray_ContiguousFromObject(weights, NPY_FLOAT32, 2, 2);
  PyArrayObject *offsets_arr =
      (PyArrayObject *)PyArray_ContiguousFromObject(offsets, NPY_INT32, 1, 1);

  if (!shifts_arr || !weights_arr || !offsets_arr) {
    Py_XDECREF(shifts_arr);
    Py_XDECREF(weights_arr);
    Py_XDECREF(offsets_arr);
    return NULL;
  }

  if (n_traces == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "Input traces must have positive dimensions");
    goto cleanup;
  }

  npy_intp *shifts_shape = PyArray_SHAPE(shifts_arr);
  Py_ssize_t n_nodes = shifts_shape[0];
  if (n_nodes == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "Input arrays must have positive dimensions");
    goto cleanup;
  }

  if (!check_array_dtype(weights_arr, NPY_FLOAT32) ||
      !check_array_dtype(offsets_arr, NPY_INT32) ||
      !check_array_dtype(shifts_arr, NPY_INT32)) {
    goto cleanup;
  }

  if (shifts_shape[0] != PyArray_SHAPE(weights_arr)[0] ||
      shifts_shape[1] != PyArray_SHAPE(weights_arr)[1]) {
    PyErr_SetString(PyExc_ValueError,
                    "Shifts and weights must have the same shape");
    goto cleanup;
  }
  if (n_traces != PyArray_SHAPE(offsets_arr)[0]) {
    PyErr_SetString(PyExc_ValueError,
                    "Number of arrays must match number of offsets");
    goto cleanup;
  }
  if (shifts_shape[1] != n_traces) {
    PyErr_SetString(PyExc_ValueError,
                    "Shifts must have the same number of columns as traces");
    goto cleanup;
  }

  int32_t *offsets_data = (int32_t *)PyArray_DATA(offsets_arr);
  int32_t *shifts_data = (int32_t *)PyArray_DATA(shifts_arr);
  float *weights_data = (float *)PyArray_DATA(weights_arr);

  *traces_list = (Trace *)malloc(n_traces * sizeof(Trace));
  *nodes_list = (Node *)malloc(n_nodes * sizeof(Node));
  if (!*traces_list || !*nodes_list) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
    goto cleanup;
  }

  for (Py_ssize_t i = 0; i < n_traces; i++) {
    PyArrayObject *trace = (PyArrayObject *)PyArray_ContiguousFromObject(
        PyList_GetItem(traces, i), NPY_FLOAT32, 1, 1);
    if (!trace)
      goto cleanup_traces;
    if (!check_array_dtype(trace, NPY_FLOAT32)) {
      Py_DECREF(trace);
      goto cleanup_traces;
    }
    (*traces_list)[i].data = (float *)PyArray_DATA(trace);
    (*traces_list)[i].size = PyArray_SIZE(trace);
    (*traces_list)[i].offset = offsets_data[i];
    Py_DECREF(trace); // We keep the data pointer, but release the array object
  }

  for (Py_ssize_t i = 0; i < n_nodes; i++) {
    (*nodes_list)[i].shifts = shifts_data + i * n_traces;
    (*nodes_list)[i].weights = weights_data + i * n_traces;
  }

  *min_shift = INT32_MAX;
  *max_shift = INT32_MIN;
  for (Py_ssize_t i = 0; i < n_nodes; i++) {
    Node node = (*nodes_list)[i];
    for (Py_ssize_t j = 0; j < n_traces; j++) {
      int32_t idx_begin = (*traces_list)[j].offset + node.shifts[j];
      int32_t idx_end = idx_begin + (*traces_list)[j].size;
      *min_shift = (*min_shift < idx_begin) ? *min_shift : idx_begin;
      *max_shift = (*max_shift > idx_end) ? *max_shift : idx_end;
    }
  }

  Py_DECREF(shifts_arr);
  Py_DECREF(weights_arr);
  Py_DECREF(offsets_arr);
  return traces;

cleanup_traces:
  free(*traces_list);
  free(*nodes_list);
cleanup:
  Py_XDECREF(shifts_arr);
  Py_XDECREF(weights_arr);
  Py_XDECREF(offsets_arr);
  return NULL;
}

static PyObject *stack_traces(PyObject *self, PyObject *args,
                              PyObject *kwargs) {
  PyObject *traces, *offsets, *shifts, *weights, *result;
  result = Py_None; // Default to None if not provided
  int n_threads = 1;
  int result_samples = 0;

  static char *kwlist[] = {"traces", "offsets",        "shifts",    "weights",
                           "result", "result_samples", "n_threads", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|Oii", kwlist, &traces,
                                   &offsets, &shifts, &weights, &result,
                                   &result_samples, &n_threads)) {
    return NULL;
  }

  Trace *traces_list;
  Node *nodes_list;
  int32_t min_shift, max_shift;
  if (!prepare(traces, offsets, shifts, weights, &traces_list, &nodes_list,
               &min_shift, &max_shift)) {
    return NULL;
  }

  Py_ssize_t n_traces = PyList_Size(traces);
  Py_ssize_t n_nodes = PyArray_SHAPE((PyArrayObject *)shifts)[0];
  Py_ssize_t result_length = max_shift - min_shift;
  if (result_samples > 0) {
    result_length = (Py_ssize_t)result_samples;
    min_shift = 0;
  }

  PyObject *result_arr;
  if (result == Py_None) {
    npy_intp dims[2] = {n_nodes, result_length};
    result_arr = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (!result_arr) {
      free(traces_list);
      free(nodes_list);
      return NULL;
    }
  } else {
    result_arr = PyArray_ContiguousFromObject(result, NPY_FLOAT32, 2, 2);
    if (!result_arr ||
        !check_array_dtype((PyArrayObject *)result_arr, NPY_FLOAT32)) {
      free(traces_list);
      free(nodes_list);
      return NULL;
    }
    npy_intp *shape = PyArray_SHAPE((PyArrayObject *)result_arr);
    if (shape[0] != n_nodes || shape[1] != result_length) {
      PyErr_SetString(PyExc_ValueError,
                      "Result array must have shape (n_nodes, length_out)");
      Py_DECREF(result_arr);
      free(traces_list);
      free(nodes_list);
      return NULL;
    }
  }

  float *result_data = (float *)PyArray_DATA((PyArrayObject *)result_arr);
  NodeWithStack *node_list =
      (NodeWithStack *)malloc(n_nodes * sizeof(NodeWithStack));
  if (!node_list) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate node_list");
    Py_DECREF(result_arr);
    free(traces_list);
    free(nodes_list);
    return NULL;
  }

  for (Py_ssize_t i = 0; i < n_nodes; i++) {
    node_list[i].shifts = nodes_list[i].shifts;
    node_list[i].weights = nodes_list[i].weights;
    node_list[i].stack = result_data + i * result_length;
  }

  Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel for num_threads(get_thread_count(n_threads))
  for (Py_ssize_t i_node = 0; i_node < n_nodes; i_node++) {
    NodeWithStack node = node_list[i_node];
    for (Py_ssize_t i_trace = 0; i_trace < n_traces; i_trace++) {
      float weight = node.weights[i_trace];
      if (weight == 0.0f)
        continue;

      Trace trace = traces_list[i_trace];
      int32_t trace_shift = trace.offset + node.shifts[i_trace];
      int32_t base_idx = trace_shift - min_shift;
      Py_ssize_t stack_nsamples = imin(result_length - base_idx, trace.size);

      Py_ssize_t i;
      __m256 weight_vec = _mm256_set1_ps(weight);

      for (i = imax(0, min_shift - trace_shift);
           i < stack_nsamples - (stack_nsamples % 8); i += 8) {
        Py_ssize_t i_res = base_idx + i;
        __m256 trace_vec = _mm256_loadu_ps(&trace.data[i]);
        __m256 stack_vec = _mm256_loadu_ps(&node.stack[i_res]);
        stack_vec = _mm256_fmadd_ps(trace_vec, weight_vec, stack_vec);
        _mm256_storeu_ps(&node.stack[i_res], stack_vec);
      }
      for (; i < stack_nsamples; i++) {
        Py_ssize_t i_res = base_idx + i;
        node.stack[i_res] += trace.data[i] * weight;
      }
    }
  }
  Py_END_ALLOW_THREADS;

  free(node_list);
  free(traces_list);
  free(nodes_list);

  PyObject *ret = PyTuple_New(2);
  PyTuple_SetItem(ret, 0, result_arr);
  PyTuple_SetItem(ret, 1, PyLong_FromLong(min_shift));
  return ret;
}

// stack_and_reduce function
static PyObject *stack_and_reduce(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyObject *traces, *offsets, *shifts, *weights;
  int n_threads = 1;

  static char *kwlist[] = {"traces",  "offsets",   "shifts",
                           "weights", "n_threads", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|i", kwlist, &traces,
                                   &offsets, &shifts, &weights, &n_threads)) {
    return NULL;
  }

  Trace *traces_list;
  Node *nodes_list;
  int32_t min_shift, max_shift;
  if (!prepare(traces, offsets, shifts, weights, &traces_list, &nodes_list,
               &min_shift, &max_shift)) {
    return NULL;
  }

  Py_ssize_t n_traces = PyList_Size(traces);
  Py_ssize_t n_nodes = PyArray_SHAPE((PyArrayObject *)shifts)[0];
  Py_ssize_t result_length = max_shift - min_shift;

  npy_intp dims = result_length;
  PyObject *node_max = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  PyObject *node_argmax = PyArray_SimpleNew(1, &dims, NPY_UINTP);
  if (!node_max || !node_argmax) {
    Py_XDECREF(node_max);
    Py_XDECREF(node_argmax);
    free(traces_list);
    free(nodes_list);
    return NULL;
  }

  float *node_max_data = (float *)PyArray_DATA((PyArrayObject *)node_max);
  uint64_t *node_argmax_data =
      (uint64_t *)PyArray_DATA((PyArrayObject *)node_argmax);
  for (Py_ssize_t i = 0; i < result_length; i++) {
    node_max_data[i] = -NPY_INFINITYF;
    node_argmax_data[i] = 0;
  }
  Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(get_thread_count(n_threads))
  {
    Py_ssize_t tile_start_idx =
        omp_get_thread_num() * result_length / get_thread_count(n_threads);
    Py_ssize_t tile_end_idx = (omp_get_thread_num() + 1) * result_length /
                              get_thread_count(n_threads);
    Py_ssize_t tile_size = tile_end_idx - tile_start_idx;
    float *tile_node_stack =
        (float *)aligned_alloc(32, tile_size * sizeof(float));

    for (Py_ssize_t i_node = 0; i_node < n_nodes; i_node++) {
      Node node = nodes_list[i_node];
      memset(tile_node_stack, 0, tile_size * sizeof(float));

      for (Py_ssize_t i_trace = 0; i_trace < n_traces; i_trace++) {
        float weight = node.weights[i_trace];
        if (weight == 0.0f)
          continue;
        Trace trace = traces_list[i_trace];
        int32_t trace_shift = trace.offset + node.shifts[i_trace];
        int32_t base_idx = trace_shift - min_shift;
        Py_ssize_t tile_base_idx = imax(0, base_idx - tile_start_idx);
        Py_ssize_t trace_start_idx = imax(0, tile_start_idx - base_idx);
        Py_ssize_t trace_end_idx = imax(0, tile_end_idx - base_idx);
        trace_start_idx = imin(trace_start_idx, trace.size);
        trace_end_idx = imin(trace_end_idx, trace.size);
        Py_ssize_t n_samples = trace_end_idx - trace_start_idx;

        Py_ssize_t i;
        __m256 weight_vec = _mm256_set1_ps(weight);

        for (i = 0; i < n_samples - (n_samples % 8); i += 8) {
          Py_ssize_t i_res = tile_base_idx + i;
          __m256 trace_vec = _mm256_loadu_ps(&trace.data[trace_start_idx + i]);
          __m256 stack_vec = _mm256_load_ps(&tile_node_stack[i_res]);
          stack_vec = _mm256_fmadd_ps(trace_vec, weight_vec, stack_vec);
          _mm256_storeu_ps(&tile_node_stack[i_res], stack_vec);
        }
        for (; i < n_samples; i++) {
          Py_ssize_t i_res = tile_base_idx + i;
          tile_node_stack[i_res] += trace.data[trace_start_idx + i] * weight;
        }
      }

      for (Py_ssize_t i = 0; i < tile_size; i++) {
        Py_ssize_t tile_idx = tile_start_idx + i;
        float node_val = tile_node_stack[i];
        if (node_val > node_max_data[tile_idx]) {
          node_max_data[tile_idx] = node_val;
          node_argmax_data[tile_idx] = i_node;
        }
      }
    }
    free(tile_node_stack);
  }
  Py_END_ALLOW_THREADS;

  free(traces_list);
  free(nodes_list);

  PyObject *ret = PyTuple_New(3);
  PyTuple_SetItem(ret, 0, node_max);
  PyTuple_SetItem(ret, 1, node_argmax);
  PyTuple_SetItem(ret, 2, PyLong_FromLong(min_shift));
  return ret;
}

// stack_snapshot function
static PyObject *stack_snapshot(PyObject *self, PyObject *args,
                                PyObject *kwargs) {
  PyObject *traces, *offsets, *shifts, *weights;
  int32_t index;

  static char *kwlist[] = {"traces",  "offsets", "shifts",
                           "weights", "index",   NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOi", kwlist, &traces,
                                   &offsets, &shifts, &weights, &index)) {
    return NULL;
  }

  Trace *traces_list;
  Node *nodes_list;
  int32_t min_shift, max_shift;
  if (!prepare(traces, offsets, shifts, weights, &traces_list, &nodes_list,
               &min_shift, &max_shift)) {
    return NULL;
  }

  Py_ssize_t n_traces = PyList_Size(traces);
  Py_ssize_t n_nodes = PyArray_SHAPE((PyArrayObject *)shifts)[0];
  Py_ssize_t result_length = max_shift - min_shift;

  if (index >= result_length || index < 0) {
    PyErr_Format(PyExc_ValueError, "Snapshot index out of bounds: %d", index);
    free(traces_list);
    free(nodes_list);
    return NULL;
  }

  npy_intp dims = n_nodes;
  PyObject *result = PyArray_SimpleNew(1, &dims, NPY_FLOAT32);
  if (!result) {
    free(traces_list);
    free(nodes_list);
    return NULL;
  }

  float *result_data = (float *)PyArray_DATA((PyArrayObject *)result);
  memset(result_data, 0, n_nodes * sizeof(float));

  for (Py_ssize_t i_node = 0; i_node < n_nodes; i_node++) {
    Node node = nodes_list[i_node];
    for (Py_ssize_t i_trace = 0; i_trace < n_traces; i_trace++) {
      float weight = node.weights[i_trace];
      Trace trace = traces_list[i_trace];
      int32_t trace_shift = trace.offset + node.shifts[i_trace];
      int32_t base_idx = trace_shift - min_shift;
      int32_t trace_sample = index - base_idx;
      if (0 <= trace_sample && trace_sample < trace.size) {
        result_data[i_node] += trace.data[trace_sample] * weight;
      }
    }
  }

  free(traces_list);
  free(nodes_list);
  return result;
}

// Method definitions
static PyMethodDef StackTracesMethods[] = {
    {"stack_traces", (PyCFunction)(void (*)(void))stack_traces,
     METH_VARARGS | METH_KEYWORDS, ""},
    {"stack_and_reduce", (PyCFunction)(void (*)(void))stack_and_reduce,
     METH_VARARGS | METH_KEYWORDS, ""},
    {"stack_snapshot", (PyCFunction)(void (*)(void))stack_snapshot,
     METH_VARARGS | METH_KEYWORDS, ""},
    {NULL, NULL, 0, NULL}};

// Module definition
static PyModuleDef stack = {PyModuleDef_HEAD_INIT, "stack", NULL, -1,
                            StackTracesMethods};

// Module initialization
PyMODINIT_FUNC PyInit_stack(void) {
  import_array(); // Initialize NumPy
  return PyModule_Create(&stack);
}
