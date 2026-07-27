#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_math.h>
#include <omp.h>
#include <stdint.h>
#include <string.h>

#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>

#define LANE_WIDTH 8

// Structure definitions equivalent to Mojo's structs
typedef struct {
  int32_t *shifts;
  float *weights;
  npy_bool masked;
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

static inline npy_intp imax(npy_intp a, npy_intp b) { return a > b ? a : b; }
static inline npy_intp imin(npy_intp a, npy_intp b) { return a < b ? a : b; }

// Function to check NumPy array dtype, dimensionality and contiguity.
//
// Callers no longer copy/convert inputs via PyArray_ContiguousFromObject
// (that hid dtype/shape mismatches behind a silent copy, and freeing that
// copy's reference immediately after use left dangling data pointers once
// the copy's refcount hit zero). Arrays are used as-is, so this check is
// the only guard against wrong dtype, wrong rank, or non-contiguous input.
static inline int check_array_dtype(PyArrayObject *arr, int expected_type,
                                    int expected_ndim) {
  if (!PyArray_Check(arr)) {
    PyErr_SetString(PyExc_TypeError, "Input must be a NumPy array");
    return 0;
  }
  if (PyArray_TYPE(arr) != expected_type) {
    const char *type_name = expected_type == NPY_FLOAT32 ? "float32"
                            : expected_type == NPY_INT32 ? "int32"
                            : expected_type == NPY_BOOL  ? "bool"
                                                         : "unknown";
    PyErr_Format(PyExc_TypeError, "Input array must be of type %s", type_name);
    return 0;
  }
  if (PyArray_NDIM(arr) != expected_ndim) {
    PyErr_Format(PyExc_ValueError,
                 "Input array must have %d dimension(s), got %d", expected_ndim,
                 PyArray_NDIM(arr));
    return 0;
  }
  if (!PyArray_ISCONTIGUOUS(arr)) {
    PyErr_SetString(PyExc_ValueError, "Input array must be C-contiguous");
    return 0;
  }
  return 1;
}

// Prepare function equivalent to Mojo's prepare
//
// Inputs are used as-is (no PyArray_ContiguousFromObject conversion): the
// caller must already pass C-contiguous arrays of the exact dtype checked
// below. This is enforced by check_array_dtype() rather than silently
// converted, since a converted copy's data pointer would otherwise be kept
// around after the copy's only reference is released -- see the note on
// check_array_dtype() above.
static PyObject *prepare(PyObject *traces, PyObject *offsets, PyObject *shifts,
                         PyObject *weights, PyObject *node_mask,
                         Trace **traces_list, Node **nodes_list,
                         int32_t *min_shift, int32_t *max_shift) {
  Py_ssize_t n_traces = PyList_Size(traces);
  PyArrayObject *shifts_arr = (PyArrayObject *)shifts;
  PyArrayObject *weights_arr = (PyArrayObject *)weights;
  PyArrayObject *offsets_arr = (PyArrayObject *)offsets;
  PyArrayObject *node_mask_arr = NULL;
  int node_mask_owned = 0;

  if (n_traces == 0) {
    PyErr_SetString(PyExc_ValueError, "Input traces must be a non-empty list");
    return NULL;
  }

  if (!check_array_dtype(shifts_arr, NPY_INT32, 2) ||
      !check_array_dtype(weights_arr, NPY_FLOAT32, 2) ||
      !check_array_dtype(offsets_arr, NPY_INT32, 1)) {
    return NULL;
  }

  npy_intp *shifts_shape = PyArray_SHAPE(shifts_arr);
  npy_intp n_nodes = shifts_shape[0];

  if (node_mask == Py_None) {
    node_mask = PyArray_ZEROS(1, &n_nodes, NPY_BOOL, 0);
    if (!node_mask) {
      PyErr_SetString(PyExc_MemoryError, "Failed to allocate node activation");
      return NULL;
    }
    node_mask_owned = 1; // We own this reference and must release it below.
  } else if (!check_array_dtype((PyArrayObject *)node_mask, NPY_BOOL, 1)) {
    return NULL;
  }
  node_mask_arr = (PyArrayObject *)node_mask;

  if (n_nodes == 0) {
    PyErr_SetString(PyExc_ValueError,
                    "Number of nodes must be greater than zero");
    goto cleanup_mask;
  }

  if (shifts_shape[0] != PyArray_SHAPE(weights_arr)[0] ||
      shifts_shape[1] != PyArray_SHAPE(weights_arr)[1]) {
    PyErr_SetString(PyExc_ValueError,
                    "Shifts and weights must have the same shape");
    goto cleanup_mask;
  }
  if (n_traces != PyArray_SHAPE(offsets_arr)[0]) {
    PyErr_SetString(PyExc_ValueError,
                    "Number of arrays must match number of offsets");
    goto cleanup_mask;
  }
  if (shifts_shape[1] != n_traces) {
    PyErr_SetString(PyExc_ValueError,
                    "Shifts must have the same number of columns as traces");
    goto cleanup_mask;
  }
  if (n_nodes != PyArray_SHAPE(node_mask_arr)[0]) {
    PyErr_SetString(PyExc_ValueError,
                    "Number of nodes must match number of activation flags");
    goto cleanup_mask;
  }

  int32_t *offsets_data = (int32_t *)PyArray_DATA(offsets_arr);
  int32_t *shifts_data = (int32_t *)PyArray_DATA(shifts_arr);
  float *weights_data = (float *)PyArray_DATA(weights_arr);
  npy_bool *node_mask_data = (npy_bool *)PyArray_DATA(node_mask_arr);

  *traces_list = (Trace *)malloc(n_traces * sizeof(Trace));
  *nodes_list = (Node *)malloc(n_nodes * sizeof(Node));
  if (!*traces_list || !*nodes_list) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
    goto cleanup_traces;
  }

  for (npy_intp i = 0; i < n_traces; i++) {
    PyArrayObject *trace = (PyArrayObject *)PyList_GetItem(traces, i);
    if (!check_array_dtype(trace, NPY_FLOAT32, 1)) {
      goto cleanup_traces;
    }
    (*traces_list)[i].data = (float *)PyArray_DATA(trace);
    (*traces_list)[i].size = PyArray_SIZE(trace);
    (*traces_list)[i].offset = offsets_data[i];
  }

  for (npy_intp i = 0; i < n_nodes; i++) {
    (*nodes_list)[i].shifts = shifts_data + i * n_traces;
    (*nodes_list)[i].weights = weights_data + i * n_traces;
    (*nodes_list)[i].masked = node_mask_data[i];
  }

  *min_shift = INT32_MAX;
  *max_shift = INT32_MIN;
  for (npy_intp i = 0; i < n_nodes; i++) {
    Node node = (*nodes_list)[i];
    for (npy_intp j = 0; j < n_traces; j++) {
      int32_t idx_begin = (*traces_list)[j].offset + node.shifts[j];
      int32_t idx_end = idx_begin + (*traces_list)[j].size;
      *min_shift = (*min_shift < idx_begin) ? *min_shift : idx_begin;
      *max_shift = (*max_shift > idx_end) ? *max_shift : idx_end;
    }
  }

  if (node_mask_owned) {
    Py_DECREF(node_mask);
  }
  return traces;

cleanup_traces:
  free(*traces_list);
  free(*nodes_list);
cleanup_mask:
  if (node_mask_owned) {
    Py_DECREF(node_mask);
  }
  return NULL;
}

static PyObject *delay_sum(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyObject *traces, *offsets, *shifts, *weights, *stack, *node_mask,
      *shift_range;
  stack = Py_None; // Default to None if not provided
  node_mask = Py_None;
  shift_range = Py_None;
  int n_threads = 1;

  static char *kwlist[] = {"traces",      "offsets",   "shifts",
                           "weights",     "node_mask", "stack",
                           "shift_range", "n_threads", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|OOOi", kwlist, &traces,
                                   &offsets, &shifts, &weights, &node_mask,
                                   &stack, &shift_range, &n_threads)) {
    return NULL;
  }

  Trace *traces_list;
  Node *nodes_list;
  int32_t min_shift, max_shift;
  if (!prepare(traces, offsets, shifts, weights, node_mask, &traces_list,
               &nodes_list, &min_shift, &max_shift)) {
    return NULL;
  }

  npy_intp n_traces = PyList_Size(traces);
  npy_intp n_nodes = PyArray_SHAPE((PyArrayObject *)shifts)[0];
  npy_intp stack_size = max_shift - min_shift;
  if (shift_range != Py_None) {
    if (!PyTuple_Check(shift_range) || PyTuple_Size(shift_range) != 2 ||
        !PyLong_Check(PyTuple_GetItem(shift_range, 0)) ||
        !PyLong_Check(PyTuple_GetItem(shift_range, 1))) {
      PyErr_SetString(
          PyExc_ValueError,
          "shift_range argument must be tuple of two integers or None.");
      free(traces_list);
      free(nodes_list);
      return NULL;
    }
    min_shift = (int32_t)PyLong_AsLong(PyTuple_GetItem(shift_range, 0));
    max_shift = (int32_t)PyLong_AsLong(PyTuple_GetItem(shift_range, 1));
    stack_size = max_shift - min_shift;

    if (max_shift <= min_shift) {
      PyErr_SetString(PyExc_ValueError,
                      "Invalid shift_range: max_shift must be greater than "
                      "min_shift.");
      free(traces_list);
      free(nodes_list);
      return NULL;
    }
  }

  if (stack != Py_None) {
    if (!check_array_dtype((PyArrayObject *)stack, NPY_FLOAT32, 2)) {
      free(traces_list);
      free(nodes_list);
      return NULL;
    }
    npy_intp *shape = PyArray_SHAPE((PyArrayObject *)stack);
    if (shape[0] != n_nodes || shape[1] != stack_size) {
      PyErr_SetString(
          PyExc_ValueError,
          "Resulting stack array must have shape (n_nodes, stack_size)");
      free(traces_list);
      free(nodes_list);
      return NULL;
    }
  } else {
    npy_intp dims[2] = {n_nodes, stack_size};
    stack = PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    if (!stack) {
      free(traces_list);
      free(nodes_list);
      return NULL;
    }
  }

  float *stack_data = (float *)PyArray_DATA((PyArrayObject *)stack);
  NodeWithStack *node_stacks =
      (NodeWithStack *)malloc(n_nodes * sizeof(NodeWithStack));
  if (!node_stacks) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate node_list");
    Py_DECREF(stack);
    free(traces_list);
    free(nodes_list);
    return NULL;
  }

  Py_BEGIN_ALLOW_THREADS;
  for (npy_intp i = 0; i < n_nodes; i++) {
    node_stacks[i].shifts = nodes_list[i].shifts;
    node_stacks[i].weights = nodes_list[i].weights;
    node_stacks[i].stack = stack_data + i * stack_size;
  }

#pragma omp parallel for num_threads(get_thread_count(n_threads))
  for (npy_intp i_node = 0; i_node < n_nodes; i_node++) {
    NodeWithStack node = node_stacks[i_node];
    for (npy_intp i_trace = 0; i_trace < n_traces; i_trace++) {
      float weight = node.weights[i_trace];
      if (weight == 0.0f)
        continue;

      Trace trace = traces_list[i_trace];
      int32_t trace_shift = trace.offset + node.shifts[i_trace];
      int32_t base_idx = trace_shift - min_shift;
      npy_intp stack_nsamples = imin(stack_size - base_idx, trace.size);

      npy_intp i;
      simde__m256 weight_vec = simde_mm256_set1_ps(weight);

      for (i = imax(0, min_shift - trace_shift);
           i < stack_nsamples - (stack_nsamples % LANE_WIDTH);
           i += LANE_WIDTH) {
        npy_intp i_res = base_idx + i;
        simde__m256 trace_vec = simde_mm256_loadu_ps(&trace.data[i]);
        simde__m256 stack_vec = simde_mm256_loadu_ps(&node.stack[i_res]);
        stack_vec = simde_mm256_fmadd_ps(trace_vec, weight_vec, stack_vec);
        simde_mm256_storeu_ps(&node.stack[i_res], stack_vec);
      }
      for (; i < stack_nsamples; i++) {
        npy_intp i_res = base_idx + i;
        node.stack[i_res] += trace.data[i] * weight;
      }
    }
  }

  free(node_stacks);
  free(traces_list);
  free(nodes_list);
  Py_END_ALLOW_THREADS;

  return Py_BuildValue("Oi", stack, min_shift);
}

// stack_and_reduce function
static PyObject *delay_sum_reduce(PyObject *self, PyObject *args,
                                  PyObject *kwargs) {
  PyObject *traces, *offsets, *shifts, *weights, *node_mask, *node_stack_max,
      *node_stack_max_idx, *shift_range;
  node_mask = Py_None;
  node_stack_max = Py_None;
  node_stack_max_idx = Py_None;
  shift_range = Py_None;

  int n_threads = 1;

  static char *kwlist[] = {
      "traces",    "offsets",     "shifts",         "weights",
      "node_mask", "shift_range", "node_stack_max", "node_stack_max_idx",
      "n_threads", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOO|OOOOi", kwlist, &traces,
                                   &offsets, &shifts, &weights, &node_mask,
                                   &shift_range, &node_stack_max,
                                   &node_stack_max_idx, &n_threads)) {
    return NULL;
  }

  if ((node_stack_max != Py_None && node_stack_max_idx == Py_None) ||
      (node_stack_max == Py_None && node_stack_max_idx != Py_None)) {
    PyErr_SetString(PyExc_ValueError, "node_stack_max and node_stack_max_idx "
                                      "must be both provided or both None");
    return NULL;
  }

  Trace *traces_list;
  Node *nodes_list;
  int32_t min_shift, max_shift;
  if (!prepare(traces, offsets, shifts, weights, node_mask, &traces_list,
               &nodes_list, &min_shift, &max_shift))
    return NULL;

  npy_intp n_traces = PyList_Size(traces);
  npy_intp n_nodes = PyArray_SHAPE((PyArrayObject *)shifts)[0];
  npy_intp stack_size = max_shift - min_shift;

  if (shift_range != Py_None) {
    if (!PyTuple_Check(shift_range) || PyTuple_Size(shift_range) != 2 ||
        !PyLong_Check(PyTuple_GetItem(shift_range, 0)) ||
        !PyLong_Check(PyTuple_GetItem(shift_range, 1))) {
      PyErr_SetString(
          PyExc_ValueError,
          "shift_range argument must be tuple of two integers or None.");
      free(traces_list);
      free(nodes_list);
      return NULL;
    }
    min_shift = (int32_t)PyLong_AsLong(PyTuple_GetItem(shift_range, 0));
    max_shift = (int32_t)PyLong_AsLong(PyTuple_GetItem(shift_range, 1));
    stack_size = max_shift - min_shift;

    if (max_shift <= min_shift) {
      PyErr_SetString(PyExc_ValueError,
                      "Invalid shift_range: max_shift must be greater than "
                      "min_shift.");
      free(traces_list);
      free(nodes_list);
      return NULL;
    }
  }

  if (node_stack_max != Py_None && node_stack_max_idx != Py_None) {
    if (!check_array_dtype((PyArrayObject *)node_stack_max, NPY_FLOAT32, 1) ||
        !check_array_dtype((PyArrayObject *)node_stack_max_idx, NPY_INT32, 1) ||
        PyArray_SHAPE((PyArrayObject *)node_stack_max_idx)[0] != stack_size ||
        PyArray_SHAPE((PyArrayObject *)node_stack_max)[0] != stack_size) {
      PyErr_SetString(
          PyExc_TypeError,
          "Provided result arrays must be 1D NumPy arrays of float32 "
          "and int respectively, with correct length");
      free(traces_list);
      free(nodes_list);
      return NULL;
    }
  } else {
    node_stack_max = PyArray_SimpleNew(1, &stack_size, NPY_FLOAT32);
    node_stack_max_idx = PyArray_ZEROS(1, &stack_size, NPY_INT32, 0);

    if (!node_stack_max || !node_stack_max_idx) {
      Py_XDECREF(node_stack_max);
      Py_XDECREF(node_stack_max_idx);
      free(traces_list);
      free(nodes_list);
      PyErr_SetString(PyExc_MemoryError,
                      "Failed to allocate memory for arrays");
      return NULL;
    }
    float *stack_max_data =
        (float *)PyArray_DATA((PyArrayObject *)node_stack_max);
    for (npy_intp i = 0; i < stack_size; i++) {
      stack_max_data[i] = -NPY_INFINITYF;
    }
  }

  float *stack_max_data =
      (float *)PyArray_DATA((PyArrayObject *)node_stack_max);
  int32_t *stack_max_idx_data =
      (int32_t *)PyArray_DATA((PyArrayObject *)node_stack_max_idx);

  Py_BEGIN_ALLOW_THREADS;
#pragma omp parallel num_threads(get_thread_count(n_threads))                  \
    shared(stack_max_data, stack_max_idx_data)
  {
    int num_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();

    npy_intp chunk_size = stack_size / num_threads;
    npy_intp remainder = stack_size % num_threads;

    npy_intp tile_start_idx =
        thread_id * chunk_size + imin(thread_id, remainder);
    npy_intp tile_end_idx =
        tile_start_idx + chunk_size + (thread_id < remainder ? 1 : 0);
    npy_intp tile_size = tile_end_idx - tile_start_idx;

    float *tile_node_stack = (float *)malloc(tile_size * sizeof(float));

    for (npy_intp i_node = 0; i_node < n_nodes; i_node++) {
      Node node = nodes_list[i_node];
      if (node.masked)
        continue;

      for (npy_intp i = 0; i < tile_size; i++) {
        tile_node_stack[i] = 0.0f;
      }

      for (npy_intp i_trace = 0; i_trace < n_traces; i_trace++) {
        float weight = node.weights[i_trace];
        if (weight == 0.0f)
          continue;
        Trace trace = traces_list[i_trace];
        int32_t trace_shift = trace.offset + node.shifts[i_trace];
        int32_t base_idx = trace_shift - min_shift;
        npy_intp tile_base_idx = imax(0, base_idx - tile_start_idx);
        npy_intp trace_start_idx = imax(0, tile_start_idx - base_idx);
        npy_intp trace_end_idx = imax(0, tile_end_idx - base_idx);

        trace_start_idx = imin(trace_start_idx, trace.size);
        trace_end_idx = imin(trace_end_idx, trace.size);
        npy_intp n_samples = trace_end_idx - trace_start_idx;

        simde__m256 weight_vec = simde_mm256_set1_ps(weight);

        npy_intp i = 0;
        for (; i < n_samples - (n_samples % LANE_WIDTH); i += LANE_WIDTH) {
          npy_intp i_res = tile_base_idx + i;
          simde__m256 trace_vec =
              simde_mm256_loadu_ps(&trace.data[trace_start_idx + i]);
          simde__m256 stack_vec = simde_mm256_loadu_ps(&tile_node_stack[i_res]);
          stack_vec = simde_mm256_fmadd_ps(trace_vec, weight_vec, stack_vec);
          simde_mm256_storeu_ps(&tile_node_stack[i_res], stack_vec);
        }
        for (; i < n_samples; i++) {
          npy_intp i_res = tile_base_idx + i;
          tile_node_stack[i_res] += trace.data[trace_start_idx + i] * weight;
        }
      }

      npy_intp i = 0;

      // simde__m256i node_vec = simde_mm256_set1_epi32((int32_t)i_node);

      // for (; i < tile_size - (tile_size % LANE_WIDTH); i += LANE_WIDTH) {
      //   npy_intp res_idx = tile_start_idx + i;
      //   simde__m256 stack_vec = simde_mm256_loadu_ps(&tile_node_stack[i]);
      //   simde__m256 max_vec = simde_mm256_loadu_ps(&stack_max_data[res_idx]);
      //   simde__m256i max_mask = (simde__m256i)simde_mm256_cmp_ps(
      //       stack_vec, max_vec, SIMDE_CMP_GT_OQ);
      //   simde_mm256_maskstore_ps(&stack_max_data[res_idx], max_mask,
      //   stack_vec); simde_mm256_maskstore_epi32(&stack_max_idx_data[res_idx],
      //   max_mask,
      //                               node_vec);
      // }
      for (; i < tile_size; i++) {
        npy_intp res_idx = tile_start_idx + i;

        if (tile_node_stack[i] > stack_max_data[res_idx]) {
          stack_max_data[res_idx] = tile_node_stack[i];
          stack_max_idx_data[res_idx] = (int32_t)i_node;
        }
      }
    }
    free(tile_node_stack);
  }

  free(traces_list);
  free(nodes_list);
  Py_END_ALLOW_THREADS;

  return Py_BuildValue("OOi", node_stack_max, node_stack_max_idx, min_shift);
}

// stack_snapshot function
static PyObject *delay_sum_snapshot(PyObject *self, PyObject *args,
                                    PyObject *kwargs) {
  PyObject *traces, *offsets, *shifts, *weights, *node_mask, *shift_range;
  node_mask = Py_None;
  shift_range = Py_None;
  int32_t index;

  static char *kwlist[] = {"traces", "offsets",     "shifts",    "weights",
                           "index",  "shift_range", "node_mask", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOOi|OO", kwlist, &traces,
                                   &offsets, &shifts, &weights, &index,
                                   &shift_range, &node_mask)) {
    return NULL;
  }

  Trace *traces_list;
  Node *nodes_list;
  int32_t min_shift, max_shift;
  if (!prepare(traces, offsets, shifts, weights, node_mask, &traces_list,
               &nodes_list, &min_shift, &max_shift)) {
    return NULL;
  }

  npy_intp n_traces = PyList_Size(traces);
  npy_intp n_nodes = PyArray_SHAPE((PyArrayObject *)shifts)[0];
  npy_intp stack_size = max_shift - min_shift;

  if (shift_range != Py_None) {
    if (!PyTuple_Check(shift_range) || PyTuple_Size(shift_range) != 2 ||
        !PyLong_Check(PyTuple_GetItem(shift_range, 0)) ||
        !PyLong_Check(PyTuple_GetItem(shift_range, 1))) {
      PyErr_SetString(
          PyExc_ValueError,
          "shift_range argument must be tuple of two integers or None.");
      free(traces_list);
      free(nodes_list);
      return NULL;
    }
    min_shift = (int32_t)PyLong_AsLong(PyTuple_GetItem(shift_range, 0));
    max_shift = (int32_t)PyLong_AsLong(PyTuple_GetItem(shift_range, 1));
    stack_size = max_shift - min_shift;
  }

  if (index >= stack_size || index < 0) {
    PyErr_Format(PyExc_ValueError, "Snapshot index out of bounds: %d", index);
    free(traces_list);
    free(nodes_list);
    return NULL;
  }

  PyObject *snapshot = PyArray_ZEROS(1, &n_nodes, NPY_FLOAT32, 0);
  if (!snapshot) {
    free(traces_list);
    free(nodes_list);
    return NULL;
  }

  float *snapshot_data = (float *)PyArray_DATA((PyArrayObject *)snapshot);

  Py_BEGIN_ALLOW_THREADS;
  for (npy_intp i_node = 0; i_node < n_nodes; i_node++) {
    Node node = nodes_list[i_node];
    if (node.masked)
      continue;
    for (npy_intp i_trace = 0; i_trace < n_traces; i_trace++) {
      Trace trace = traces_list[i_trace];
      int32_t trace_shift = trace.offset + node.shifts[i_trace];
      int32_t base_idx = trace_shift - min_shift;
      int32_t trace_sample = index - base_idx;
      float weight = node.weights[i_trace];
      if (weight != 0.0 && 0 <= trace_sample && trace_sample < trace.size) {
        snapshot_data[i_node] += trace.data[trace_sample] * weight;
      }
    }
  }

  free(traces_list);
  free(nodes_list);
  Py_END_ALLOW_THREADS;
  return snapshot;
}

// Method definitions
static PyMethodDef DelaySumMethods[] = {
    {"delay_sum", (PyCFunction)(void (*)(void))delay_sum,
     METH_VARARGS | METH_KEYWORDS, ""},
    {"delay_sum_reduce", (PyCFunction)(void (*)(void))delay_sum_reduce,
     METH_VARARGS | METH_KEYWORDS, ""},
    {"delay_sum_snapshot", (PyCFunction)(void (*)(void))delay_sum_snapshot,
     METH_VARARGS | METH_KEYWORDS, ""},
    {NULL, NULL, 0, NULL}};

// Module definition
static PyModuleDef delay_sum_module = {PyModuleDef_HEAD_INIT, "delay_sum", NULL,
                                       -1, DelaySumMethods};

// Module initialization
PyMODINIT_FUNC PyInit_delay_sum(void) {
  import_array(); // Initialize NumPy
  return PyModule_Create(&delay_sum_module);
}
