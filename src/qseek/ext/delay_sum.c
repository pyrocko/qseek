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
  int masked;
  int trace_group;
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

// Function to check NumPy array dtype
static inline int check_array(PyObject *arr, int expected_type) {
  if (!PyArray_Check(arr)) {
    PyErr_SetString(PyExc_TypeError, "Input must be a NumPy array");
    return 0;
  }
  if (PyArray_TYPE((PyArrayObject *)arr) != expected_type) {
    PyErr_Format(PyExc_TypeError, "Input array must be of type %s",
                 expected_type == NPY_FLOAT32 ? "float32" : "unknown");
    return 0;
  }
  if (!PyArray_ISCONTIGUOUS((PyArrayObject *)arr)) {
    PyErr_SetString(PyExc_ValueError, "Input array must be contiguous");
    return 0;
  }
  return 1;
}

static int prepare(PyObject *nodes, PyObject *traces, PyObject *offsets,
                   Node **nodes_list, Trace **traces_list, int32_t *min_shift,
                   int32_t *max_shift) {
  if (!PyList_Check(nodes) || !PyList_Check(traces)) {
    PyErr_SetString(PyExc_TypeError, "nodes and traces must be lists");
    return 0;
  }

  Py_ssize_t n_traces = PyList_Size(traces);
  Py_ssize_t n_nodes = PyList_Size(nodes);

  if (!check_array(offsets, NPY_INT32) ||
      PyArray_SHAPE((PyArrayObject *)offsets)[0] != n_traces) {
    PyErr_SetString(PyExc_ValueError,
                    "Number of arrays must match number of offsets");
    return 0;
  }
  int32_t *offsets_data = (int32_t *)PyArray_DATA((PyArrayObject *)offsets);

  *traces_list = (Trace *)malloc(n_traces * sizeof(Trace));
  *nodes_list = (Node *)malloc(n_nodes * sizeof(Node));
  if (!*traces_list || !*nodes_list) {
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate memory");
    return 0;
  }

  for (npy_intp i = 0; i < n_traces; i++) {
    PyObject *trace = PyList_GET_ITEM(traces, i);
    if (!check_array(trace, NPY_FLOAT32)) {
      Py_DECREF(trace);
      free(*traces_list);
      free(*nodes_list);
      return 0;
    }
    (*traces_list)[i].data = (float *)PyArray_DATA((PyArrayObject *)trace);
    (*traces_list)[i].size = PyArray_SIZE((PyArrayObject *)trace);
    (*traces_list)[i].offset = offsets_data[i];
  }

  for (npy_intp i = 0; i < n_nodes; i++) {
    PyObject *node_tuple = PyList_GET_ITEM(nodes, i);
    if (!PyTuple_Check(node_tuple) || PyTuple_Size(node_tuple) < 3) {
      PyErr_SetString(
          PyExc_TypeError,
          "Each node must be a tuple of (shifts, weights, masked, ...)");
      free(*nodes_list);
      free(*traces_list);
      return 0;
    }
    PyObject *shifts_arr = (PyObject *)PyTuple_GET_ITEM(node_tuple, 0);
    PyObject *weights_arr = (PyObject *)PyTuple_GET_ITEM(node_tuple, 1);
    PyObject *masked_obj = (PyObject *)PyTuple_GET_ITEM(node_tuple, 2);
    PyObject *trace_group_obj = (PyObject *)PyTuple_GET_ITEM(node_tuple, 3);

    if (!check_array(shifts_arr, NPY_INT32) ||
        !check_array(weights_arr, NPY_FLOAT32)) {
      free(*nodes_list);
      free(*traces_list);
      return 0;
    }
    if (PyArray_NDIM((PyArrayObject *)shifts_arr) != 1 ||
        PyArray_NDIM((PyArrayObject *)weights_arr) != 1 ||
        PyArray_SIZE((PyArrayObject *)shifts_arr) != n_traces ||
        PyArray_SIZE((PyArrayObject *)weights_arr) != n_traces) {
      PyErr_SetString(PyExc_ValueError, "Shifts and weights must be 1D arrays");
      free(*nodes_list);
      free(*traces_list);
      return 0;
    }

    (*nodes_list)[i].shifts = PyArray_DATA((PyArrayObject *)shifts_arr);
    (*nodes_list)[i].weights = PyArray_DATA((PyArrayObject *)weights_arr);
    (*nodes_list)[i].masked = PyObject_IsTrue(masked_obj);
    // (*nodes_list)[i].trace_group = (int32_t)PyLong_AsLong(trace_group_obj);
    (*nodes_list)[i].trace_group = 0;
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
  return 1;
}

static PyObject *delay_sum(PyObject *self, PyObject *args, PyObject *kwargs) {
  PyObject *traces, *offsets, *nodes, *stack, *shift_range;
  stack = Py_None; // Default to None if not provided
  shift_range = Py_None;
  int n_threads = 1;

  static char *kwlist[] = {"traces",      "offsets",   "nodes", "stack",
                           "shift_range", "n_threads", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|OOi", kwlist, &traces,
                                   &offsets, &nodes, &stack, &shift_range,
                                   &n_threads)) {
    return NULL;
  }

  Trace *traces_list;
  Node *nodes_list;
  int32_t min_shift, max_shift;
  if (!prepare(nodes, traces, offsets, &nodes_list, &traces_list, &min_shift,
               &max_shift)) {
    return NULL;
  }

  npy_intp n_traces = PyList_Size(traces);
  npy_intp n_nodes = PyList_Size(nodes);
  npy_intp stack_size = max_shift - min_shift;
  if (shift_range != Py_None) {
    if (!PyTuple_Check(shift_range) || PyTuple_Size(shift_range) != 2 ||
        !PyLong_Check(PyTuple_GetItem(shift_range, 0)) ||
        !PyLong_Check(PyTuple_GetItem(shift_range, 1))) {
      PyErr_SetString(
          PyExc_ValueError,
          "shift_range argument must be tuple of two integers or None.");
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
    if (!check_array(stack, NPY_FLOAT32)) {
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
  PyObject *traces, *offsets, *nodes, *node_stack_max, *node_stack_max_idx,
      *shift_range;
  node_stack_max = Py_None;
  node_stack_max_idx = Py_None;
  shift_range = Py_None;

  int n_threads = 1;

  static char *kwlist[] = {"traces",         "offsets",
                           "nodes",          "shift_range",
                           "node_stack_max", "node_stack_max_idx",
                           "n_threads",      NULL};
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OOO|OOOi", kwlist, &traces, &offsets, &nodes,
          &shift_range, &node_stack_max, &node_stack_max_idx, &n_threads)) {
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
  if (!prepare(nodes, traces, offsets, &nodes_list, &traces_list, &min_shift,
               &max_shift))
    return NULL;

  npy_intp n_traces = PyList_Size(traces);
  npy_intp n_nodes = PyList_Size(nodes);
  npy_intp stack_size = max_shift - min_shift;

  if (shift_range != Py_None) {
    if (!PyTuple_Check(shift_range) || PyTuple_Size(shift_range) != 2 ||
        !PyLong_Check(PyTuple_GetItem(shift_range, 0)) ||
        !PyLong_Check(PyTuple_GetItem(shift_range, 1))) {
      PyErr_SetString(
          PyExc_ValueError,
          "shift_range argument must be tuple of two integers or None.");
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
    if (!check_array(node_stack_max, NPY_FLOAT32) ||
        !check_array(node_stack_max_idx, NPY_INT32) ||
        PyArray_NDIM((PyArrayObject *)node_stack_max) != 1 ||
        PyArray_NDIM((PyArrayObject *)node_stack_max_idx) != 1 ||
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
          simde__m256 stack_vec = simde_mm256_load_ps(&tile_node_stack[i_res]);
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
      //   simde__m256 max_vec =
      //   simde_mm256_loadu_ps(&stack_max_data[res_idx]); simde__m256i
      //   max_mask = (simde__m256i)simde_mm256_cmp_ps(
      //       stack_vec, max_vec, SIMDE_CMP_GT_OQ);
      //   simde_mm256_maskstore_ps(&stack_max_data[res_idx], max_mask,
      //   stack_vec);
      //   simde_mm256_maskstore_epi32(&stack_max_idx_data[res_idx], max_mask,
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
  PyObject *traces, *offsets, *nodes, *shift_range;
  shift_range = Py_None;
  int32_t index;

  static char *kwlist[] = {"traces", "offsets",     "nodes",
                           "index",  "shift_range", NULL};
  if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOi|O", kwlist, &traces,
                                   &offsets, &nodes, &index, &shift_range)) {
    return NULL;
  }

  Trace *traces_list;
  Node *nodes_list;
  int32_t min_shift, max_shift;
  if (!prepare(nodes, traces, offsets, &nodes_list, &traces_list, &min_shift,
               &max_shift)) {
    return NULL;
  }

  npy_intp n_traces = PyList_Size(traces);
  npy_intp n_nodes = PyList_Size(nodes);
  npy_intp stack_size = max_shift - min_shift;

  if (shift_range != Py_None) {
    if (!PyTuple_Check(shift_range) || PyTuple_Size(shift_range) != 2 ||
        !PyLong_Check(PyTuple_GetItem(shift_range, 0)) ||
        !PyLong_Check(PyTuple_GetItem(shift_range, 1))) {
      PyErr_SetString(
          PyExc_ValueError,
          "shift_range argument must be tuple of two integers or None.");
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
