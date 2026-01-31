from python import PythonObject, Python
from python.bindings import PythonModuleBuilder
from python.python import CPython
from sys.info import simd_width_of
from os import abort
from algorithm.functional import vectorize, parallelize
from memory import LegacyUnsafePointer as UnsafePointer
from memory import memset
from sys.intrinsics import masked_store, prefetch, PrefetchOptions
from sys import num_logical_cores

from layout import (
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    Layout,
    IntTuple,
    UNKNOWN_VALUE,
)


@export
fn PyInit_stack_ext() -> PythonObject:
    try:
        var m = PythonModuleBuilder("stack_traces")
        m.def_function[stack_wrapper](
            "stack_traces",
            docstring="",
        )
        m.def_function[stack_and_reduce_wrapper](
            "stack_and_reduce",
            docstring="",
        )
        m.def_function[stack_snapshot_wrapper](
            "stack_snapshot",
            docstring="",
        )
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module:", e))


@fieldwise_init
struct Node[dtype: DType](ImplicitlyCopyable & Movable):
    var shifts: UnsafePointer[Int32]
    var weights: UnsafePointer[Scalar[Self.dtype]]


@fieldwise_init
struct NodeWithStack[dtype: DType](ImplicitlyCopyable & Movable):
    var shifts: UnsafePointer[Int32]
    var weights: UnsafePointer[Scalar[Self.dtype]]
    var stack: UnsafePointer[Scalar[Self.dtype]]

    def __init__(
        out self, node: Node[Self.dtype], stack: UnsafePointer[Scalar[Self.dtype]]
    ):
        self.shifts = node.shifts
        self.weights = node.weights
        self.stack = stack


@fieldwise_init
struct Trace[dtype: DType](ImplicitlyCopyable & Movable):
    var data: UnsafePointer[Scalar[Self.dtype]]
    var size: Int
    var offset: Int

    fn __init__(out self, array: PythonObject, offset: Int) raises:
        if Int(array.ndim) != 1:
            raise "Each trace must be a 1D array"
        try:
            check_array_dtype[Self.dtype](array)
        except:
            raise "Input trace is of wrong dtype"

        self.data = array.ctypes.data.unsafe_get_as_pointer[Self.dtype]()
        self.size = Int(array.size)
        self.offset = offset


@always_inline
fn get_thread_count(n_threads: Int) -> Int:
    if n_threads <= 0:
        return num_logical_cores()
    return n_threads


@always_inline
fn get_dtype_char[dtype: DType]() raises -> String:
    @parameter
    if dtype is DType.float16:
        return "e"
    elif dtype is DType.float32:
        return "f"
    elif dtype is DType.float64:
        return "d"
    elif dtype is DType.int32:
        return "i"
    else:
        raise "Unsupported dtype"


@always_inline
fn check_array_dtype[dtype: DType](numpy_array: PythonObject) raises:
    dtype_char = get_dtype_char[dtype]()
    if String(numpy_array.dtype.char) != dtype_char:
        raise Error("Input array must be of type " + String(dtype_char))


fn stack_wrapper(
    traces: PythonObject,
    offsets: PythonObject,
    shifts: PythonObject,
    weights: PythonObject,
    result: PythonObject,
    n_threads: PythonObject,
) raises -> PythonObject:
    # np = Python.import_module("numpy")

    # n_traces = len(traces)
    # if n_traces == 0:
    #     raise "Input traces must have positive dimensions"
    # trace = traces[0]

    # if trace.dtype == np.float16:
    #     prepare_func = prepare[DType.float16]
    # elif trace.dtype == np.float32:
    #     prepare_func = prepare[DType.float32]
    # elif trace.dtype == np.float64:
    #     prepare_func = prepare[DType.float64]
    # else:
    #     raise "Unsupported dtype: " + String(trace.dtype)

    var out = prepare[DType.float32](
        traces=traces,
        offsets=offsets,
        shifts=shifts,
        weights=weights,
    )
    ref traces_list = out[0]
    ref nodes_list = out[1]
    var min_shift = out[2]
    var max_shift = out[3]

    return stack_traces(
        traces_list,
        nodes_list,
        min_shift,
        max_shift,
        result,
        get_thread_count(Int(n_threads)),
    )


fn stack_and_reduce_wrapper(
    traces: PythonObject,
    offsets: PythonObject,
    shifts: PythonObject,
    weights: PythonObject,
    n_threads: PythonObject,
) raises -> PythonObject:
    var out = prepare[DType.float32](
        traces=traces,
        offsets=offsets,
        shifts=shifts,
        weights=weights,
    )
    ref traces_list = out[0]
    ref nodes_list = out[1]
    var min_shift = out[2]
    var max_shift = out[3]

    return stack_and_reduce(
        traces_list,
        nodes_list,
        min_shift,
        max_shift,
        get_thread_count(Int(n_threads)),
    )


fn stack_snapshot_wrapper(
    traces: PythonObject,
    offsets: PythonObject,
    shifts: PythonObject,
    weights: PythonObject,
    index: PythonObject,
) raises -> PythonObject:
    var out = prepare[DType.float32](
        traces=traces,
        offsets=offsets,
        shifts=shifts,
        weights=weights,
    )
    ref traces_list = out[0]
    ref nodes_list = out[1]
    var min_shift = out[2]
    var max_shift = out[3]

    return stack_snapshot(
        traces_list,
        nodes_list,
        min_shift,
        max_shift,
        Int(index),
    )


fn prepare[
    dtype: DType
](
    traces: PythonObject,
    offsets: PythonObject,
    shifts: PythonObject,
    weights: PythonObject,
) raises -> Tuple[List[Trace[dtype]], List[Node[dtype]], Int32, Int32]:
    np = Python.import_module("numpy")

    n_traces = len(traces)
    n_nodes = Int(shifts.shape[0])

    if n_nodes == 0:
        raise "Input arrays must have positive dimensions"
    if n_traces == 0:
        raise "Input traces must have positive dimensions"

    check_array_dtype[dtype](weights)
    check_array_dtype[DType.int32](offsets)
    check_array_dtype[DType.int32](shifts)

    if shifts.shape != weights.shape:
        raise "Shifts and weights must have the same shape"
    if n_traces != Int(offsets.shape[0]):
        raise "Number of arrays must match number of offsets"
    if Int(shifts.shape[1]) != n_traces:
        raise "Shifts must have the same number of columns as traces"

    var offsets_data = offsets.ctypes.data.unsafe_get_as_pointer[DType.int32]()
    var shifts_data = shifts.ctypes.data.unsafe_get_as_pointer[DType.int32]()
    var weights_data = weights.ctypes.data.unsafe_get_as_pointer[dtype]()

    traces_list = List[Trace[dtype]](capacity=n_traces)
    node_list = List[Node[dtype]](capacity=n_nodes)

    for i_trace in range(n_traces):
        traces_list.append(
            Trace[dtype](
                array=traces[i_trace],
                offset=Int(offsets_data[i_trace]),
            )
        )

    for i_node in range(n_nodes):
        node_list.append(
            Node(
                shifts=shifts_data + i_node * n_traces,
                weights=weights_data + i_node * n_traces,
            )
        )

    min_shift = Int32.MAX
    max_shift = Int32.MIN
    for node in node_list:
        for i_trace in range(n_traces):
            idx_begin = traces_list[i_trace].offset + node.shifts[i_trace]
            idx_end = idx_begin + traces_list[i_trace].size
            min_shift = min(min_shift, idx_begin)
            max_shift = max(idx_end, max_shift)

    return traces_list^, node_list^, min_shift, max_shift


fn stack_traces[
    dtype: DType
](
    read traces: List[Trace[dtype]],
    nodes: List[Node[dtype]],
    min_shift: Int32,
    max_shift: Int32,
    result_arr: PythonObject,
    n_threads: Int = 16,
) raises -> PythonObject:
    cpython = CPython()
    np = Python.import_module("numpy")

    result_length = max_shift - min_shift
    n_nodes = len(nodes)
    n_traces = len(traces)

    result_shape = Python.tuple(n_nodes, result_length)
    if result_arr is PythonObject(None):
        result = np.zeros(
            shape=Python.tuple(n_nodes, result_length),
            dtype=PythonObject(get_dtype_char[dtype]()) ,
        )
    else:
        if result_arr.shape != Python.tuple(n_nodes, result_length):
            raise "Result array must have shape (n_nodes, length_out)"
        check_array_dtype[dtype](result_arr)
        result = result_arr

    result_data = result.ctypes.data.unsafe_get_as_pointer[dtype]()
    node_list = List[NodeWithStack[dtype]](capacity=n_nodes)
    for i_node in range(n_nodes):
        node_list.append(
            NodeWithStack[dtype](
                node=nodes[i_node],
                stack=result_data + i_node * result_length,
            )
        )

    @parameter
    fn stack_node(i_node: Int):
        node = node_list[i_node]
        # prefetch[PrefetchOptions().high_locality()](node.result)

        for i_trace in range(n_traces):
            weight = node.weights[i_trace]
            if weight == 0.0:
                continue
            trace = traces[i_trace]
            trace_shift = trace.offset + node.shifts[i_trace]
            base_idx = trace_shift - min_shift

            @parameter
            fn stack[width: Int](i_sample: Int) unified {mut}:
                i_res = base_idx + i_sample
                trace_samples = trace.data.load[width=width](i_sample)
                stacked_samples = node.stack.load[width=width](i_res)

                stacked_samples += trace_samples * weight
                node.stack.store(i_res, stacked_samples)

            stack_nsamples = min(result_length - base_idx, trace.size)
            vectorize[simd_width_of[dtype]()](Int(stack_nsamples), stack)

    state = cpython.PyGILState_Ensure()
    parallelize[stack_node](n_nodes, n_threads)
    cpython.PyGILState_Release(state)

    return Python.tuple(result, min_shift)


fn stack_and_reduce[
    dtype: DType
](
    traces: List[Trace[dtype]],
    nodes: List[Node[dtype]],
    min_shift: Int32,
    max_shift: Int32,
    n_threads: Int = 16,
) raises -> PythonObject:
    cpython = CPython()
    np = Python.import_module("numpy")
    result_length = max_shift - min_shift
    n_nodes = len(nodes)
    n_traces = len(traces)

    node_max = np.full(
        result_length,
        np.finfo(np.float32).min,
        dtype=PythonObject(get_dtype_char[dtype]()),
    )
    node_argmax = np.zeros(result_length, dtype=np.intp)
    node_max_data = node_max.ctypes.data.unsafe_get_as_pointer[dtype]()
    node_argmax_data = node_argmax.ctypes.data.unsafe_get_as_pointer[DType.uint64]()

    @parameter
    fn stack_tile(i_thread: Int):
        tile_start_idx = i_thread * result_length // n_threads
        tile_end_idx = (i_thread + 1) * result_length // n_threads
        tile_size = Int(tile_end_idx - tile_start_idx)

        tile_node_stack = UnsafePointer[Scalar[dtype]].alloc(tile_size)

        for i_node in range(n_nodes):
            node = nodes[i_node]
            # prefetch[PrefetchOptions().high_locality()](node.node_max)
            memset(tile_node_stack, 0, tile_size)

            for i_trace in range(n_traces):
                weight = node.weights[i_trace]

                if weight == 0.0:
                    continue

                trace = traces[i_trace]
                trace_shift = trace.offset + node.shifts[i_trace]

                base_idx = trace_shift - min_shift
                tile_base_idx = max(0, base_idx - tile_start_idx)

                trace_start_idx = max(0, tile_start_idx - base_idx)
                trace_end_idx = max(0, tile_end_idx - base_idx)

                trace_start_idx = min(trace_start_idx, trace.size)
                trace_end_idx = min(trace_end_idx, trace.size)

                n_samples = trace_end_idx - trace_start_idx

                @parameter
                fn stack[width: Int](i_sample: Int) unified {mut}:
                    i_res = tile_base_idx + i_sample

                    trace_samples = trace.data.load[width=width](
                        trace_start_idx + i_sample
                    )
                    stacked_samples = tile_node_stack.load[width=width](i_res)

                    # trace_samples.fma(weight, stacked_samples)
                    # stacked_samples += trace_samples * weight
                    tile_node_stack.store(
                        i_res, trace_samples.fma(weight, stacked_samples)
                    )

                vectorize[simd_width_of[dtype]()](Int(n_samples), stack)

            @parameter
            fn reduce_max[width: Int](idx: Int) unified {mut}:
                tile_idx = tile_start_idx + idx
                node_vec = tile_node_stack.load[width=width](idx)
                max_old_vec = node_max_data.load[width=width](tile_idx)

                max_new_vec = max(node_vec, max_old_vec)
                update_mask = max_old_vec.ne(max_new_vec)

                node_max_data.store(tile_idx, max_new_vec)
                masked_store(
                    SIMD[DType.uint64, width](i_node),
                    node_argmax_data + tile_idx,
                    mask=update_mask,
                )

            vectorize[simd_width_of[dtype]()](tile_size, reduce_max)

        tile_node_stack.free()

    state = cpython.PyGILState_Ensure()
    parallelize[stack_tile](n_threads, n_threads)
    cpython.PyGILState_Release(state)

    return Python.tuple(node_max, node_argmax, min_shift)


fn stack_snapshot[
    dtype: DType
](
    traces: List[Trace[dtype]],
    nodes: List[Node[dtype]],
    min_shift: Int32,
    max_shift: Int32,
    index: Int32,
) raises -> PythonObject:
    cpython = CPython()
    np = Python.import_module("numpy")

    result_length = max_shift - min_shift
    if index >= result_length or index < 0:
        raise Error("Snapshot index out of bounds: " + String(index))

    n_nodes = len(nodes)
    n_traces = len(traces)

    result = np.zeros(shape=n_nodes, dtype=PythonObject(get_dtype_char[dtype]()))
    result_data = result.ctypes.data.unsafe_get_as_pointer[dtype]()

    state = cpython.PyGILState_Ensure()
    for i_node in range(n_nodes):
        node = nodes[i_node]

        @parameter
        fn stack_traces[width: Int](i_trace: Int) unified {read}:

            trace_samples = SIMD[dtype, width](0)
            trace_weights = SIMD[dtype, width](0)

            @parameter
            for idx_vector in range(width):
                trace_idx = i_trace + idx_vector

                weight = node.weights[trace_idx]
                trace = traces[trace_idx]

                trace_shift = trace.offset + node.shifts[trace_idx]
                base_idx = trace_shift - min_shift
                trace_sample = Int(index - base_idx)

                if 0 <= trace_sample < trace.size:
                    trace_samples[idx_vector] = trace.data[trace_sample]
                    trace_weights[idx_vector] = node.weights[trace_idx]

            result_data[i_node] += (trace_samples * trace_weights).reduce_add()

        vectorize[simd_width_of[dtype]()](Int(n_traces), stack_traces)

    cpython.PyGILState_Release(state)

    return result
