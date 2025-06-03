from python import PythonObject, Python
from python.bindings import PythonModuleBuilder
from python.python import CPython
from sys.info import simdwidthof
from os import abort
from algorithm.functional import vectorize
from memory.unsafe_pointer import UnsafePointer
from sys.intrinsics import prefetch, PrefetchOptions

from layout import (
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    Layout,
    IntTuple,
    UNKNOWN_VALUE,
)


@value
struct Node[dtype: DType]:
    var shifts: UnsafePointer[Int32]
    var weights: UnsafePointer[Scalar[dtype]]
    var result: UnsafePointer[Scalar[dtype]]


@value
struct Trace[dtype: DType]:
    var data: UnsafePointer[Scalar[dtype]]
    var size: Int
    var offset: Int

    fn __init__(out self, array: PythonObject, offset: Int) raises:
        if array.ndim != 1:
            raise "Each trace must be a 1D array"
        try:
            check_array_dtype[dtype](array)
        except:
            raise "Input trace is of wrong dtype"

        self.data = array.ctypes.data.unsafe_get_as_pointer[dtype]()
        self.size = Int(array.size)
        self.offset = offset


fn get_shift_ranges[
    dtype: DType
](nodes: List[Node[dtype]], traces: List[Trace[dtype]]) -> (Int32, Int32):
    min_idx = Int32.MAX
    max_idx = Int32.MIN

    for i_node in range(len(nodes)):
        node = nodes[i_node]
        for i_trace in range(len(traces)):
            idx_begin = traces[i_trace].offset + node.shifts[i_trace]
            idx_end = idx_begin + traces[i_trace].size
            min_idx = min(min_idx, idx_begin)
            max_idx = max(idx_end, max_idx)

    return (Int32(min_idx), Int32(max_idx))


fn get_dtype_char[dtype: DType]() raises -> String:
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


fn check_array_dtype[dtype: DType](numpy_array: PythonObject) raises:
    @parameter
    char = get_dtype_char[dtype]()
    if numpy_array.dtype.char != char:
        raise "Input array must be of type " + char



@export
fn PyInit_parstack() -> PythonObject:
    try:
        var m = PythonModuleBuilder("stack_traces")
        m.def_function[stack_traces_wrapper](
            "stack_traces",
            docstring="",
        )
        return m.finalize()
    except e:
        return abort[PythonObject](
            String("error creating Python Mojo module:", e)
        )



fn stack_traces_wrapper(
    traces: PythonObject,
    offsets: PythonObject,
    shifts: PythonObject,
    weights: PythonObject,
    result: PythonObject,
    length_out: PythonObject,
    ) raises -> PythonObject:
    np = Python.import_module("numpy")

    n_traces = len(traces)
    if n_traces == 0:
        raise "Input traces must have positive dimensions"

    trace = traces[0]

    if trace.dtype == np.float16:
        stack_func = stack_traces[DType.float16]
    elif trace.dtype == np.float32:
        stack_func = stack_traces[DType.float32]
    elif trace.dtype == np.float64:
        stack_func = stack_traces[DType.float64]
    else:
        raise "Unsupported dtype: " + String(trace.dtype)
    return stack_func(
        traces, offsets, shifts, weights, result, length_out
    )


fn stack_traces[
    dtype: DType
](
    traces: PythonObject,
    offsets: PythonObject,
    shifts: PythonObject,
    weights: PythonObject,
    result: PythonObject,
    length_out: PythonObject,
) raises -> PythonObject:
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

    if n_traces != Int(offsets.shape[0]):
        raise "Number of arrays must match number of offsets"
    if Int(shifts.shape[1]) != n_traces:
        raise "Shifts must have the same number of columns as traces"
    if shifts.shape != weights.shape:
        raise "Shifts and weights must have the same shape"

    offsets_data = offsets.ctypes.data.unsafe_get_as_pointer[DType.int32]()
    shifts_data = shifts.ctypes.data.unsafe_get_as_pointer[DType.int32]()
    weights_data = weights.ctypes.data.unsafe_get_as_pointer[dtype]()

    traces_list = List[Trace[dtype]]()
    node_list = List[Node[dtype]]()

    for i_trace in range(n_traces):
        traces_list.append(
            Trace[dtype](
                array=traces[i_trace],
                offset=Int(offsets_data[i_trace]),
            )
        )

    dummy_offset = UnsafePointer[Scalar[dtype]].alloc(1)
    for i_node in range(n_nodes):
        node_list.append(
            Node(
                shifts=shifts_data + i_node * n_traces,
                weights=weights_data + i_node * n_traces,
                result=dummy_offset,
            )
        )

    min_shift, max_shift = get_shift_ranges(node_list, traces_list)
    result_length = Int(length_out) if length_out else max_shift - min_shift
    if result is None:
        result_arr = np.zeros(
            shape=Python.tuple(n_nodes, result_length), dtype=get_dtype_char[dtype](),
        )
    else:
        if result.shape is not Python.tuple(n_nodes, result_length):
            raise "Result array must have shape (n_nodes, length_out)"
        check_array_dtype[dtype](result)
        result_arr = result

    result_data = result_arr.ctypes.data.unsafe_get_as_pointer[dtype]()

    for i_node in range(len(node_list)):
        node_list[i_node].result = result_data + i_node * result_length

    for i_node in range(n_nodes):
        node = node_list[i_node]
        # prefetch[PrefetchOptions().high_locality()](node.result)

        for i_trace in range(n_traces):
            weight = node.weights[i_trace]
            if weight == 0.0:
                continue
            trace = traces_list[i_trace]
            trace_shift = trace.offset + node.shifts[i_trace]
            base_idx = trace_shift - min_shift

            @parameter
            fn stack[width: Int](i_sample: Int):
                i_res = base_idx + i_sample
                trace_samples = trace.data.load[width=width](i_sample)
                stacked_samples = node.result.load[width=width](i_res)

                stacked_samples += trace_samples * weight
                node.result.store(i_res, stacked_samples)

            stack_nsamples = min(result_length - base_idx, trace.size)

            vectorize[stack, simdwidthof[dtype]()](Int(stack_nsamples))

    return Python.tuple(result_arr, min_shift)
