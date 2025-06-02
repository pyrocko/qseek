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

alias simd_width = simdwidthof[DType.float32]()


@value
struct Node:
    var shifts: UnsafePointer[Int32]
    var weights: UnsafePointer[Float32]
    var result: UnsafePointer[Float32]


@value
struct Trace:
    var data: UnsafePointer[Float32]
    var size: Int
    var offset: Int

    fn __init__(out self, array: PythonObject, offset: Int) raises:
        if array.ndim != 1:
            raise "Each trace must be a 1D array"
        self.data = array.ctypes.data.unsafe_get_as_pointer[DType.float32]()
        self.size = Int(array.size)
        self.offset = offset


fn get_shift_ranges(nodes: List[Node], traces: List[Trace]) -> (Int32, Int32):
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


@export
fn PyInit_parstack() -> PythonObject:
    try:
        var m = PythonModuleBuilder("parstack")
        m.def_function[parstack](
            "parstack",
            docstring="",
        )
        return m.finalize()
    except e:
        return abort[PythonObject](
            String("error creating Python Mojo module:", e)
        )


fn parstack(
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
    if n_traces != Int(offsets.shape[0]):
        raise "Number of arrays must match number of offsets"
    if Int(shifts.shape[1]) != n_traces:
        raise "Shifts must have the same number of columns as traces"
    if shifts.shape != weights.shape:
        raise "Shifts and weights must have the same shape"

    offsets_data = offsets.ctypes.data.unsafe_get_as_pointer[DType.int32]()
    shifts_data = shifts.ctypes.data.unsafe_get_as_pointer[DType.int32]()
    weights_data = weights.ctypes.data.unsafe_get_as_pointer[DType.float32]()

    traces_list = List[Trace]()
    node_list = List[Node]()

    dummy_offset = UnsafePointer[Float32].alloc(1)
    for i_node in range(n_nodes):
        node_list.append(
            Node(
                shifts=shifts_data + i_node * n_traces,
                weights=weights_data + i_node * n_traces,
                result=dummy_offset,
            )
        )

    for i_trace in range(n_traces):
        traces_list.append(
            Trace(
                array=traces[i_trace],
                offset=Int(offsets_data[i_trace]),
            )
        )

    min_shift, max_shift = get_shift_ranges(node_list, traces_list)
    result_length = Int(length_out) if length_out else max_shift - min_shift
    if result is None:
        result_arr = np.zeros(
            shape=Python.tuple(n_nodes, result_length), dtype=np.float32
        )
    else:
        if result.shape is not Python.tuple(n_nodes, result_length):
            raise "Result array must have shape (n_nodes, length_out)"
        if result.dtype != np.float32:
            raise "Result array must be of type float32"
        result_arr = result
    result_data = result_arr.ctypes.data.unsafe_get_as_pointer[DType.float32]()
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
            vectorize[stack, simd_width](Int(stack_nsamples))

    return Python.tuple(result_arr, min_shift)
