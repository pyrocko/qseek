"""Delay-and-sum beamforming, ported from src/qseek/ext/delay_sum.c to Mojo.

Mirrors the C/SIMDE + OpenMP extension function-for-function:
`delay_sum`, `delay_sum_reduce`, and `delay_sum_snapshot` backproject
seismic energy from traces into an irregular grid of nodes using
per-node integer sample shifts and float32 weights.

SIMD lanes come from `vectorize` (compiled against the host's native
vector width) instead of hand-rolled AVX2/SIMDE intrinsics.

Multi-threading (`n_threads`) is accepted for API compatibility with the
C extension but not yet implemented: `std.runtime.asyncrt.TaskGroup`
(the intended replacement for OpenMP `#pragma omp parallel for`) reads
garbage out of `List[...]`-typed arguments on the first loop iteration
of a freshly-scheduled task in this Mojo 1.0.0 toolchain -- confirmed
with a minimal repro outside this file. Everything runs single-threaded
until that is fixed upstream.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.memory.memory import unsafe_memset_zero
from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of, num_logical_cores


@export
def PyInit_delay_sum() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("delay_sum")
        m.def_function[delay_sum](
            "delay_sum",
            docstring="Delay-and-sum beamforming of seismic traces.",
        )
        m.def_function[delay_sum_reduce](
            "delay_sum_reduce",
            docstring="Delay-and-sum beamforming with max-semblance reduction.",
        )
        m.def_function[delay_sum_snapshot](
            "delay_sum_snapshot",
            docstring="Snapshot of delay-and-sum at a single sample index.",
        )
        return m.finalize()
    except e:
        abort(String("error creating Python Mojo module: ", e))


@fieldwise_init
struct Trace(Copyable, Movable):
    var data: Pointer[Float32, MutUntrackedOrigin]
    var size: Int
    var offset: Int


@fieldwise_init
struct Node(Copyable, Movable):
    var shifts: Pointer[Int32, MutUntrackedOrigin]
    var weights: Pointer[Float32, MutUntrackedOrigin]
    var masked: Bool


@always_inline
def get_thread_count(n_threads: Int) -> Int:
    if n_threads <= 0:
        return num_logical_cores()
    return n_threads


@always_inline
def numpy_ptr[
    dtype: DType
](array: PythonObject) raises -> Pointer[Scalar[dtype], MutUntrackedOrigin]:
    var addr = Int(py=array.ctypes.data)
    return Pointer[Scalar[dtype], MutUntrackedOrigin](unsafe_from_address=addr)


@always_inline
def dtype_char[dtype: DType]() raises -> String:
    comptime if dtype == DType.float32:
        return "f"
    elif dtype == DType.int32:
        return "i"
    elif dtype == DType.bool:
        return "?"
    else:
        raise Error("Unsupported dtype")


@always_inline
def check_array_dtype[dtype: DType](array: PythonObject) raises:
    var expected = dtype_char[dtype]()
    var actual = String(array.dtype.char)
    if actual != expected:
        raise Error(
            "Input array must be of type " + expected + ", got " + actual
        )


def prepare(
    traces: PythonObject,
    offsets: PythonObject,
    shifts: PythonObject,
    weights: PythonObject,
    node_mask: PythonObject,
) raises -> Tuple[List[Trace], List[Node], Int32, Int32]:
    var n_traces = len(traces)
    if n_traces == 0:
        raise Error("Input traces must be a non-empty list")

    check_array_dtype[DType.int32](shifts)
    check_array_dtype[DType.float32](weights)
    check_array_dtype[DType.int32](offsets)

    if Int(py=shifts.ndim) != 2 or Int(py=weights.ndim) != 2:
        raise Error("Shifts and weights must be 2D arrays")
    if Int(py=offsets.ndim) != 1:
        raise Error("Offsets must be a 1D array")

    var n_nodes = Int(py=shifts.shape[0])
    if n_nodes == 0:
        raise Error("Number of nodes must be greater than zero")

    if Int(py=shifts.shape[0]) != Int(py=weights.shape[0]) or Int(
        py=shifts.shape[1]
    ) != Int(py=weights.shape[1]):
        raise Error("Shifts and weights must have the same shape")
    if n_traces != Int(py=offsets.shape[0]):
        raise Error("Number of arrays must match number of offsets")
    if Int(py=shifts.shape[1]) != n_traces:
        raise Error("Shifts must have the same number of columns as traces")

    var has_mask = node_mask is not None
    if has_mask:
        check_array_dtype[DType.bool](node_mask)
        if Int(py=node_mask.ndim) != 1 or Int(py=node_mask.shape[0]) != n_nodes:
            raise Error("Number of nodes must match number of activation flags")

    var offsets_ptr = numpy_ptr[DType.int32](offsets)
    var shifts_ptr = numpy_ptr[DType.int32](shifts)
    var weights_ptr = numpy_ptr[DType.float32](weights)

    var traces_list = List[Trace](capacity=n_traces)
    for i_trace in range(n_traces):
        var trace_arr = traces[i_trace]
        check_array_dtype[DType.float32](trace_arr)
        if Int(py=trace_arr.ndim) != 1:
            raise Error("Each trace must be a 1D array")
        traces_list.append(
            Trace(
                data=numpy_ptr[DType.float32](trace_arr),
                size=Int(py=trace_arr.size),
                offset=Int(offsets_ptr.unsafe_load(i_trace)),
            )
        )

    var nodes_list = List[Node](capacity=n_nodes)
    if has_mask:
        var mask_ptr = numpy_ptr[DType.uint8](node_mask)
        for i_node in range(n_nodes):
            nodes_list.append(
                Node(
                    shifts=shifts_ptr.unsafe_offset(i_node * n_traces),
                    weights=weights_ptr.unsafe_offset(i_node * n_traces),
                    masked=Bool(mask_ptr.unsafe_load(i_node) != 0),
                )
            )
    else:
        for i_node in range(n_nodes):
            nodes_list.append(
                Node(
                    shifts=shifts_ptr.unsafe_offset(i_node * n_traces),
                    weights=weights_ptr.unsafe_offset(i_node * n_traces),
                    masked=False,
                )
            )

    var min_shift = Int32.MAX
    var max_shift = Int32.MIN
    for i_node in range(n_nodes):
        ref node = nodes_list[i_node]
        for i_trace in range(n_traces):
            var idx_begin = Int32(
                traces_list[i_trace].offset
            ) + node.shifts.unsafe_load(i_trace)
            var idx_end = idx_begin + Int32(traces_list[i_trace].size)
            min_shift = min(min_shift, idx_begin)
            max_shift = max(max_shift, idx_end)

    return traces_list^, nodes_list^, min_shift, max_shift


def resolve_shift_range(
    shift_range: PythonObject, min_shift: Int32, max_shift: Int32
) raises -> Tuple[Int32, Int32]:
    if shift_range is None:
        return min_shift, max_shift

    if len(shift_range) != 2:
        raise Error(
            "shift_range argument must be tuple of two integers or None."
        )

    var new_min = Int32(Int(py=shift_range[0]))
    var new_max = Int32(Int(py=shift_range[1]))
    if new_max <= new_min:
        raise Error(
            "Invalid shift_range: max_shift must be greater than min_shift."
        )
    return new_min, new_max


# ===----------------------------------------------------------------=== #
# delay_sum
# ===----------------------------------------------------------------=== #


def stack_chunk(
    node_start: Int,
    node_end: Int,
    n_traces: Int,
    traces: List[Trace],
    nodes: List[Node],
    stack_data: Pointer[Float32, MutUntrackedOrigin],
    stack_size: Int,
    min_shift: Int32,
):
    comptime simd_width = simd_width_of[DType.float32]()

    for i_node in range(node_start, node_end):
        ref node = nodes[i_node]
        var node_stack = stack_data.unsafe_offset(i_node * stack_size)

        for i_trace in range(n_traces):
            var weight = node.weights.unsafe_load(i_trace)
            if weight == Float32(0):
                continue

            var trace = traces[i_trace].copy()
            var trace_shift = Int32(trace.offset) + node.shifts.unsafe_load(
                i_trace
            )
            var base_idx = Int(trace_shift - min_shift)
            var start = max(0, -base_idx)
            var stack_nsamples = min(stack_size - base_idx, trace.size)
            var n_samples = stack_nsamples - start
            if n_samples <= 0:
                continue

            def kernel[width: Int](i: Int) {imm}:
                var i_dst = base_idx + start + i
                var t = trace.data.unsafe_load[width=width](start + i)
                var s = node_stack.unsafe_load[width=width](i_dst)
                node_stack.unsafe_store(i_dst, t.fma(weight, s))

            vectorize[simd_width](n_samples, kernel)


def delay_sum(
    traces: PythonObject,
    offsets: PythonObject,
    shifts: PythonObject,
    weights: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    var node_mask = kwargs.pop(String("node_mask"), PythonObject(None))
    var stack = kwargs.pop(String("stack"), PythonObject(None))
    var shift_range = kwargs.pop(String("shift_range"), PythonObject(None))
    var n_threads = Int(py=kwargs.pop(String("n_threads"), PythonObject(1)))

    var out = prepare(traces, offsets, shifts, weights, node_mask)
    var traces_list = out[0].copy()
    var nodes_list = out[1].copy()

    var min_shift: Int32
    var max_shift: Int32
    min_shift, max_shift = resolve_shift_range(shift_range, out[2], out[3])

    var n_traces = len(traces_list)
    var n_nodes = len(nodes_list)
    var stack_size = Int(max_shift - min_shift)

    var np = Python.import_module("numpy")
    if stack is None:
        stack = np.zeros(
            Python.tuple(n_nodes, stack_size), dtype=PythonObject("float32")
        )
    else:
        check_array_dtype[DType.float32](stack)
        if (
            Int(py=stack.ndim) != 2
            or Int(py=stack.shape[0]) != n_nodes
            or Int(py=stack.shape[1]) != stack_size
        ):
            raise Error(
                "Resulting stack array must have shape (n_nodes, stack_size)"
            )

    var stack_data = numpy_ptr[DType.float32](stack)
    _ = get_thread_count(
        n_threads
    )  # accepted for API compatibility; see module docstring

    stack_chunk(
        0,
        n_nodes,
        n_traces,
        traces_list,
        nodes_list,
        stack_data,
        stack_size,
        min_shift,
    )

    return Python.tuple(stack, Int(min_shift))


# ===----------------------------------------------------------------=== #
# delay_sum_reduce
# ===----------------------------------------------------------------=== #


def reduce_tile(
    tile_start: Int,
    tile_end: Int,
    n_traces: Int,
    traces: List[Trace],
    nodes: List[Node],
    min_shift: Int32,
    stack_max_data: Pointer[Float32, MutUntrackedOrigin],
    stack_max_idx_data: Pointer[Int32, MutUntrackedOrigin],
):
    comptime simd_width = simd_width_of[DType.float32]()
    var tile_size = tile_end - tile_start
    var tile_stack = unsafe_alloc[Float32](tile_size)

    for i_node in range(len(nodes)):
        ref node = nodes[i_node]
        if node.masked:
            continue

        unsafe_memset_zero(tile_stack, tile_size)

        for i_trace in range(n_traces):
            var weight = node.weights.unsafe_load(i_trace)
            if weight == Float32(0):
                continue

            var trace = traces[i_trace].copy()
            var trace_shift = Int32(trace.offset) + node.shifts.unsafe_load(
                i_trace
            )
            var base_idx = Int(trace_shift - min_shift)

            var tile_base_idx = max(0, base_idx - tile_start)
            var trace_start_idx = max(0, tile_start - base_idx)
            var trace_end_idx = max(0, tile_end - base_idx)
            trace_start_idx = min(trace_start_idx, trace.size)
            trace_end_idx = min(trace_end_idx, trace.size)
            var n_samples = trace_end_idx - trace_start_idx
            if n_samples <= 0:
                continue

            def kernel[width: Int](i: Int) {imm}:
                var i_dst = tile_base_idx + i
                var t = trace.data.unsafe_load[width=width](trace_start_idx + i)
                var s = tile_stack.unsafe_load[width=width](i_dst)
                tile_stack.unsafe_store(i_dst, t.fma(weight, s))

            vectorize[simd_width](n_samples, kernel)

        # Vectorized max/argmax update: SIMD compare + select on both the
        # value and index lanes. The C original leaves this loop scalar
        # (its SIMD attempt is commented out, using masked_store on both
        # arrays), because SIMDE didn't offer a portable masked int32
        # store; `SIMD.select` sidesteps that entirely.
        var node_idx = Int32(i_node)

        def update_max[width: Int](i: Int) {imm}:
            var res_idx = tile_start + i
            var new_val = tile_stack.unsafe_load[width=width](i)
            var old_val = stack_max_data.unsafe_load[width=width](res_idx)
            var is_greater = new_val.gt(old_val)
            stack_max_data.unsafe_store(
                res_idx, is_greater.select(new_val, old_val)
            )
            var old_idx = stack_max_idx_data.unsafe_load[width=width](res_idx)
            var node_vec = SIMD[DType.int32, width](node_idx)
            stack_max_idx_data.unsafe_store(
                res_idx, is_greater.select(node_vec, old_idx)
            )

        vectorize[simd_width](tile_size, update_max)

    tile_stack.unsafe_free()


def delay_sum_reduce(
    traces: PythonObject,
    offsets: PythonObject,
    shifts: PythonObject,
    weights: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    var node_mask = kwargs.pop(String("node_mask"), PythonObject(None))
    var shift_range = kwargs.pop(String("shift_range"), PythonObject(None))
    var node_stack_max = kwargs.pop(
        String("node_stack_max"), PythonObject(None)
    )
    var node_stack_max_idx = kwargs.pop(
        String("node_stack_max_idx"), PythonObject(None)
    )
    var n_threads = Int(py=kwargs.pop(String("n_threads"), PythonObject(1)))

    if (node_stack_max is None) != (node_stack_max_idx is None):
        raise Error(
            "node_stack_max and node_stack_max_idx must be both provided or"
            " both None"
        )

    var out = prepare(traces, offsets, shifts, weights, node_mask)
    var traces_list = out[0].copy()
    var nodes_list = out[1].copy()

    var min_shift: Int32
    var max_shift: Int32
    min_shift, max_shift = resolve_shift_range(shift_range, out[2], out[3])

    var n_traces = len(traces_list)
    var stack_size = Int(max_shift - min_shift)

    var np = Python.import_module("numpy")
    var owns_result = node_stack_max is None
    if owns_result:
        node_stack_max = np.full(
            stack_size,
            np.finfo(PythonObject("float32")).min,
            dtype=PythonObject("float32"),
        )
        node_stack_max_idx = np.zeros(stack_size, dtype=PythonObject("int32"))
    else:
        check_array_dtype[DType.float32](node_stack_max)
        check_array_dtype[DType.int32](node_stack_max_idx)
        if (
            Int(py=node_stack_max.shape[0]) != stack_size
            or Int(py=node_stack_max_idx.shape[0]) != stack_size
        ):
            raise Error(
                "Provided result arrays must be 1D NumPy arrays of float32 and"
                " int respectively, with correct length"
            )

    var stack_max_data = numpy_ptr[DType.float32](node_stack_max)
    var stack_max_idx_data = numpy_ptr[DType.int32](node_stack_max_idx)
    _ = get_thread_count(
        n_threads
    )  # accepted for API compatibility; see module docstring

    reduce_tile(
        0,
        stack_size,
        n_traces,
        traces_list,
        nodes_list,
        min_shift,
        stack_max_data,
        stack_max_idx_data,
    )

    return Python.tuple(node_stack_max, node_stack_max_idx, Int(min_shift))


# ===----------------------------------------------------------------=== #
# delay_sum_snapshot
# ===----------------------------------------------------------------=== #


def delay_sum_snapshot(
    traces: PythonObject,
    offsets: PythonObject,
    shifts: PythonObject,
    weights: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    if not ("index" in kwargs):
        raise Error(
            "delay_sum_snapshot() missing required keyword argument: 'index'"
        )
    var index = Int(py=kwargs.pop(String("index"), PythonObject(0)))
    var shift_range = kwargs.pop(String("shift_range"), PythonObject(None))
    var node_mask = kwargs.pop(String("node_mask"), PythonObject(None))

    var out = prepare(traces, offsets, shifts, weights, node_mask)
    var traces_list = out[0].copy()
    var nodes_list = out[1].copy()

    var min_shift: Int32
    var max_shift: Int32
    min_shift, max_shift = resolve_shift_range(shift_range, out[2], out[3])

    var stack_size = Int(max_shift - min_shift)
    if index < 0 or index >= stack_size:
        raise Error("Snapshot index out of bounds: " + String(index))

    var n_traces = len(traces_list)
    var n_nodes = len(nodes_list)

    var np = Python.import_module("numpy")
    var snapshot = np.zeros(n_nodes, dtype=PythonObject("float32"))
    var snapshot_data = numpy_ptr[DType.float32](snapshot)

    for i_node in range(n_nodes):
        ref node = nodes_list[i_node]
        if node.masked:
            continue
        var acc = Float32(0)
        for i_trace in range(n_traces):
            var weight = node.weights.unsafe_load(i_trace)
            if weight == Float32(0):
                continue
            ref trace = traces_list[i_trace]
            var trace_shift = Int32(trace.offset) + node.shifts.unsafe_load(
                i_trace
            )
            var base_idx = Int(trace_shift - min_shift)
            var trace_sample = index - base_idx
            if 0 <= trace_sample < trace.size:
                acc += trace.data.unsafe_load(trace_sample) * weight
        snapshot_data.unsafe_store(i_node, acc)

    return snapshot
