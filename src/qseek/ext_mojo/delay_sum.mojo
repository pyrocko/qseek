"""Delay-and-sum beamforming, a Mojo port of `src/qseek/ext/delay_sum.c`.

All entry points take a list of traces (`qseek.reduce.TraceInput`: `data`,
`offset`) and a list of nodes (`qseek.reduce.NodeStack`: `index`, `shifts`,
`weights`). Callers pass the nodes they want stacked instead of a node mask.

Threading follows the C extension: `delay_sum` runs the nodes in parallel,
each writing its own row, and `delay_sum_reduce` splits the time axis into
one tile per thread, each folding every node into the running maximum.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.python.numpy import from_numpy_array, from_numpy_tensor
from std.python._cpython import PyThreadState
from std.os import abort
from std.runtime import initialize_runtime
from std.memory import Layout, alloc, dealloc
from std.memory.memory import unsafe_memset_zero
from std.algorithm.functional import vectorize
from std.sys.info import simd_width_of, num_logical_cores
from max.algorithm import parallelize

comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime SIMD_WIDTH = simd_width_of[DType.float32]()


@export
def PyInit_delay_sum() abi("C") -> PythonObject:
    # Starts the thread pool `parallelize` runs on; no Mojo `main()` does it
    # for a Python extension.
    initialize_runtime()
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


struct GILReleased:
    """Releases the GIL for the duration of a `with` block.

    Nothing inside the block may touch a `PythonObject`.
    """

    var state: Optional[Pointer[PyThreadState, MutUntrackedOrigin]]

    def __init__(out self):
        self.state = Python().cpython().PyEval_SaveThread()

    def __enter__(self):
        pass

    def __exit__(deinit self):
        Python().cpython().PyEval_RestoreThread(self.state)


def get_thread_count(n_threads: Int) -> Int:
    return num_logical_cores() if n_threads <= 0 else n_threads


def borrow_1d[
    dtype: DType
](array: PythonObject) raises -> Tuple[
    Pointer[Scalar[dtype], MutUntrackedOrigin], Int
]:
    """Borrow a 1-D C-contiguous array of `dtype` as (pointer, length).

    The caller keeps the array alive while the pointer is in use.
    """
    var span = from_numpy_array[dtype](array)
    var ptr = Pointer[Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=Int(span.unsafe_ptr())
    )
    return (ptr, len(span))


@fieldwise_init
struct Trace(Copyable, Movable):
    var data: F32Ptr
    var size: Int
    var offset: Int


@fieldwise_init
struct NodeStack(Copyable, Movable):
    var shifts: I32Ptr
    var weights: F32Ptr
    var index: Int32


def accumulate(dest: F32Ptr, src: F32Ptr, weight: Float32, n_samples: Int):
    """`dest[:n_samples] += weight * src[:n_samples]`."""

    def kernel[width: Int](i: Int) {imm}:
        var acc = dest.unsafe_load[width=width](i)
        var sample = src.unsafe_load[width=width](i)
        dest.unsafe_store(i, sample.fma(weight, acc))

    vectorize[SIMD_WIDTH](n_samples, kernel)


def update_running_max(
    stack: F32Ptr,
    n_samples: Int,
    node_index: Int32,
    stack_max: F32Ptr,
    stack_max_idx: I32Ptr,
):
    """Fold one node's stack into the running (max, argmax)."""

    def kernel[width: Int](i: Int) {imm}:
        var value = stack.unsafe_load[width=width](i)
        var current = stack_max.unsafe_load[width=width](i)
        var greater = value.gt(current)
        stack_max.unsafe_store(i, greater.select(value, current))
        var current_idx = stack_max_idx.unsafe_load[width=width](i)
        stack_max_idx.unsafe_store(
            i, greater.select(SIMD[DType.int32, width](node_index), current_idx)
        )

    vectorize[SIMD_WIDTH](n_samples, kernel)


struct Grid(Movable):
    """One call's traces and nodes, and the result window they span."""

    var traces: List[Trace]
    var nodes: List[NodeStack]
    var min_shift: Int
    var stack_size: Int

    def __init__(
        out self,
        traces: PythonObject,
        nodes: PythonObject,
        shift_range: PythonObject,
    ) raises:
        var n_traces = len(traces)
        if n_traces == 0:
            raise Error("Input traces must be a non-empty list")
        if len(nodes) == 0:
            raise Error("Number of nodes must be greater than zero")

        self.traces = List[Trace](capacity=n_traces)
        for trace in traces:
            var data = borrow_1d[DType.float32](trace.data)
            self.traces.append(Trace(data[0], data[1], Int(py=trace.offset)))

        self.nodes = List[NodeStack](capacity=len(nodes))
        for node in nodes:
            var shifts = borrow_1d[DType.int32](node.shifts)
            var weights = borrow_1d[DType.float32](node.weights)
            if shifts[1] != n_traces or weights[1] != n_traces:
                raise Error(
                    "node.shifts and node.weights must have length n_traces"
                )
            self.nodes.append(
                NodeStack(shifts[0], weights[0], Int32(Int(py=node.index)))
            )

        var min_shift: Int
        var max_shift: Int
        if shift_range is None:
            min_shift = Int.MAX
            max_shift = Int.MIN
            for ref node in self.nodes:
                for i_trace in range(n_traces):
                    ref trace = self.traces[i_trace]
                    var start = trace.offset + Int(
                        node.shifts.unsafe_load(i_trace)
                    )
                    min_shift = min(min_shift, start)
                    max_shift = max(max_shift, start + trace.size)
        else:
            if len(shift_range) != 2:
                raise Error(
                    "shift_range argument must be tuple of two integers or"
                    " None."
                )
            min_shift = Int(py=shift_range[0])
            max_shift = Int(py=shift_range[1])
            if max_shift <= min_shift:
                raise Error(
                    "Invalid shift_range: max_shift must be greater than"
                    " min_shift."
                )
        self.min_shift = min_shift
        self.stack_size = max_shift - min_shift

    def stack_node(
        self, node: NodeStack, dest: F32Ptr, window_start: Int, window_end: Int
    ):
        """Delay-and-sum `node` into `dest`, which holds the result samples
        [window_start, window_end).
        """
        for i_trace in range(len(self.traces)):
            var weight = node.weights.unsafe_load(i_trace)
            if weight == 0:
                continue
            ref trace = self.traces[i_trace]
            var base = (
                trace.offset
                + Int(node.shifts.unsafe_load(i_trace))
                - self.min_shift
            )
            var start = max(base, window_start)
            var end = min(base + trace.size, window_end)
            if end > start:
                accumulate(
                    dest.unsafe_offset(start - window_start),
                    trace.data.unsafe_offset(start - base),
                    weight,
                    end - start,
                )

    def sample(self, node: NodeStack, index: Int) -> Float32:
        """Delay-and-sum `node` at the single result sample `index`."""
        var acc = Float32(0)
        for i_trace in range(len(self.traces)):
            var weight = node.weights.unsafe_load(i_trace)
            if weight == 0:
                continue
            ref trace = self.traces[i_trace]
            var i_sample = (
                index
                - trace.offset
                - Int(node.shifts.unsafe_load(i_trace))
                + self.min_shift
            )
            if 0 <= i_sample < trace.size:
                acc += trace.data.unsafe_load(i_sample) * weight
        return acc


def delay_sum(
    traces: PythonObject,
    nodes: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    """Stack every node into its own row of a (n_nodes, stack_size) array."""
    var stack = kwargs.pop(String("stack"), PythonObject(None))
    var shift_range = kwargs.pop(String("shift_range"), PythonObject(None))
    var n_threads = Int(py=kwargs.pop(String("n_threads"), PythonObject(1)))

    var grid = Grid(traces, nodes, shift_range)
    var n_nodes = len(grid.nodes)
    var stack_size = grid.stack_size

    if stack is None:
        var np = Python.import_module("numpy")
        stack = np.zeros(
            Python.tuple(n_nodes, stack_size), dtype=PythonObject("float32")
        )
    var view = from_numpy_tensor[DType.float32, 2](stack)
    if view.shape[0] != n_nodes or view.shape[1] != stack_size:
        raise Error("stack must have shape (", n_nodes, ", ", stack_size, ")")
    var stack_data = F32Ptr(unsafe_from_address=Int(view.data.unsafe_ptr()))

    def stack_row(i_node: Int) {imm grid, imm stack_data, imm stack_size}:
        grid.stack_node(
            grid.nodes[i_node],
            stack_data.unsafe_offset(i_node * stack_size),
            0,
            stack_size,
        )

    with GILReleased():
        parallelize(
            stack_row, n_nodes, min(get_thread_count(n_threads), n_nodes)
        )

    return Python.tuple(stack, grid.min_shift)


def delay_sum_reduce(
    traces: PythonObject,
    nodes: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    """Fold every node's stack into a running (max, argmax) over time.

    The argmax records `NodeStack.index`, so callers can fold in a subset of
    the nodes per call.
    """
    var shift_range = kwargs.pop(String("shift_range"), PythonObject(None))
    var node_max = kwargs.pop(String("node_stack_max"), PythonObject(None))
    var node_max_idx = kwargs.pop(
        String("node_stack_max_idx"), PythonObject(None)
    )
    var n_threads = Int(py=kwargs.pop(String("n_threads"), PythonObject(1)))

    if (node_max is None) != (node_max_idx is None):
        raise Error(
            "node_stack_max and node_stack_max_idx must be both provided or"
            " both None"
        )

    var grid = Grid(traces, nodes, shift_range)
    var stack_size = grid.stack_size

    if node_max is None:
        var np = Python.import_module("numpy")
        node_max = np.full(stack_size, -np.inf, dtype=PythonObject("float32"))
        node_max_idx = np.zeros(stack_size, dtype=PythonObject("int32"))

    var max_borrow = borrow_1d[DType.float32](node_max)
    var idx_borrow = borrow_1d[DType.int32](node_max_idx)
    if max_borrow[1] != stack_size or idx_borrow[1] != stack_size:
        raise Error(
            "node_stack_max and node_stack_max_idx must both have length",
            stack_size,
        )
    var stack_max = max_borrow[0]
    var stack_max_idx = idx_borrow[0]

    # One tile of the time axis per thread, each with its own slice of
    # `scratch` to stack a node into.
    var n_tiles = min(get_thread_count(n_threads), stack_size)
    var scratch = alloc(Layout[Float32](count=stack_size))
    var scratch_data = F32Ptr(unsafe_from_address=Int(scratch.unsafe_ptr()))

    def reduce_tile(
        i_tile: Int,
    ) {
        imm grid,
        imm scratch_data,
        imm stack_max,
        imm stack_max_idx,
        imm stack_size,
        imm n_tiles,
    }:
        var start = i_tile * stack_size // n_tiles
        var end = (i_tile + 1) * stack_size // n_tiles
        var tile = scratch_data.unsafe_offset(start)
        for ref node in grid.nodes:
            unsafe_memset_zero(tile, end - start)
            grid.stack_node(node, tile, start, end)
            update_running_max(
                tile,
                end - start,
                node.index,
                stack_max.unsafe_offset(start),
                stack_max_idx.unsafe_offset(start),
            )

    with GILReleased():
        parallelize(reduce_tile, n_tiles, n_tiles)
    dealloc(scratch^)

    return Python.tuple(node_max, node_max_idx, grid.min_shift)


def delay_sum_snapshot(
    traces: PythonObject,
    nodes: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    """Delay-and-sum of every node at one sample, aligned with `nodes`."""
    if "index" not in kwargs:
        raise Error(
            "delay_sum_snapshot() missing required keyword argument: 'index'"
        )
    var index = Int(py=kwargs.pop(String("index"), PythonObject(0)))
    var shift_range = kwargs.pop(String("shift_range"), PythonObject(None))

    var grid = Grid(traces, nodes, shift_range)
    if index < 0 or index >= grid.stack_size:
        raise Error("Snapshot index out of bounds: ", index)

    var np = Python.import_module("numpy")
    var snapshot = np.zeros(len(grid.nodes), dtype=PythonObject("float32"))
    var snapshot_data = borrow_1d[DType.float32](snapshot)[0]

    with GILReleased():
        for i_node in range(len(grid.nodes)):
            snapshot_data.unsafe_store(
                i_node, grid.sample(grid.nodes[i_node], index)
            )

    return snapshot
