"""Delay-and-sum beamforming, ported from src/qseek/ext/delay_sum.c to Mojo.

Struct-oriented design (inspired by the older `parstack.mojo` prototype on
branch `features/mojo`, adapted to Mojo 1.0):

- `Trace` and `Node` are pointer-backed value types with their own compute
  methods. `Node.accumulate_trace` clips one trace against an arbitrary
  destination window, so the full-range `delay_sum` path and the per-tile
  `delay_sum_reduce` path share it instead of duplicating the offset
  arithmetic that the C original repeats twice.
- `Grid` validates and owns the traces/nodes for one call, and resolves the
  shift range, so each entry point is just "build a Grid, then compute".
  `Grid.view()` hands tasks a borrowed `GridView` so the owner stays on the
  calling thread.

SIMD lanes come from `vectorize` (compiled against the host's native vector
width) instead of hand-rolled AVX2/SIMDE intrinsics.

Both stacking paths iterate nodes in blocks (`stack_block`) with the trace
loop hoisted outside the node loop, so each trace slice is pulled from L3
once per block and reused from L1 for the rest of it. Accumulation is ~98%
of the work and was bound by re-reading the entire trace working set once
per node; blocking cuts that traffic by the block factor.


Threading: this module depends on a pre-stable API
--------------------------------------------------

Parallelism uses `std.runtime.asyncrt.TaskGroup` (`async def` + `create_task`
+ `wait`). **That API is not stable in Mojo 1.0.0** and is the one piece of
this module likely to break on a toolchain upgrade:

- The Mojo manual states plainly that "Mojo doesn't yet support async
  execution", and "First-class `async` support: fully integrated with Mojo's
  type and memory models" is still an open (unchecked) roadmap item.
- `runtime.asyncrt` is documented only as a "low level concurrency library";
  it is absent from the manual's user-facing chapters.

It is used anyway because Mojo 1.0.0's stable stdlib has no parallel-for at
all: `parallelize` does not exist, and `algorithm.map` is serial (measured at
1.2x across 32 cores). The alternatives were single-threaded (~3.5x slower)
or hand-rolled POSIX threads via `std.ffi` -- which was implemented and
measured, but scaled worse than `TaskGroup` (2.4x vs 3.6x on 4 threads),
because freshly spawned threads land on this class of CPU's efficiency cores
while the runtime's warm pool stays on performance cores.

Two consequences of async not yet being integrated with the memory model
are worked around here; both were confirmed with minimal repros outside this
file, and both are pinned by the tests in `test/test_delay_sum_mojo.py`:

1. `List[...]` passed into an `async def` reads garbage on a task's first
   loop iteration. Hence `Grid`/`GridView` hold `Pointer`-backed arrays
   rather than `List`.
2. Task lifetimes are not tracked, so a value's `__deinit__` can run before
   the tasks using it are scheduled. `Grid` owns heap allocations, and Mojo
   destroys a value at its last lexical use -- which would be inside the
   `create_task` loop, freeing the traces/nodes out from under the running
   tasks. Hence the `_ = grid^` after every `TaskGroup.wait()`; removing it
   reintroduces a use-after-free.

If a future Mojo release ships a stable parallel-for, replacing the
`TaskGroup` blocks in `delay_sum`/`delay_sum_reduce` with it should also let
workarounds (1) and (2) be dropped.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.os import abort
from std.memory import Pointer
from std.memory.alloc import unsafe_alloc
from std.memory.memory import unsafe_memset_zero
from std.algorithm.functional import vectorize
from std.math import ceildiv
from std.sys.info import simd_width_of, num_logical_cores
from std.runtime.asyncrt import TaskGroup

comptime F32Ptr = Pointer[Float32, MutUntrackedOrigin]
comptime I32Ptr = Pointer[Int32, MutUntrackedOrigin]
comptime SIMD_WIDTH = simd_width_of[DType.float32]()

# Nodes are stacked in blocks so each trace slice is read from L3 once per
# block and then reused from L1 for the rest of it. `NODE_BLOCK` caps the
# block; `BLOCK_SCRATCH_FLOATS` caps it again by size where the block needs
# scratch buffers, so they stay resident in L2.
comptime NODE_BLOCK = 8
comptime BLOCK_SCRATCH_FLOATS = 262144  # 1 MiB of float32 stacks


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


# ===----------------------------------------------------------------=== #
# NumPy helpers
# ===----------------------------------------------------------------=== #


@always_inline
def get_thread_count(n_threads: Int) -> Int:
    return num_logical_cores() if n_threads <= 0 else n_threads


@always_inline
def numpy_ptr[
    dtype: DType
](array: PythonObject) raises -> Pointer[Scalar[dtype], MutUntrackedOrigin]:
    return Pointer[Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=Int(py=array.ctypes.data)
    )


@always_inline
def dtype_char[dtype: DType]() -> StaticString:
    comptime if dtype == DType.float32:
        return "f"
    elif dtype == DType.int32:
        return "i"
    else:
        return "?"


@always_inline
def check_array_dtype[dtype: DType](array: PythonObject) raises:
    var expected = dtype_char[dtype]()
    if String(array.dtype.char) != expected:
        raise Error("Input array must be of type ", expected)


# ===----------------------------------------------------------------=== #
# Trace / Node
# ===----------------------------------------------------------------=== #


@fieldwise_init
struct Trace(Copyable, Movable):
    """A single seismic trace: a flat float32 buffer and its static offset."""

    var data: F32Ptr
    var size: Int
    var offset: Int

    def accumulate(
        self,
        weight: Float32,
        dest: F32Ptr,
        dest_start: Int,
        src_start: Int,
        n_samples: Int,
    ):
        """SIMD `dest[dest_start:] += weight * self.data[src_start:]`."""
        var data = self.data

        def kernel[width: Int](i: Int) {imm}:
            var d = dest_start + i
            var t = data.unsafe_load[width=width](src_start + i)
            var s = dest.unsafe_load[width=width](d)
            dest.unsafe_store(d, t.fma(weight, s))

        vectorize[SIMD_WIDTH](n_samples, kernel)


@fieldwise_init
struct Node(Copyable, Movable):
    """A grid node: per-trace integer sample shifts and float32 weights."""

    var shifts: I32Ptr
    var weights: F32Ptr
    var masked: Bool

    @always_inline
    def shifted_start(
        self, trace: Trace, i_trace: Int, min_shift: Int32
    ) -> Int:
        """Where this node places `trace`'s first sample in the result."""
        return Int(
            Int32(trace.offset) + self.shifts.unsafe_load(i_trace) - min_shift
        )

    def accumulate_trace(
        self,
        trace: Trace,
        i_trace: Int,
        dest: F32Ptr,
        window_start: Int,
        window_end: Int,
        min_shift: Int32,
    ):
        """Delay-and-sum one trace into `dest`, which covers the result
        window [window_start, window_end) -- `dest[0]` is sample
        `window_start`. Callers writing the full result pass 0; tiled
        callers pass their tile bounds.
        """
        var weight = self.weights.unsafe_load(i_trace)
        if weight == Float32(0):
            return

        # Intersect the shifted trace with the destination window.
        var base = self.shifted_start(trace, i_trace, min_shift)
        var begin = max(base, window_start)
        var end = min(base + trace.size, window_end)
        if end <= begin:
            return

        trace.accumulate(
            weight, dest, begin - window_start, begin - base, end - begin
        )

    def sample_at(
        self, grid: GridView, index: Int, min_shift: Int32
    ) -> Float32:
        """Delay-and-sum a single result sample across all traces."""
        var acc = Float32(0)
        for i_trace in range(grid.n_traces):
            var weight = self.weights.unsafe_load(i_trace)
            if weight == Float32(0):
                continue
            var trace = grid.traces[unsafe_offset=i_trace].copy()
            var sample = index - self.shifted_start(trace, i_trace, min_shift)
            if 0 <= sample < trace.size:
                acc += trace.data.unsafe_load(sample) * weight
        return acc


# ===----------------------------------------------------------------=== #
# Grid: validates and owns one call's traces/nodes
# ===----------------------------------------------------------------=== #


@fieldwise_init
struct GridView(Copyable, Movable):
    """Borrowed view of a `Grid`, cheap to copy into a task."""

    var traces: Pointer[Trace, MutUntrackedOrigin]
    var n_traces: Int
    var nodes: Pointer[Node, MutUntrackedOrigin]
    var n_nodes: Int


struct Grid:
    """Validated, pointer-backed view over one call's inputs.

    Also resolves the result window, so callers read `min_shift` and
    `stack_size` off the grid rather than re-deriving them.
    """

    var traces: Pointer[Trace, MutUntrackedOrigin]
    var n_traces: Int
    var nodes: Pointer[Node, MutUntrackedOrigin]
    var n_nodes: Int
    var min_shift: Int32
    var stack_size: Int

    def __init__(
        out self,
        traces: PythonObject,
        offsets: PythonObject,
        shifts: PythonObject,
        weights: PythonObject,
        node_mask: PythonObject,
        shift_range: PythonObject,
    ) raises:
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

        # `.shape` builds a fresh Python tuple on every access (it's a
        # property, not a field read like the C extension's PyArray_SHAPE),
        # so fetch each array's shape once and reuse it below.
        var shifts_shape = shifts.shape
        var weights_shape = weights.shape
        var n_nodes = Int(py=shifts_shape[0])
        if n_nodes == 0:
            raise Error("Number of nodes must be greater than zero")
        if Int(py=weights_shape[0]) != n_nodes or Int(
            py=shifts_shape[1]
        ) != Int(py=weights_shape[1]):
            raise Error("Shifts and weights must have the same shape")
        if n_traces != Int(py=offsets.shape[0]):
            raise Error("Number of arrays must match number of offsets")
        if Int(py=shifts_shape[1]) != n_traces:
            raise Error("Shifts must have the same number of columns as traces")

        var has_mask = node_mask is not None
        if has_mask:
            check_array_dtype[DType.bool](node_mask)
            if (
                Int(py=node_mask.ndim) != 1
                or Int(py=node_mask.shape[0]) != n_nodes
            ):
                raise Error(
                    "Number of nodes must match number of activation flags"
                )

        var offsets_ptr = numpy_ptr[DType.int32](offsets)
        var shifts_ptr = numpy_ptr[DType.int32](shifts)
        var weights_ptr = numpy_ptr[DType.float32](weights)

        self.n_traces = n_traces
        self.n_nodes = n_nodes
        self.traces = unsafe_alloc[Trace](n_traces)
        self.nodes = unsafe_alloc[Node](n_nodes)

        for i_trace in range(n_traces):
            var array = traces[i_trace]
            check_array_dtype[DType.float32](array)
            if Int(py=array.ndim) != 1:
                raise Error("Each trace must be a 1D array")
            self.traces.unsafe_offset(i_trace).unsafe_write(
                Trace(
                    numpy_ptr[DType.float32](array),
                    Int(py=array.size),
                    Int(offsets_ptr.unsafe_load(i_trace)),
                )
            )

        for i_node in range(n_nodes):
            self.nodes.unsafe_offset(i_node).unsafe_write(
                Node(
                    shifts_ptr.unsafe_offset(i_node * n_traces),
                    weights_ptr.unsafe_offset(i_node * n_traces),
                    masked=False,
                )
            )

        if has_mask:
            # Resolved once outside the loop: `numpy_ptr` walks a Python
            # attribute chain, which must not land on a per-node path.
            var mask_ptr = numpy_ptr[DType.uint8](node_mask)
            for i_node in range(n_nodes):
                ref node = self.nodes[unsafe_offset=i_node]
                node.masked = mask_ptr.unsafe_load(i_node) != 0

        var min_shift = Int32.MAX
        var max_shift = Int32.MIN
        for i_node in range(n_nodes):
            ref node = self.nodes[unsafe_offset=i_node]
            for i_trace in range(n_traces):
                ref trace = self.traces[unsafe_offset=i_trace]
                var begin = Int32(trace.offset) + node.shifts.unsafe_load(
                    i_trace
                )
                min_shift = min(min_shift, begin)
                max_shift = max(max_shift, begin + Int32(trace.size))

        if shift_range is not None:
            if len(shift_range) != 2:
                raise Error(
                    "shift_range argument must be tuple of two integers or"
                    " None."
                )
            min_shift = Int32(Int(py=shift_range[0]))
            max_shift = Int32(Int(py=shift_range[1]))
            if max_shift <= min_shift:
                raise Error(
                    "Invalid shift_range: max_shift must be greater than"
                    " min_shift."
                )

        self.min_shift = min_shift
        self.stack_size = Int(max_shift - min_shift)

    def __deinit__(deinit self):
        self.traces.unsafe_free()
        self.nodes.unsafe_free()

    @always_inline
    def view(self) -> GridView:
        return GridView(self.traces, self.n_traces, self.nodes, self.n_nodes)


# ===----------------------------------------------------------------=== #
# Shared blocked stacking kernel
# ===----------------------------------------------------------------=== #


@always_inline
def stack_block[
    skip_masked: Bool
](
    grid: GridView,
    node_start: Int,
    node_end: Int,
    dest: F32Ptr,
    dest_stride: Int,
    window_start: Int,
    window_end: Int,
    min_shift: Int32,
):
    """Stack nodes [node_start, node_end) into `dest_stride`-spaced buffers.

    The trace loop is outermost so each trace slice is loaded once for the
    whole block; node `i` accumulates into `dest[(i - node_start) * stride]`.
    """
    for i_trace in range(grid.n_traces):
        var trace = grid.traces[unsafe_offset=i_trace].copy()
        for i_node in range(node_start, node_end):
            var node = grid.nodes[unsafe_offset=i_node].copy()
            comptime if skip_masked:
                if node.masked:
                    continue
            node.accumulate_trace(
                trace,
                i_trace,
                dest.unsafe_offset((i_node - node_start) * dest_stride),
                window_start,
                window_end,
                min_shift,
            )


@always_inline
def chunk_bounds(i_chunk: Int, n_chunks: Int, total: Int) -> Tuple[Int, Int]:
    return i_chunk * total // n_chunks, (i_chunk + 1) * total // n_chunks


# ===----------------------------------------------------------------=== #
# delay_sum: threaded over nodes, each writing its own row of the result
# ===----------------------------------------------------------------=== #


async def stack_chunk(
    grid: GridView,
    node_start: Int,
    node_end: Int,
    stack_data: F32Ptr,
    stack_size: Int,
    min_shift: Int32,
):
    for block_start in range(node_start, node_end, NODE_BLOCK):
        var block_end = min(block_start + NODE_BLOCK, node_end)
        stack_block[skip_masked=False](
            grid,
            block_start,
            block_end,
            stack_data.unsafe_offset(block_start * stack_size),
            stack_size,
            0,
            stack_size,
            min_shift,
        )


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

    var grid = Grid(traces, offsets, shifts, weights, node_mask, shift_range)
    var stack_size = grid.stack_size

    if stack is None:
        var np = Python.import_module("numpy")
        stack = np.zeros(
            Python.tuple(grid.n_nodes, stack_size),
            dtype=PythonObject("float32"),
        )
    else:
        check_array_dtype[DType.float32](stack)
        var shape = stack.shape
        if (
            Int(py=stack.ndim) != 2
            or Int(py=shape[0]) != grid.n_nodes
            or Int(py=shape[1]) != stack_size
        ):
            raise Error(
                "Resulting stack array must have shape (n_nodes, stack_size)"
            )

    var stack_data = numpy_ptr[DType.float32](stack)
    var n_chunks = max(1, min(get_thread_count(n_threads), grid.n_nodes))

    var tg = TaskGroup()
    for i_chunk in range(n_chunks):
        var bounds = chunk_bounds(i_chunk, n_chunks, grid.n_nodes)
        tg.create_task(
            stack_chunk(
                grid.view(),
                bounds[0],
                bounds[1],
                stack_data,
                stack_size,
                grid.min_shift,
            )
        )
    tg.wait()
    # Keep the Grid's allocations alive past the tasks; see module docstring.
    _ = grid^

    return Python.tuple(stack, Int(grid.min_shift))


# ===----------------------------------------------------------------=== #
# delay_sum_reduce: threaded over the result's time axis, each task scanning
# every node (like the C original) and folding into a running max/argmax
# ===----------------------------------------------------------------=== #


def update_running_max(
    src: F32Ptr,
    n_samples: Int,
    dest_start: Int,
    node_idx: Int32,
    stack_max: F32Ptr,
    stack_max_idx: I32Ptr,
):
    """Fold one node's stack into the running (max, argmax) result arrays.

    Vectorized with SIMD compare + select on both the value and index lanes.
    The C original leaves this loop scalar (its SIMD attempt is commented
    out, using masked_store on both arrays) because SIMDE didn't offer a
    portable masked int32 store; `SIMD.select` sidesteps that entirely.
    """

    def kernel[width: Int](i: Int) {imm}:
        var d = dest_start + i
        var new_val = src.unsafe_load[width=width](i)
        var old_val = stack_max.unsafe_load[width=width](d)
        var is_greater = new_val.gt(old_val)
        stack_max.unsafe_store(d, is_greater.select(new_val, old_val))
        var old_idx = stack_max_idx.unsafe_load[width=width](d)
        var node_vec = SIMD[DType.int32, width](node_idx)
        stack_max_idx.unsafe_store(d, is_greater.select(node_vec, old_idx))

    vectorize[SIMD_WIDTH](n_samples, kernel)


async def reduce_tile(
    grid: GridView,
    tile_start: Int,
    tile_end: Int,
    min_shift: Int32,
    stacks: F32Ptr,
    block: Int,
    stack_max: F32Ptr,
    stack_max_idx: I32Ptr,
):
    var tile_size = tile_end - tile_start

    for block_start in range(0, grid.n_nodes, block):
        var block_end = min(block_start + block, grid.n_nodes)
        unsafe_memset_zero(stacks, (block_end - block_start) * tile_size)

        stack_block[skip_masked=True](
            grid,
            block_start,
            block_end,
            stacks,
            tile_size,
            tile_start,
            tile_end,
            min_shift,
        )

        for i_node in range(block_start, block_end):
            if grid.nodes[unsafe_offset=i_node].copy().masked:
                continue
            update_running_max(
                stacks.unsafe_offset((i_node - block_start) * tile_size),
                tile_size,
                tile_start,
                Int32(i_node),
                stack_max,
                stack_max_idx,
            )


def delay_sum_reduce(
    traces: PythonObject,
    offsets: PythonObject,
    shifts: PythonObject,
    weights: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    var node_mask = kwargs.pop(String("node_mask"), PythonObject(None))
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

    var grid = Grid(traces, offsets, shifts, weights, node_mask, shift_range)
    var stack_size = grid.stack_size

    if node_max is None:
        var np = Python.import_module("numpy")
        node_max = np.full(
            stack_size,
            np.finfo(PythonObject("float32")).min,
            dtype=PythonObject("float32"),
        )
        node_max_idx = np.zeros(stack_size, dtype=PythonObject("int32"))
    else:
        check_array_dtype[DType.float32](node_max)
        check_array_dtype[DType.int32](node_max_idx)
        if (
            Int(py=node_max.shape[0]) != stack_size
            or Int(py=node_max_idx.shape[0]) != stack_size
        ):
            raise Error(
                "Provided result arrays must be 1D NumPy arrays of float32 and"
                " int respectively, with correct length"
            )

    var stack_max = numpy_ptr[DType.float32](node_max)
    var stack_max_idx = numpy_ptr[DType.int32](node_max_idx)
    var n_chunks = max(1, min(get_thread_count(n_threads), stack_size))

    # One scratch allocation shared by every task, sliced per chunk:
    # allocating inside each task instead put a malloc large enough to be
    # mmap'd (and its page faults) on the hot path, which showed up as
    # run-to-run jitter.
    var max_tile = ceildiv(stack_size, n_chunks)
    var block = max(1, min(NODE_BLOCK, BLOCK_SCRATCH_FLOATS // max_tile))
    var scratch = unsafe_alloc[Float32](n_chunks * block * max_tile)

    var tg = TaskGroup()
    for i_chunk in range(n_chunks):
        var bounds = chunk_bounds(i_chunk, n_chunks, stack_size)
        tg.create_task(
            reduce_tile(
                grid.view(),
                bounds[0],
                bounds[1],
                grid.min_shift,
                scratch.unsafe_offset(i_chunk * block * max_tile),
                block,
                stack_max,
                stack_max_idx,
            )
        )
    tg.wait()
    scratch.unsafe_free()
    # Keep the Grid's allocations alive past the tasks; see module docstring.
    _ = grid^

    return Python.tuple(node_max, node_max_idx, Int(grid.min_shift))


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
    if "index" not in kwargs:
        raise Error(
            "delay_sum_snapshot() missing required keyword argument: 'index'"
        )
    var index = Int(py=kwargs.pop(String("index"), PythonObject(0)))
    var shift_range = kwargs.pop(String("shift_range"), PythonObject(None))
    var node_mask = kwargs.pop(String("node_mask"), PythonObject(None))

    var grid = Grid(traces, offsets, shifts, weights, node_mask, shift_range)
    if index < 0 or index >= grid.stack_size:
        raise Error("Snapshot index out of bounds: ", index)

    var np = Python.import_module("numpy")
    var snapshot = np.zeros(grid.n_nodes, dtype=PythonObject("float32"))
    var snapshot_data = numpy_ptr[DType.float32](snapshot)
    var view = grid.view()

    for i_node in range(grid.n_nodes):
        var node = grid.nodes[unsafe_offset=i_node].copy()
        if node.masked:
            continue
        snapshot_data.unsafe_store(
            i_node, node.sample_at(view, index, grid.min_shift)
        )

    return snapshot
