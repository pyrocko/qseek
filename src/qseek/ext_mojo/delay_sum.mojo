"""Delay-and-sum beamforming, ported from src/qseek/ext/delay_sum.c to Mojo.

Struct-oriented design (inspired by the older `parstack.mojo` prototype on
branch `features/mojo`, adapted to Mojo 1.0):

- `Trace` and `NodeStack` are pointer-backed value types with their own
  compute methods. `NodeStack.accumulate_trace` clips one trace against an
  arbitrary destination window, so the full-range `delay_sum` path and the
  per-tile `delay_sum_reduce` path share it instead of duplicating the
  offset arithmetic that the C original repeats twice.
- `Grid` validates and owns the traces/nodes for one call, and resolves the
  result window, so each entry point is just "build a Grid, then compute".
  `Grid.view()` hands tasks a borrowed `GridView` so the owner stays on the
  calling thread. Inputs are validated and borrowed through `borrow_1d`,
  which wraps the stdlib's `from_numpy_array` -- that checks dtype, rank
  and C-contiguity natively, the last mattering because a transposed or
  strided view would otherwise be read as though packed, silently
  returning wrong numbers.
  All three entry points take the same inputs: a list of trace arrays and
  a list of per-node objects (`qseek.reduce.NodeStack`), rather than the C
  extension's shared 2D (shifts, weights) arrays plus a `node_mask`. Node
  bookkeeping -- which nodes are new, which are leaves -- is Python's job,
  so callers pass the subset they want stacked instead of everything plus
  a mask, and `Grid` needs neither row-striding nor mask handling.

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

It is used anyway because the alternatives measured worse on the workload
this module actually runs (100 nodes x 100 traces x 30k samples,
`delay_sum_reduce`, median ms):

- `max.algorithm.parallelize` is the native parallel-for, but it lives in
  the MAX package rather than the Mojo stdlib. It ties `TaskGroup` up to 8
  threads (nt=4: 6.04 vs 6.11) and then falls behind as threads oversubscribe
  the performance cores -- at `n_threads=0`, which is what qseek defaults to,
  it runs 5.34 vs `TaskGroup`'s 3.05, losing even to the C extension's 3.61.
- Hand-rolled POSIX threads via `std.ffi` scaled worse still (2.4x vs 3.6x on
  4 threads): freshly spawned threads land on this class of CPU's efficiency
  cores, while a warm pool stays on the performance cores.
- The stable stdlib alone offers nothing: `std.algorithm.map` is serial
  (measured at 1.2x across 32 cores), leaving single-threaded at ~3.5x slower.

`parallelize` is the natural replacement the moment its high-thread-count
scheduling improves; switching back is a two-call-site change.

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

Moving off `async` (to `parallelize` or anything else synchronous) removes
the cause of workaround (1) entirely. Workaround (2) is about destruction
order rather than async, so re-check it rather than assuming it can go.
"""

from std.python import PythonObject, Python
from std.python.bindings import PythonModuleBuilder
from std.python.numpy import from_numpy_array
from std.python._cpython import PyThreadState
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
# GIL
# ===----------------------------------------------------------------=== #


struct GILReleased:
    """Drops the GIL for the duration of a `with` block.

    Mojo functions bound with `PythonModuleBuilder` are entered holding the
    GIL and keep it until they return, so a long compute would block every
    other Python thread -- including the asyncio event loop that
    `qseek.reduce` offloads to via `asyncio.to_thread`. The C extension this
    module replaced released it around the same loops with
    `Py_BEGIN_ALLOW_THREADS`; this is that, scoped.

    Mojo 1.0 has no ready-made guard for this (nothing GIL-related appears
    in the public stdlib index), but the primitive is reachable through the
    documented `Python.cpython()` handle.

    **Nothing inside the block may touch a `PythonObject`** -- that is what
    the GIL protects. Build and borrow every Python input before entering,
    and construct results after leaving. `__exit__` also runs when the block
    raises, so the GIL is always reacquired before the error propagates.
    """

    var state: Optional[Pointer[PyThreadState, MutUntrackedOrigin]]

    def __init__(out self):
        self.state = Python().cpython().PyEval_SaveThread()

    def __enter__(self):
        pass

    def __exit__(deinit self):
        Python().cpython().PyEval_RestoreThread(self.state)


# ===----------------------------------------------------------------=== #
# NumPy helpers
# ===----------------------------------------------------------------=== #


@always_inline
def get_thread_count(n_threads: Int) -> Int:
    return num_logical_cores() if n_threads <= 0 else n_threads


@always_inline
def borrow_1d[
    dtype: DType
](array: PythonObject) raises -> Tuple[
    Pointer[Scalar[dtype], MutUntrackedOrigin], Int
]:
    """Validate a 1-D NumPy array and borrow its buffer as (pointer, length).

    `from_numpy_array` does the checking natively -- dtype, rank and
    C-contiguity, the last mattering as much as the first because every
    kernel below walks the raw buffer with strides it computes itself, so a
    transposed or strided view would be read as if it were packed and
    silently give wrong numbers instead of failing.

    Its `Span` is then dropped to an untracked pointer: `Grid` stores these
    in heap arrays that outlive the `PythonObject` binding the span's origin
    is tied to, and hands them to tasks whose lifetimes Mojo does not yet
    track (see the module docstring). Keeping the array alive for as long as
    the pointer is used is therefore the caller's job -- `Grid`'s callers
    hold the Python list of traces/nodes for the whole call.
    """
    var span = from_numpy_array[dtype](array)
    var ptr = Pointer[Scalar[dtype], MutUntrackedOrigin](
        unsafe_from_address=Int(span.unsafe_ptr())
    )
    return (ptr, len(span))


@always_inline
def borrow_2d_f32(
    array: PythonObject, rows: Int, cols: Int, name: StaticString
) raises -> F32Ptr:
    """Validate a 2-D float32 array of shape (rows, cols) and borrow it.

    The 2-D counterpart to `borrow_1d`, needed only for `delay_sum`'s output
    array. `from_numpy_array` is 1-D only, so the shape and contiguity checks
    are spelled out here; this runs once per call, not per node.
    """
    var shape = array.shape
    if Int(py=array.ndim) != 2:
        raise Error(name, " must be a 2-D array")
    if Int(py=shape[0]) != rows or Int(py=shape[1]) != cols:
        raise Error(name, " must have shape (", rows, ", ", cols, ")")
    if String(array.dtype.char) != "f":
        raise Error(name, " must have dtype 'f'")
    if not array.flags["C_CONTIGUOUS"]:
        raise Error(name, " must be C-contiguous")
    return F32Ptr(unsafe_from_address=Int(py=array.ctypes.data))


# ===----------------------------------------------------------------=== #
# Trace / NodeStack
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

        # Unrolled: this is the hot loop (~98% of the module's work), and
        # unrolling measured ~3% faster single-threaded. It makes no
        # difference once threads saturate memory bandwidth, but costs
        # nothing there either.
        vectorize[SIMD_WIDTH, unroll_factor=4](n_samples, kernel)


@fieldwise_init
struct NodeStack(Copyable, Movable):
    """One grid node's delay-and-sum inputs: per-trace shifts and weights.

    Named `NodeStack` (not `Node`) to keep it distinct from qseek's octree
    `Node` -- the two are related (one `NodeStack` per octree node) but this
    one only carries what the kernels below need: which output slot it owns
    (`index`) and its per-trace shift/weight.

    `index` matters only where results are folded into caller-owned buffers
    across several calls (`delay_sum_reduce`); `delay_sum` and
    `delay_sum_snapshot` write row/element `i` for the `i`-th node passed.
    """

    var shifts: I32Ptr
    var weights: F32Ptr
    var index: Int32

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
    var nodes: Pointer[NodeStack, MutUntrackedOrigin]
    var n_nodes: Int


def resolve_window(
    grid: GridView, shift_range: PythonObject
) raises -> Tuple[Int32, Int32]:
    """Resolve (min_shift, max_shift), either from `shift_range` or by
    scanning every node's shifted traces for the window they span.

    A free function rather than a `Grid` method: it runs from inside
    `Grid.__init__`, before `self.min_shift`/`self.stack_size` exist, and
    Mojo forbids calling a method on a partly-initialized `self`. A
    `GridView` over the fields that *are* initialized by then is fine.
    """
    if shift_range is None:
        # Derive the window from the data: the span covered by every trace
        # once every node has shifted it.
        var min_shift = Int32.MAX
        var max_shift = Int32.MIN
        for i_node in range(grid.n_nodes):
            ref node = grid.nodes[unsafe_offset=i_node]
            for i_trace in range(grid.n_traces):
                ref trace = grid.traces[unsafe_offset=i_trace]
                var begin = Int32(trace.offset) + node.shifts.unsafe_load(
                    i_trace
                )
                min_shift = min(min_shift, begin)
                max_shift = max(max_shift, begin + Int32(trace.size))
        return (min_shift, max_shift)

    # Caller pinned the window, so skip the scan above -- it is
    # O(n_nodes * n_traces) and its result would only be discarded.
    if len(shift_range) != 2:
        raise Error(
            "shift_range argument must be tuple of two integers or None."
        )
    var min_shift = Int32(Int(py=shift_range[0]))
    var max_shift = Int32(Int(py=shift_range[1]))
    if max_shift <= min_shift:
        raise Error(
            "Invalid shift_range: max_shift must be greater than min_shift."
        )
    return (min_shift, max_shift)


struct Grid:
    """Validated, pointer-backed view over one call's inputs.

    Also resolves the result window, so callers read `min_shift` and
    `stack_size` off the grid rather than re-deriving them.
    """

    var traces: Pointer[Trace, MutUntrackedOrigin]
    var n_traces: Int
    var nodes: Pointer[NodeStack, MutUntrackedOrigin]
    var n_nodes: Int
    var min_shift: Int32
    var stack_size: Int

    def __init__(
        out self,
        traces: PythonObject,
        nodes: PythonObject,
        shift_range: PythonObject,
    ) raises:
        """Build from a list of trace-like and a list of node-like objects.

        Each trace needs `data` (float32, 1-D) and `offset` (int); each node
        needs `shifts` (int32, length n_traces), `weights` (float32, same
        length) and `index`. See `qseek.reduce`'s `TraceInput`/`NodeStack`,
        the `NamedTuple`s this is built for.

        Both are plain Python lists rather than the C extension's packed 2-D
        arrays plus a parallel offsets array and a node mask. Bookkeeping --
        which nodes are new, which are leaves -- stays on the Python side:
        callers pass the subset they want stacked, so there is no mask to
        parse and no row-striding, just one borrow per list item.
        """
        var n_traces = len(traces)
        if n_traces == 0:
            raise Error("Input traces must be a non-empty list")

        var n_nodes = len(nodes)
        if n_nodes == 0:
            raise Error("Number of nodes must be greater than zero")

        self.n_traces = n_traces
        self.n_nodes = n_nodes
        self.traces = unsafe_alloc[Trace](n_traces)
        self.nodes = unsafe_alloc[NodeStack](n_nodes)

        for i_trace in range(n_traces):
            var trace_obj = traces[i_trace]
            var data = borrow_1d[DType.float32](trace_obj.data)
            self.traces.unsafe_offset(i_trace).unsafe_write(
                Trace(data[0], data[1], Int(py=trace_obj.offset))
            )

        for i_node in range(n_nodes):
            var node_obj = nodes[i_node]
            var shifts = borrow_1d[DType.int32](node_obj.shifts)
            var weights = borrow_1d[DType.float32](node_obj.weights)
            if shifts[1] != n_traces or weights[1] != n_traces:
                raise Error(
                    "node.shifts and node.weights must have length n_traces"
                )
            self.nodes.unsafe_offset(i_node).unsafe_write(
                NodeStack(
                    shifts[0],
                    weights[0],
                    index=Int32(Int(py=node_obj.index)),
                )
            )

        var window = resolve_window(
            GridView(self.traces, self.n_traces, self.nodes, self.n_nodes),
            shift_range,
        )
        self.min_shift = window[0]
        self.stack_size = Int(window[1] - window[0])

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
def stack_block(
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
    Every node in `grid` is stacked -- callers that only want a subset (e.g.
    `delay_sum_reduce`'s incremental updates) pass a `Grid` built from just
    that subset rather than masking it out here.
    """
    for i_trace in range(grid.n_traces):
        var trace = grid.traces[unsafe_offset=i_trace].copy()
        for i_node in range(node_start, node_end):
            var node = grid.nodes[unsafe_offset=i_node].copy()
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
        stack_block(
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
    nodes: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    """Stack every node into its own row of a (n_nodes, stack_size) array.

    Row `i` holds `nodes[i]` -- positionally, so `NodeStack.index` is
    unused here (only `delay_sum_reduce` needs it). This is the whole
    per-node stack, so it is memory-hungry by design; production code wants
    `delay_sum_reduce`, which folds the same work into a running max.
    """
    var stack = kwargs.pop(String("stack"), PythonObject(None))
    var shift_range = kwargs.pop(String("shift_range"), PythonObject(None))
    var n_threads = Int(py=kwargs.pop(String("n_threads"), PythonObject(1)))

    var grid = Grid(traces, nodes, shift_range)
    var stack_size = grid.stack_size

    if stack is None:
        var np = Python.import_module("numpy")
        stack = np.zeros(
            Python.tuple(grid.n_nodes, stack_size),
            dtype=PythonObject("float32"),
        )

    var stack_data = borrow_2d_f32(stack, grid.n_nodes, stack_size, "stack")
    var n_chunks = max(1, min(get_thread_count(n_threads), grid.n_nodes))
    var min_shift = grid.min_shift

    with GILReleased():
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
                    min_shift,
                )
            )
        tg.wait()
        # Keep the Grid's allocations alive past the tasks; see module
        # docstring. Also keeps it inside the GIL-released scope, so its
        # `__deinit__` cannot free buffers a task is still reading.
        _ = grid^

    return Python.tuple(stack, Int(min_shift))


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

        stack_block(
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
            var node = grid.nodes[unsafe_offset=i_node].copy()
            update_running_max(
                stacks.unsafe_offset((i_node - block_start) * tile_size),
                tile_size,
                tile_start,
                node.index,
                stack_max,
                stack_max_idx,
            )


def delay_sum_reduce(
    traces: PythonObject,
    nodes: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    """Fold every node's stack into a running (max, argmax) over time.

    Callers doing incremental accumulation (reduce.py's `DelaySumReduce`)
    pass only the nodes not yet folded into `node_stack_max` /
    `node_stack_max_idx` -- `NodeStack.index` is what makes those buffers
    record the right node even though this call only sees a subset.
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
        node_max = np.full(
            stack_size,
            np.finfo(PythonObject("float32")).min,
            dtype=PythonObject("float32"),
        )
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
    var n_chunks = max(1, min(get_thread_count(n_threads), stack_size))
    var min_shift = grid.min_shift

    # One scratch allocation shared by every task, sliced per chunk:
    # allocating inside each task instead put a malloc large enough to be
    # mmap'd (and its page faults) on the hot path, which showed up as
    # run-to-run jitter.
    var max_tile = ceildiv(stack_size, n_chunks)
    var block = max(1, min(NODE_BLOCK, BLOCK_SCRATCH_FLOATS // max_tile))

    with GILReleased():
        var scratch = unsafe_alloc[Float32](n_chunks * block * max_tile)
        var tg = TaskGroup()
        for i_chunk in range(n_chunks):
            var bounds = chunk_bounds(i_chunk, n_chunks, stack_size)
            tg.create_task(
                reduce_tile(
                    grid.view(),
                    bounds[0],
                    bounds[1],
                    min_shift,
                    scratch.unsafe_offset(i_chunk * block * max_tile),
                    block,
                    stack_max,
                    stack_max_idx,
                )
            )
        tg.wait()
        scratch.unsafe_free()
        # Keep the Grid's allocations alive past the tasks; see module
        # docstring. Also keeps it inside the GIL-released scope, so its
        # `__deinit__` cannot free buffers a task is still reading.
        _ = grid^

    return Python.tuple(node_max, node_max_idx, Int(min_shift))


# ===----------------------------------------------------------------=== #
# delay_sum_snapshot
# ===----------------------------------------------------------------=== #


def delay_sum_snapshot(
    traces: PythonObject,
    nodes: PythonObject,
    var **kwargs: PythonObject,
) raises -> PythonObject:
    """`nodes` is the same node-list convention as `delay_sum_reduce`. The
    returned array is positionally aligned with `nodes` -- callers wanting
    only a subset (e.g. reduce.py's `get_snapshot(leaf_only=True)`) filter
    the list before calling rather than passing a mask.
    """
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
    var snapshot = np.zeros(grid.n_nodes, dtype=PythonObject("float32"))
    var snapshot_data = borrow_1d[DType.float32](snapshot)[0]
    var view = grid.view()
    var min_shift = grid.min_shift

    with GILReleased():
        for i_node in range(grid.n_nodes):
            var node = grid.nodes[unsafe_offset=i_node].copy()
            snapshot_data.unsafe_store(
                i_node, node.sample_at(view, index, min_shift)
            )
        _ = grid^

    return snapshot
