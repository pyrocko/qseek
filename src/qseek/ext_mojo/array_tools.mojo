from python import PythonObject, Python
from python.bindings import PythonModuleBuilder
from python.python import CPython
import math
from memory import UnsafePointer
from memory.memory import memset
import time
from os import abort
from sys.info import simdwidthof, num_physical_cores
from sys.intrinsics import masked_store
from algorithm.functional import vectorize, parallelize


alias simd_width = simdwidthof[DType.float32]()


@export
fn PyInit_array_tools() -> PythonObject:
    try:
        var m = PythonModuleBuilder("array_tools")
        m.def_function[argmax_masked](
            "argmax_masked",
            docstring=(
                "Find the index of the maximum value in each column of a 2D"
                " array, ignoring masked values."
            ),
        )
        return m.finalize()
    except e:
        return abort[PythonObject](
            String("error creating Python Mojo module:", e)
        )


fn argmax_masked(
    array: PythonObject,
    mask: PythonObject,
) raises -> PythonObject:
    cpython = CPython()
    n_threads = num_physical_cores()
    np = Python.import_module("numpy")

    if array.ndim != 2:
        raise "Input array must be 2-dimensional"
    if array.dtype != np.float32:
        raise "Input array must be of type float32"
    if mask.dtype != np.bool_:
        raise "Mask must be of type bool"

    n_nodes = Int(array.shape[0])
    n_samples = Int(array.shape[1])

    if n_nodes == 0 or n_samples == 0:
        raise "Input array must have positive dimensions"
    if mask.size != n_nodes:
        raise "Mask size must match the number of nodes in the array"

    array_data = array.ctypes.data.unsafe_get_as_pointer[DType.float32]()
    mask_data = mask.ctypes.data.unsafe_get_as_pointer[DType.bool]()

    result_values = np.full(
        n_samples, np.finfo(np.float32).min, dtype=np.float32
    )
    result_values_data = result_values.ctypes.data.unsafe_get_as_pointer[
        DType.float32
    ]()

    result_indices = np.empty(n_samples, dtype=np.intp)
    result_indices_data = result_indices.ctypes.data.unsafe_get_as_pointer[
        DType.uint64
    ]()

    state = cpython.PyEval_SaveThread()

    offsets = UnsafePointer[UInt64].alloc(n_samples)
    for idx in range(n_samples):
        (offsets + idx).init_pointee_copy(idx)

    # Parallelize lead to a crash, waiting for fix
    # @parameter
    # fn sample_block(thread_num: Int):
    #     start_sample = (thread_num * n_samples) // n_threads
    #     end_sample = ((thread_num + 1) * n_samples) // n_threads

    for i_node in range(n_nodes):
        if not mask_data[i_node]:
            continue

        @parameter
        fn vectorize_max[width: Int](i_sample: Int):
            idx = i_sample
            value_vec = array_data.load[width=width](i_node * n_samples + idx)
            max_old_vec = result_values_data.load[width=width](idx)

            max_new_vec = max(value_vec, max_old_vec)
            updated_elements = max_old_vec != max_new_vec

            result_values_data.store(idx, max_new_vec)
            new_indices = masked_store(
                SIMD[DType.uint64, width](i_node),
                result_indices_data + idx,
                mask=updated_elements,
            )

        vectorize[vectorize_max, simdwidthof[DType.float32]()](n_samples)

    offsets.free()

    cpython.PyEval_RestoreThread(state)
    return Python.tuple(result_indices, result_values)
