try:
    import cupy
    _cupy_import_error = None
except ImportError as e:
    _cupy_import_error = e
import torch


def use_mempool_in_cupy_malloc():
    _ensure_cupy()
    cupy.cuda.set_allocator(cupy.get_default_memory_pool().malloc)


def use_torch_in_cupy_malloc():
    _ensure_cupy()
    cupy.cuda.set_allocator(_torch_alloc)


def _ensure_cupy():
    if _cupy_import_error is not None:
        raise RuntimeError(
            'cupy is not available; import error is:\n{}', _cupy_import_error)


def _torch_alloc(size):
    device = cupy.cuda.Device().id
    tensor = torch.empty(size, dtype=torch.uint8, device=device)
    return cupy.cuda.MemoryPointer(
        cupy.cuda.UnownedMemory(tensor.data_ptr(), size, tensor), 0)
