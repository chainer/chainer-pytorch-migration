from chainer.backends import cuda
import numpy
import torch


def asarray(tensor):
    """Create an ndarray view of a given tensor.

    Args:
        tensor (torch.Tensor): Tensor to be converted.

    Returns:
        An ndarray view of ``tensor``. The returned array shares the underlying
        buffer with ``tensor``. The ownership is also shared, so the buffer is
        released only after both the original tensor and the returned ndarray
        view are gone.

    """
    dev_type = tensor.device.type
    if dev_type == 'cuda':
        dev_id = tensor.device.index
        cupy = cuda.cupy
        with cupy.cuda.Device(dev_id):
            # If the tensor is not allocated in torch (empty)
            # we just create a new one
            if tensor.data_ptr() == 0:
                return cupy.ndarray(
                    tuple(tensor.shape),
                    dtype=to_numpy_dtype(tensor.dtype))
            itemsize = tensor.element_size()
            storage = tensor.storage()
            memptr = cupy.cuda.MemoryPointer(
                cupy.cuda.UnownedMemory(
                    storage.data_ptr(), storage.size() * itemsize, tensor,
                ),
                tensor.storage_offset() * itemsize,
            )
            return cupy.ndarray(
                tuple(tensor.shape),
                dtype=to_numpy_dtype(tensor.dtype),
                memptr=memptr,
                strides=tuple(s * itemsize for s in tensor.stride()),
            )
    if dev_type == 'cpu':
        return tensor.detach().numpy()
    raise ValueError('tensor on device "{}" is not supported', dev_type)


def astensor(array):
    """Create a tensor view of a given ndarray.

    Args:
        array (numpy.ndarray or cupy.ndarray): Source array to make a view of.

    Returns:
        A :class:`torch.Tensor` view of ``array``. The returned tensor shares
        the buffer with ``array``. The ownership is also shared, so the buffer
        is released only after both the original array and the returned tensor
        view are gone.

    Note:
        If the array has negative strides, a copy is made
    """
    if array is None:
        raise TypeError('array cannot be None')

    # Torch does not support negative strides, make a implicit copy of the
    # array in such case
    if any(s < 0 for s in array.strides):
        array = array.copy()
    if isinstance(array, cuda.ndarray):
        # If the array is not allocated (empty)
        # we just create a new one
        if array.data.ptr == 0:
            return torch.empty(array.shape, dtype=to_torch_dtype(array.dtype))
        return torch.as_tensor(
            _ArrayWithCudaArrayInterfaceHavingStrides(array),
            device=array.device.id,
        )
    if isinstance(array, numpy.ndarray):
        return torch.from_numpy(array)
    raise TypeError('array of type {} is not supported'.format(type(array)))


# Workaround to avoid a bug in converting cupy.ndarray to torch.Tensor via
# __cuda_array_interface__. See: https://github.com/pytorch/pytorch/pull/24947
class _ArrayWithCudaArrayInterfaceHavingStrides:

    def __init__(self, array):
        self._array = array

    @property
    def __cuda_array_interface__(self):
        d = self._array.__cuda_array_interface__
        d['strides'] = self._array.strides
        return d


def to_numpy_dtype(torch_dtype):
    """Convert PyTorch dtype to NumPy dtype.

    Args:
        torch_dtype: PyTorch's dtype object.

    Returns:
        NumPy type object.

    """
    numpy_dtype = _torch_dtype_mapping.get(torch_dtype, None)
    if numpy_dtype is None:
        raise TypeError('{} does not have corresponding numpy dtype'.format(
            torch_dtype
        ))
    return numpy_dtype


def to_torch_dtype(numpy_dtype):
    """Convert NumPy dtype to PyTorch dtype.

    Args:
        numpy_dtype: NumPy's dtype object.

    Returns:
        PyTorch type object.

    """
    torch_dtype = _numpy_dtype_mapping.get(numpy_dtype, None)
    if torch_dtype is None:
        raise TypeError('{} does not have corresponding numpy dtype'.format(
            numpy_dtype
        ))
    return torch_dtype


_torch_dtype_mapping = {
    torch.bool: numpy.dtype('bool'),
    torch.uint8: numpy.dtype('uint8'),
    torch.int8: numpy.dtype('int8'),
    torch.int16: numpy.dtype('int16'),
    torch.int32: numpy.dtype('int32'),
    torch.int64: numpy.dtype('int64'),
    torch.float16: numpy.dtype('float16'),
    torch.float32: numpy.dtype('float32'),
    torch.float64: numpy.dtype('float64'),
    # Note: numpy does not have complex32
    # torch.complex32: numpy.dtype('complex32'),
    torch.complex64: numpy.dtype('complex64'),
    torch.complex128: numpy.dtype('complex128'),
}

_numpy_dtype_mapping = {v: k for k, v in _torch_dtype_mapping.items()}
