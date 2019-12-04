import cupy
import numpy
import pytest
import torch

from chainer_pytorch_migration import tensor


def test_asarray_cpu():
    t = torch.arange(5, dtype=torch.float32)
    a = tensor.asarray(t)
    assert isinstance(a, numpy.ndarray)
    a += 1
    numpy.testing.assert_array_equal(a, t.numpy())


def test_asarray_gpu():
    t = torch.arange(5, dtype=torch.float32, device='cuda')
    a = tensor.asarray(t)
    assert isinstance(a, cupy.ndarray)
    a += 1
    numpy.testing.assert_array_equal(a.get(), t.cpu().numpy())


def test_astensor_cpu():
    a = numpy.arange(5, dtype=numpy.float32)
    t = tensor.astensor(a)
    assert isinstance(t, torch.Tensor)
    t += 1
    numpy.testing.assert_array_equal(a, t.numpy())


def test_astensor_gpu():
    a = cupy.arange(5, dtype=cupy.float32)
    t = tensor.astensor(a)
    assert isinstance(t, torch.Tensor)
    t += 1
    numpy.testing.assert_array_equal(a.get(), t.cpu().numpy())


def test_astensor_negative_stride():
    a = numpy.array([1, 2, 3])
    a = a[::-1]
    t = tensor.astensor(a)
    numpy.testing.assert_array_equal(a, t.numpy())


def test_asarray_empty_cpu():
    t = torch.tensor([], dtype=torch.float32)
    a = tensor.asarray(t)


def test_asarray_empty_gpu():
    t = torch.tensor([], dtype=torch.float32, device='cuda')
    a = tensor.asarray(t)

def test_astensor_empty_cpu():
    a = numpy.array([], dtype=numpy.float32)
    t = tensor.astensor(a)

def test_astensor_empty_gpu():
    a = cupy.array([], dtype=cupy.float32)
    t = tensor.astensor(a)
    assert isinstance(t, torch.Tensor)
    t += 1
    numpy.testing.assert_array_equal(a.get(), t.cpu().numpy())




@pytest.mark.parametrize('dtype', [
    'bool',
    'uint8', 'int8', 'int16', 'int32', 'int64',
    'float16', 'float32', 'float64',
    'complex64', 'complex128',
])
def test_to_numpy_dtype(dtype):
    torch_dtype = getattr(torch, dtype)
    numpy_dtype = numpy.dtype(dtype)
    assert tensor.to_numpy_dtype(torch_dtype) == numpy_dtype
