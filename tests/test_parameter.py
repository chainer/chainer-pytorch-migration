import chainer
import numpy
import pytest
import torch

import chainer_pytorch_migration as cpm


@pytest.mark.parametrize('shape', [(3, 2), (2, 0, 1)])
def test_chainer_parameter(shape):
    # initialized parameter
    arr = numpy.full(shape, 17, 'float32')
    chainer_param = chainer.Parameter(arr)

    # Conversion
    torch_param = cpm.ChainerParameter(chainer_param)

    assert isinstance(torch_param, torch.nn.Parameter)
    assert torch_param.shape == shape
    assert (torch_param.data.numpy() == numpy.full(shape, 17, 'float32')).all()

    # Test memory sharing
    new_arr = numpy.random.randint(-4, 4, shape)
    torch_param.data[...] = torch.tensor(new_arr.copy())
    assert (chainer_param.array == new_arr).all()


def test_chainer_parameter_uninitialized():
    # Uninitialized parameters are not supported
    chainer_param = chainer.Parameter()

    with pytest.raises(TypeError):
        cpm.ChainerParameter(chainer_param)


@pytest.mark.parametrize('shape', [(3, 2), (2, 0, 1)])
def test_chainer_parameter_grad_getter(shape):
    arr = numpy.full(shape, 17, 'float32')
    grad = numpy.full(shape, 9, 'float32')
    chainer_param = chainer.Parameter(arr)
    chainer_param.grad = grad.copy()

    # Conversion
    torch_param = cpm.ChainerParameter(chainer_param)

    # Getter
    torch_grad = torch_param.grad

    assert isinstance(torch_grad, torch.Tensor)
    assert (torch_grad.numpy() == grad).all()

    # Test memory sharing
    new_arr = numpy.random.randint(-4, 4, shape)
    torch_grad[...] = torch.tensor(new_arr.copy())
    assert (chainer_param.grad == new_arr).all()


@pytest.mark.parametrize('shape', [(3, 2), (2, 0, 1)])
def test_chainer_parameter_grad_setter(shape):
    arr = numpy.full(shape, 17, 'float32')
    chainer_param = chainer.Parameter(arr)

    # Conversion
    torch_param = cpm.ChainerParameter(chainer_param)
    # Initialize grad
    torch_param.requires_grad = True
    optimizer = torch.optim.SGD([torch_param], lr=0.01, momentum=0.9)
    optimizer.zero_grad()

    # Setter
    grad = torch.full(shape, 9, dtype=torch.float32)
    torch_param.grad = grad
    numpy.testing.assert_array_equal(grad, torch_param.grad)


def test_to_chainer_parameters():
    # initialized parameter
    a_arr = numpy.ones((3,  2), 'float32')
    a_chainer_param = chainer.Parameter(a_arr)
    # 0-size parameter
    b_arr = numpy.ones((2, 0, 1), 'float32')
    b_chainer_param = chainer.Parameter(b_arr)

    link = chainer.Link()
    with link.init_scope():
        link.a = a_chainer_param
        link.b = b_chainer_param

    # Conversion
    torched = cpm.LinkAsTorchModel(link)
    params = list(torched.parameters())
    assert len(params) == 2
    assert isinstance(params[0], torch.nn.Parameter)
    assert isinstance(params[1], torch.nn.Parameter)
    assert params[0].shape == (3, 2)
    assert params[1].shape == (2, 0, 1)
    assert (params[0].data.numpy() == numpy.ones((3, 2))).all()

    # Test memory sharing
    params[0].data[...] = torch.tensor(numpy.arange(6).reshape((3, 2)))
    assert (a_chainer_param.array == numpy.arange(6).reshape((3, 2))).all()


def test_to_chainer_parameters_uninitialized():
    # Uninitialized parameters are not supported
    a_chainer_param = chainer.Parameter()

    link = chainer.Link()
    with link.init_scope():
        link.a = a_chainer_param

    with pytest.raises(TypeError):
        torched = cpm.LinkAsTorchModel(link)
        torched.parameters()


def test_state_dict():
    a_arr = numpy.ones((3,  2), 'float32')
    a_chainer_param = chainer.Parameter(a_arr)
    # 0-size parameter
    b_arr = numpy.ones((2, 0, 1), 'float32')
    b_chainer_param = chainer.Parameter(b_arr)

    link = chainer.Link()
    with link.init_scope():
        link.a = a_chainer_param
        link.b = b_chainer_param

    torched = cpm.LinkAsTorchModel(link)
    state_dict = torched.state_dict()
    assert '/a' in state_dict
    numpy.testing.assert_array_equal(a_arr, state_dict['/a'].detach())
    assert '/b' in state_dict
    numpy.testing.assert_array_equal(b_arr, state_dict['/b'].detach())


def test_named_params():
    a_arr = numpy.ones((3,  2), 'float32')
    a_chainer_param = chainer.Parameter(a_arr)
    # 0-size parameter
    b_arr = numpy.ones((2, 0, 1), 'float32')
    b_chainer_param = chainer.Parameter(b_arr)

    link = chainer.Link()
    with link.init_scope():
        link.a = a_chainer_param
        link.b = b_chainer_param

    torched = cpm.LinkAsTorchModel(link)
    n_params = dict(torched.named_parameters())
    assert '/a' in n_params
    numpy.testing.assert_array_equal(a_arr, n_params['/a'].detach())
    assert '/b' in n_params
    numpy.testing.assert_array_equal(b_arr, n_params['/b'].detach())
