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


def test_link_as_torch_model():
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


def test_link_as_torch_model_nested():
    dtype = numpy.float32

    # l2: MyLink2 := p2 * l1(x)
    #   - p2
    #   - l1: MyLink1 := p1 * x
    #     - p1
    class MyLink1(chainer.Link):
        def __init__(self):
            super().__init__()
            with self.init_scope():
                self.p1 = chainer.Parameter(numpy.array([2], dtype))
        def forward(self, x1):
            return self.p1 * x1

    class MyLink2(chainer.Chain):
        def __init__(self):
            super().__init__()
            with self.init_scope():
                self.p2 = chainer.Parameter(numpy.array([3], dtype))
                self.l1 = MyLink1()
        def forward(self, x2):
            return self.p2 * self.l1(x2)

    # Dummy optimizer that always writes the constant value 9.
    class MyOptim(torch.optim.Optimizer):
        def __init__(self, params):
            super().__init__(params, {})
        def step(self):
            for group in self.param_groups:
                for param in group['params']:
                    param.data[...] = 9

    link = MyLink2()
    module = cpm.LinkAsTorchModel(link)
    assert isinstance(module.p2, torch.nn.Parameter)
    assert isinstance(module.l1, torch.nn.Module)
    assert isinstance(module.l1.p1, torch.nn.Parameter)
    assert len(list(module.parameters(recurse=False))) == 1
    assert len(list(module.l1.parameters(recurse=False))) == 1
    assert len(list(module.parameters(recurse=True))) == 2

    optimizer = MyOptim(module.parameters())
    x = numpy.array([4], dtype)

    # Forward
    y = module(x)
    assert isinstance(y, torch.Tensor)
    numpy.testing.assert_array_equal(y.detach().numpy(), [24])

    # Backward
    y.backward()

    numpy.testing.assert_array_equal(module.l1.p1.grad.detach().numpy(), [12])
    numpy.testing.assert_array_equal(module.p2.grad.detach().numpy(), [8])

    # Optimizer step
    optimizer.step()

    numpy.testing.assert_array_equal(link.p2.array, [9])
    numpy.testing.assert_array_equal(link.l1.p1.array, [9])
    numpy.testing.assert_array_equal(module.p2.detach().numpy(), [9])
    numpy.testing.assert_array_equal(module.l1.p1.detach().numpy(), [9])


def test_link_as_torch_model_uninitialized():
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
    assert 'a' in state_dict
    numpy.testing.assert_array_equal(a_arr, state_dict['a'].detach())
    assert 'b' in state_dict
    numpy.testing.assert_array_equal(b_arr, state_dict['b'].detach())


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
    assert 'a' in n_params
    numpy.testing.assert_array_equal(a_arr, n_params['a'].detach())
    assert 'b' in n_params
    numpy.testing.assert_array_equal(b_arr, n_params['b'].detach())
