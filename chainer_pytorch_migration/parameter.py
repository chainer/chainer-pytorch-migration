import chainer
import torch

import chainer_pytorch_migration as cpm


def _named_children(link):
    assert isinstance(link, chainer.Link)
    if isinstance(link, chainer.Chain):
        for name in link._children:
            yield name, getattr(link, name)


def _named_params(link):
    assert isinstance(link, chainer.Link)
    for name in link._params:
        yield name, getattr(link, name)


class LinkAsTorchModel(torch.nn.Module):

    '''Converts a Chainer Link to a PyTorch module.

    The parameters of the link are automatically
    wrapped using `ChainerParameter` and added
    to the module as its own parameters.

    Args:
        link (:class:`chainer.Link`): A link. Must have been initialized.
    '''

    def __init__(self, link):
        super().__init__()
        uninitialized_params = [
            n for n, p in sorted(_named_params(link)) if p.array is None]
        if uninitialized_params:
            raise RuntimeError(
                'Link with uninitialized parameters cannot be wrapped with '
                'LinkAsTorchModel. '
                'Please initialize parameters before wrapping, by feeding a '
                'dummy batch to the Chainer model, for example. '
                'Uninitialized params: [{}]'.format(
                    ', '.join(repr(n) for n in uninitialized_params)))

        for name, child in _named_children(link):
            child_module = LinkAsTorchModel(child)
            setattr(self, name, child_module)
        for name, param in sorted(_named_params(link)):
            setattr(self, name, ChainerParameter(param))

        self.link = link

    def forward(self, *input):
        # The computation graph should be done in Chainer.
        # Forward converts the input tensors to numpy/cupy arrays
        # as accepted by Chainer.
        # The return value should be a tensor as well.
        input = [cpm.tensor.asarray(x) if isinstance(x, torch.Tensor)
                 else x for x in input]
        outputs = self.link.forward(*input)
        ret = self.__as_tensor(outputs)
        return ret

    def __as_tensor(self, value):
        if isinstance(value, tuple):
            return tuple(self.__as_tensor(x) for x in value)
        if isinstance(value, list):
            return [self.__as_tensor(x) for x in value]
        if isinstance(value, chainer.Variable):
            return _ChainerTensor(value)
        return value


class Optimizer(torch.optim.Optimizer):

    def __init__(self, base_optimizer):
        assert isinstance(base_optimizer, torch.optim.Optimizer)
        super().__init__(base_optimizer.param_groups, base_optimizer.defaults)
        self._base_optimizer = base_optimizer

    def __getattr__(self, name):
        if name in ('step', 'zero_grad', '_base_optimizer'):
            return object.__getattribute__(self, name)
        return getattr(self._base_optimizer, name)

    def step(self, closure=None):
        for param_group in self._base_optimizer.param_groups:
            for param in param_group['params']:
                assert isinstance(param, ChainerParameter)
                param.grad.copy_(cpm.astensor(param._param.grad))
        self._base_optimizer.step(closure)

    def zero_grad(self):
        self._base_optimizer.zero_grad()
        for param_group in self._base_optimizer.param_groups:
            for param in param_group['params']:
                assert isinstance(param, ChainerParameter)
                param._param.zerograd()


class _ChainerTensor(torch.Tensor):
    '''
    Torch tensor from which backprop can be performed.
    '''
    def __new__(cls, variable):
        assert isinstance(variable, chainer.Variable)
        obj = cpm.astensor(variable.array)
        obj.__class__ = cls
        return obj

    def __init__(self, variable):
        self._variable = variable

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        assert retain_graph is None or retain_graph == False  # True not supported
        assert self._variable is not None

        var = self._variable
        if gradient is not None:
            var.grad = cpm.tensor.asarray(gradient)
        var.backward(
            enable_double_backprop=create_graph,
        )

    def zero_(self):
        super().zero_()
        self._variable.array[...] = 0


class ChainerParameter(torch.nn.Parameter):

    '''Wraps a Chainer parameter for use with a PyTorch optimizer.

    It is used to share the data, and more importantly, the gradient memory
    buffer between Chainer and PyTorch, since :class:`chainer.Parameter.grad`
    may be reassigned a new buffer after each backward. Computational graphs
    must be constructed and backpropagated through on the Chainer-side.

    Args:
        param (:class:`chainer.Parameter`): A parameter to convert.
    Returns:
        A :class:`ChainerParameter`.
    '''

    __grad = None

    def __new__(cls, param):
        return super().__new__(cls, cpm.astensor(param.array))

    def __init__(self, param):
        super().__init__()
        self._param = param

    @property
    def grad(self):
        if self.__grad is None:
            if self._param.grad is not None:
                self.__grad = _ChainerTensor(self._param.grad_var)
        return self.__grad

    @grad.setter
    def grad(self, g):
        if self._param.grad is not None:
            self.grad[...] = g
        else:
            self._param.grad = cpm.asarray(g)

    def zero_(self):
        super().zero_()
        self._param.cleargrad()
        self._param.array[...] = 0
        self.__grad = None
