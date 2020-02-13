import chainer_pytorch_migration as cpm
import torch
from torch import nn


class LinkAsTorchModel(nn.Module):

    '''Converts a Chainer Link to a PyTorch module.

    The parameters of the link are automatically
    wrapped using `ChainerParameter` and added
    to the module as its own parameters.

    Args:
        link (:class:`chainer.Link`): A link. Must have been initialized.
    '''

    def __init__(self, link):
        super(LinkAsTorchModel, self).__init__()
        for n, p in sorted(link.namedparams()):
            self.__setattr__(n, ChainerParameter(p))
        self.link = link

    def forward(self, *input):
        # The computation graph should be done in Chainer.
        # Forward converts the input tensors to numpy/cupy arrays
        # as accepted by Chainer.
        # The return value should be a tensor as well.
        input = [cpm.tensor.asarray(x) if isinstance(x, torch.Tensor)
                 else x for x in input]
        return cpm.tensor.astensor(self.link.forward(*input).array)


class ChainerParameter(nn.Parameter):

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

    def __new__(cls, param):
        return super().__new__(cls, cpm.astensor(param.array))

    def __init__(self, param):
        super().__init__()
        self._param = param

    @property
    def grad(self):
        # TODO(hvy): Cache constructed `torch.Tensor`.
        if self._param.grad is not None:
            return cpm.astensor(self._param.grad)
        else:
            return None

    @grad.setter
    def grad(self, g):
        if self._param.grad is not None:
            self.grad[...] = g
        else:
            self._param.grad = cpm.asarray(g)
