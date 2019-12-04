import chainer_pytorch_migration as cpm

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
