import chainer
from chainer.backends import cuda

from chainer_pytorch_migration import tensor


class TorchModule(chainer.Chain):

    """Chain that wraps PyTorch module.

    ``TorchModule`` wraps a :class:`torch.nn.Module` object with
    :class:`chainer.Link` interface. The module hierarchy is reproduced as a
    link hierarchy. The parameters and persistences of each link are views of
    the parameters and buffers of the corresponding module.

    This class does not provide ``forward`` implementation. To perform forward
    (and backward) propagations, use :attr:`module` directly. When the backprop
    is performed to compute the gradients, the gradient with respect to each
    parameter is automatically reflected to the corresponding link parameter.

    .. Note:
       For device transfer, only :meth:`to_cpu` and :meth:`to_gpu` keep track
       of the mapping. Currently, :meth:`to_device` breaks the mapping and
       makes the module and link diverge from each other.

    """
    def __init__(self, module):
        super().__init__()
        self._module = module

        with self.init_scope():
            for name, child in module.named_children():
                if name == 'module':
                    # DataParallel objects have the model stored as `module`
                    # causing a conflict.
                    name = 'wrapped_module'
                setattr(self, name, TorchModule(child))
            for name, param in module.named_parameters(recurse=False):
                ch_param = chainer.Parameter(tensor.asarray(param))
                setattr(self, name, ch_param)
                # Gradient computed at PyTorch side is automatically
                # synchronized to Chainer side with this hook.
                param.register_hook(_get_grad_setter(ch_param))
            for name, buffer in module.named_buffers(recurse=False):
                self.add_persistent(name, tensor.asarray(buffer))

    @property
    def module(self):
        """PyTorch module that this object wraps."""
        return self._module

    # TODO(beam2d): Fix Chainer to enable TorchModule to override to_device.
    def to_cpu(self):
        self.module.cpu()
        self._sync_from_torch()

        # This super call does not transfer the parameters and arrays, but is
        # needed to correctly change the metadata.
        super().to_cpu()

    def to_gpu(self, device=None):
        self.module.cuda(cuda.cupy.cuda.Device(device).id)
        self._sync_from_torch()

        # This super call does not transfer the parameters and arrays, but is
        # needed to correctly change the metadata.
        super().to_gpu(device)

    def _sync_from_torch(self):
        for child in self.children():
            child._sync_from_torch()
        for name, param in self.module.named_parameters(recurse=False):
            getattr(self, name).array = tensor.asarray(param)
        for name, buffer in self.module.named_buffers(recurse=False):
            setattr(self, name, tensor.asarray(buffer))


def _get_grad_setter(param):
    def hook(grad):
        param.grad = tensor.asarray(grad)
    return hook
