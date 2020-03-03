import chainer
import torch


def to_chainer_device(device):
    """Create a chainer device from a given torch device.

    Args:
        device (torch.device): Device to be converted.

    Returns:
        A ``chainer.device`` object corresponding to the given input.
    """
    if not isinstance(device, torch.device):
        raise TypeError('The argument should be torch device.')
    if device.type == 'cpu':
        return chainer.get_device('@numpy')
    if device.type == 'cuda':
        device_index = 0 if device.index is None else device.index
        return chainer.get_device('@cupy:{}'.format(device_index))
    raise ValueError('{} is not supported.'.format(device.type))


def to_torch_device(device):
    """Create a torch device from a given chainer device.

    Args:
        device (chainer.Device): Device to be converted.

    Returns:
        A ``torch.device`` object corresponding to the given input.
    """
    if not isinstance(device, chainer.backend.Device):
        raise TypeError('The argument should be chainer device.')
    if device.name == '@numpy':
        return torch.device('cpu')
    if device.name.startswith('@cupy:'):
        cuda_device_index = int(device.name.split(':')[1])
        return torch.device('cuda', cuda_device_index)
    raise ValueError('{} is not supported.'.format(device.name))
