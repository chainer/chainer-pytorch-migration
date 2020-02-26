import chainer
import torch


def to_chainer_device(device):
    """Create a chainer device from a given torch device.

    Args:
        device (torch.device): Device to be converted.

    Returns:
        A ``torch.device`` object corresponding to the given input.
    """
    assert isinstance(device, torch.device)
    if device.type == 'cpu':
        return chainer.get_device('@numpy')
    if device.type == 'cuda':
        device_index = 0 if device.index is None else device.index
        return chainer.get_device('@cupy:{}'.format(device_index))
    raise ValueError('{} is not supported.'.format(device.type))
