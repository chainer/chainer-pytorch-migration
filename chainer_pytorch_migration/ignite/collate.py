import torch

from chainer_pytorch_migration import tensor


def collate_to_array(batch):
    data = torch.utils.data._utils.collate.default_collate(batch)
    return [tensor.asarray(x) for x in data]
