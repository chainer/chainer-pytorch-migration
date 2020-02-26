import torch

import chainer_pytorch_migration as cpm


def test_to_chainer_device_cpu():
    device = torch.device('cpu')
    chainer_device = cpm.to_chainer_device(device)
    assert chainer_device.name == '@numpy'

def test_to_chainer_device_gpu():
    device = torch.device('cuda')
    chainer_device = cpm.to_chainer_device(device)
    assert chainer_device.name == '@cupy:0'

def test_to_chainer_device_gpu_0():
    device = torch.device('cuda:0')
    chainer_device = cpm.to_chainer_device(device)
    assert chainer_device.name == '@cupy:0'

def test_to_chainer_device_gpu_1():
    device = torch.device('cuda:1')
    chainer_device = cpm.to_chainer_device(device)
    assert chainer_device.name == '@cupy:1'
