import chainer
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

def test_to_torch_device_cpu():
    device = chainer.get_device('@numpy')
    torch_device = cpm.to_torch_device(device)
    assert torch_device.type == 'cpu'

def test_to_torch_device_gpu():
    device = chainer.get_device('@cupy:0')
    torch_device = cpm.to_torch_device(device)
    assert torch_device.type == 'cuda'
    assert torch_device.index == 0

def test_to_torch_device_gpu_0():
    device = chainer.get_device('@cupy:1')
    torch_device = cpm.to_torch_device(device)
    assert torch_device.type == 'cuda'
    assert torch_device.index == 1
