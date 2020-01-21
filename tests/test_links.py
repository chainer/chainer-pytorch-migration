import chainer
import torch
import numpy

import chainer_pytorch_migration as cpm


def test_to_torch_module():
    model = torch.nn.Linear(3, 1)
    model.weight.data = torch.ones(1, 3)
    # Conversion
    chained = cpm.TorchModule(model)

    assert isinstance(chained.weight, chainer.Variable)
    assert isinstance(chained.bias, chainer.Variable)
    assert chained.weight.shape == (1, 3)
    assert chained.bias.shape == (1,)
    assert (chained.weight.array == numpy.ones((1, 3))).all()

    # Test memory sharing
    chained.weight.array[...] = numpy.arange(3).reshape((1, 3))
    assert (model.weight.data == torch.arange(3).reshape((1, 3))).all()


def test_to_torch_module_data_parallel():
    model = torch.nn.Linear(3, 1)
    model = torch.nn.DataParallel(model, device_ids=[0])
    model.module.weight.data = torch.ones(1, 3)
    # Conversion
    chained = cpm.TorchModule(model)

    assert isinstance(chained.wrapped_module.weight, chainer.Variable)
    assert isinstance(chained.wrapped_module.bias, chainer.Variable)
    assert chained.wrapped_module.weight.shape == (1, 3)
    assert chained.wrapped_module.bias.shape == (1,)
    assert (chained.wrapped_module.weight.array == numpy.ones((1, 3))).all()

    # # Test memory sharing
    chained.wrapped_module.weight.array[...] = numpy.arange(3).reshape((1, 3))
    assert (model.module.weight.data == torch.arange(3).reshape((1, 3))).all()
