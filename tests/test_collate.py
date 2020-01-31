import numpy
import torch
import pytest

import chainer_pytorch_migration.ignite as cpm_ignite


@pytest.mark.parametrize(
    'data, batch_size',
    [([], 1), (list(range(10)), 1), (list(range(100)), 10)])
def test_collate(data, batch_size):
    collate = cpm_ignite.collate_to_array
    dl = torch.utils.data.DataLoader(
        data, collate_fn=collate, batch_size=batch_size)
    for i, x in enumerate(dl):
        for e in x:
            assert isinstance(e, numpy.ndarray)
        expected = [
            numpy.array(e) for e in data[i * batch_size:(i + 1) * batch_size]]
        numpy.testing.assert_array_equal(x, expected)
