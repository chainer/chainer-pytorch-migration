import numpy

import chainer
import chainer_pytorch_migration as cpm


def _transform(in_data):
    img, label = in_data
    img = img - 0.5  # scale to [-0.5, 0.5]
    return img, label


def test_transform_dataset():
    dataset, _ = chainer.datasets.get_mnist()

    chainer_dataset = chainer.datasets.TransformDataset(dataset, _transform)
    cpm_dataset = cpm.TransformDataset(dataset, _transform)

    assert len(chainer_dataset) == len(cpm_dataset)

    for x, y in zip(chainer_dataset, cpm_dataset):
        x_data, x_label = x
        y_data, y_label = y
        numpy.testing.assert_array_equal(x_data, y_data)
        assert x_data.dtype == y_data.dtype
        assert x_label == y_label
