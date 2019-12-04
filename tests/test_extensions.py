import chainer
import ignite
import torch
import unittest
import os

from ignite.engine import Events

import chainer_pytorch_migration.ignite
import chainer_pytorch_migration as cpm


def test_chainer_extensions():

    count = 0

    def dummy_extension(trainer):
        nonlocal count
        count += 1

    engine = ignite.engine.Engine(lambda engine, x: [])
    # We just create dummy models as we won't be utilizing them
    # We only want the training loop to call our extension
    model = torch.nn.Linear(128, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    optimizer.target = model
    cpm.ignite.add_trainer_extension(engine, optimizer, dummy_extension)
    engine.run([1, 2, 3], max_epochs=1)
    assert count == 3


class SnapshotBaseMixin(object):

    def setUp(self):
        self.engine = ignite.engine.Engine(lambda engine, x: [])
        self.engine.out = '.'
        self.model = torch.nn.Linear(128, 1)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=1.0, momentum=0.5)
        self.optimizer.target = self.model
        w = chainer.training.extensions.snapshot_writers.SimpleWriter()
        snapshot = chainer.training.extensions.snapshot(writer=w)

        cpm.ignite.add_trainer_extension(
            self.engine, self.optimizer, snapshot, trigger=(1, 'epoch'))

    def tearDown(self):
        snapshot_ch = os.path.join(self.engine.out, 'snapshot_iter_3')
        snapshot_pt = os.path.join(self.engine.out, 'snapshot_iter_3-torch')
        if os.path.exists(snapshot_ch):
            os.remove(snapshot_ch)
        if os.path.exists(snapshot_pt):
            os.remove(snapshot_pt)


class TestSnapshotSaveFile(SnapshotBaseMixin, unittest.TestCase):

    def test_save_file(self):
        self.engine.run([1, 2, 3], max_epochs=1)
        snapshot_ch = os.path.join(self.engine.out, 'snapshot_iter_3')
        snapshot_pt = os.path.join(self.engine.out, 'snapshot_iter_3-torch')
        assert os.path.exists(snapshot_ch)
        assert os.path.exists(snapshot_pt)


def compare_state_dicts(d1, d2):
    if type(d1) != type(d2):
        return False
    if isinstance(d1, torch.Tensor):
        return torch.equal(d1, d2)
    if type(d1) is dict:
        # Params just hold pointers, should not be restored
        return (d1.keys() == d2.keys()) and all([
            compare_state_dicts(d1[k], d2[k]) for k in d1 if k != 'params'])
    if type(d1) is list:
        return len(d1) == len(d2) and all(
            [compare_state_dicts(l1, l2) for l1, l2 in zip(d1, d2)])
    return d1 == d2


class TestSnapshotLoadFile(SnapshotBaseMixin, unittest.TestCase):

    def verify_snapshot_on_start(self, engine, model, optimizer):
        assert engine.state.epoch == self.engine.state.epoch
        assert engine.state.iteration == self.engine.state.iteration
        for p1, p2 in zip(model.parameters(), self.model.parameters()):
            assert torch.equal(p1, p2)
        assert compare_state_dicts(
            optimizer.state_dict(), self.optimizer.state_dict())

    def setup_model(self):
        self.engine.run([1, 2, 3], max_epochs=1)
        self.snapshot_ch = os.path.join(self.engine.out,
                                        'snapshot_iter_3')
        self.snapshot_pt = os.path.join(self.engine.out,
                                        'snapshot_iter_3-torch')
        assert os.path.exists(self.snapshot_ch)
        assert os.path.exists(self.snapshot_pt)

        # Create a new trainer, load the state and compare model and optimizer
        # params
        engine = ignite.engine.Engine(lambda engine, x: [])
        model = torch.nn.Linear(128, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        optimizer.target = model
        return engine, model, optimizer

    def test_load_file(self):
        engine, model, optimizer = self.setup_model()
        cpm.ignite.load_chainer_snapshot(engine, optimizer, 'snapshot_iter_3')
        # Need to defer state loading because of some ignite particularities
        engine.add_event_handler(Events.STARTED, self.verify_snapshot_on_start,
                                 model=model, optimizer=optimizer)
        engine.run([1, 2, 3], max_epochs=2)

    def test_load_file_torch(self):
        engine, model, optimizer = self.setup_model()
        cpm.ignite.load_chainer_snapshot(engine, optimizer,
                                         self.snapshot_ch, self.snapshot_pt)
        # Need to defer state loading because of some ignite particularities
        engine.add_event_handler(Events.STARTED, self.verify_snapshot_on_start,
                                 model=model, optimizer=optimizer)
        engine.run([1, 2, 3], max_epochs=2)

    def test_load_single_file_obj(self):
        engine, model, optimizer = self.setup_model()
        # Need to defer state loading because of some ignite particularities
        with open(self.snapshot_ch, "rb") as f:
            cpm.ignite.load_chainer_snapshot(engine, optimizer, f)
            engine.add_event_handler(Events.STARTED,
                                     self.verify_snapshot_on_start,
                                     model=model, optimizer=optimizer)
            engine.run([1, 2, 3], max_epochs=2)

    def test_load_both_file_obj(self):
        engine, model, optimizer = self.setup_model()
        # Need to defer state loading because of some ignite particularities
        with open(self.snapshot_ch, "rb") as f_ch:
            with open(self.snapshot_pt, "rb") as f_pt:
                cpm.ignite.load_chainer_snapshot(engine, optimizer,
                                                 f_ch, f_pt)
                engine.add_event_handler(Events.STARTED,
                                         self.verify_snapshot_on_start,
                                         model=model, optimizer=optimizer)
                engine.run([1, 2, 3], max_epochs=2)
