import time
import six
import os

import chainer
import torch

import chainer_pytorch_migration as cpm
from chainer.training import trigger as trigger_module
from ignite.engine import Events, Engine

# Torch computational graph here
# https://gist.github.com/wangg12/f11258583ffcc4728eb71adc0f38e832I
# Make our own extensions for the nocompat ones and
# reroute them?


"""
Currently supported extensions
+ExponentialShift (optimizer must be None)
+FailOnNonNumber
+InverseShift (optimizer must be None)
+LinearShift (optimizer must be None)
+LogReport
+MicroAverage (must be registered before LogReport)
+MultistepShift
+ParameterStatistics
+PlotReport
+PolynomialShift (optimizer must be None)
+PrintReport
+ProgressBar
+SnapshotWriters
+StepShift (optimizer must be None)
+observe_lr (observe_value)
+VariableStatisticsPlot
+WarmupShift


Not working
+ComputationalGraph
+Evaluator
+unchain_variables
"""


# In case multiple engines are used?
engines = {}


def add_trainer_extension(engine, optimizer, extension,
                          name=None, trigger=None, priority=None, **kwargs):

    """Function to register a `chainer.training.Extension` in a
       `ignite.Engine` trainer.

    Args:
        engine (:class:`ignite.Engine`): The ignite trainer object to which
               the extension will be associated.
        optimizer: (:class: `torch.optim.Optimizer`): A Torch
               with the .target attribute set to the model
        extension: (:class: `chainer.training.Extension`): Chainer extension
               to be executed.
        trigger (tuple or Trigger): Trigger object that determines when to
                invoke the extension. If it is ``None``, ``extension.trigger``
                is used instead. If it is ``None`` and the extension does not
                have the trigger attribute, the extension is triggered at every
                iteration by default. If the trigger is not callable, it is
                passed to :class:`IntervalTrigger` to build an interval
                trigger.
        priority (int): Invocation priority of the extension. Extensions
                are invoked in the descending order of priorities in each
                iteration. If this is ``None``, ``extension.priority`` is used
                instead.

    """
    if isinstance(extension, (
            chainer.training.extensions.DumpGraph,
            chainer.training.extensions.Evaluator,
            chainer.training.extensions.unchain_variables)):
        raise ValueError('Extension {} is not supported in cpm'.format(
            extension.__class__.__name__))
    if not id(engine) in engines:
        engines[id(engine)] = ExtensionTrainerAdapter(engine, optimizer)

    adapter = engines[id(engine)]
    adapter._chainer_trainer.extend(extension, name, trigger, priority,
                                    **kwargs)


def load_chainer_snapshot(engine, optimizer, snapshot_file,
                          snapshot_file_torch=None):
    """Function to load a torch/chainer combined snapshot
       using the cpm interface

    Args:
        engine (:class:`ignite.Engine`): The ignite trainer object to which
               the extension will be associated.
        optimizer: (:class: `torch.optim.Optimizer`): A Torch
               with the .target attribute set to the model
        extension: (:class: `chainer.training.Extension`): Chainer extension
               to be executed.
        snapshot_file (str or file-like): Target chainer snapshot
               obtained with the `chainer.extensions.snapshot`
               ExtensionTrainerAdapter loaded through the cpi tools
        snapshot_file_torch (str or file-like): Target torch snapshot.
               If not given, torch data would be loaded from
               "`snapshot_file`-torch".

    """
    if not id(engine) in engines:
        engines[id(engine)] = ExtensionTrainerAdapter(engine, optimizer)

    adapter = engines[id(engine)]
    if snapshot_file_torch is None:
        # if the torch snapshot is not given, we pass the filename
        # of chainer snapshot and let the ExtensionUpdaterAdapter to generate
        # the torch snapshot name
        if isinstance(snapshot_file, six.string_types):
            adapter.torch_snapshot = snapshot_file
        else:
            adapter.torch_snapshot = snapshot_file.name
        adapter.torch_snapshot += '-torch'
    else:
        adapter.torch_snapshot = snapshot_file_torch

    # Need to defer state loading because of some ignite particularities
    @engine.on(Events.STARTED)
    def set_load_snapshot_on_start(engine):
        chainer.serializers.load_npz(snapshot_file, adapter)


class ExtensionUpdaterAdapter(object):

    """Bridge between the extensions and `ignite.Engine`

    Keeps tracking of the current training status and allows
    the extensions to retrieve it using the same API than
    the Chainer updaters

    """
    def __init__(self, engine, optimizer):
        self.engine = engine
        self._optimizers = {'main': optimizer}

    @property
    def iteration(self):
        return self.engine.state.iteration

    @property
    def epoch(self):
        return self.engine.state.epoch - 1

    @property
    def epoch_detail(self):
        epoch_size = len(self.engine.state.dataloader)
        return self.iteration/epoch_size

    def get_optimizer(self, name):
        return self._optimizers[name]

    def get_all_optimizers(self):
        return self._optimizers

    def connect_trainer(self, trainer):
        pass

    def serialize(self, serializer, state):

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.serialize(serializer['optimizer:' + name], state)

        if isinstance(serializer, chainer.serializer.Serializer):
            state['iteration'] = self.engine.state.iteration
            state['epoch_length'] = self.engine.state.epoch_length
        elif isinstance(serializer, chainer.serializer.Deserializer):
            self.engine.state.iteration = state['iteration']
            self.engine.state.epoch = (state['iteration']
                                       // state['epoch_length'])


class ExtensionTrainerAdapter(object):

    """Bridge between the extensions and `ignite.Engine`

    Manages the extensions by using a dummy `chainer.training.Trainer`
    It provides a chainer Trainer compatible API so that the extensions
    can interact with the `ignite.Engine` without modifications.

    This class registers several handlers on ignite and forces the order
    of handler execution so that user defined ignite handlers are executed
    before the chainer extensions.

    """
    def __init__(self, engine, optimizer):

        self.engine = engine
        engine.run = self.pre_run

        self.optimizer = ExtensionOptimizerAdapter(optimizer)
        self.updater = ExtensionUpdaterAdapter(engine, self.optimizer)

        self.max_epochs = 0
        self.stop_trigger = None
        self.observation = {}
        self.cm = None

        self._start_time = 0
        self.out = getattr(engine, 'out', 'result')
        if not os.path.exists(self.out):
            os.makedirs(self.out)
        self.snapshot_file = None

        # We hold a chainer.Trainer dummy object to deal with all
        # the extensions registration mechanism and reporter population
        self._chainer_trainer = chainer.training.Trainer(self.updater)
        # The reporter has several observers associated (links)
        self.reporter = self._chainer_trainer.reporter

        self.set_ignite_handlers()

    @property
    def is_before_training(self):
        return self.updater.iteration == 0

    @property
    def elapsed_time(self):
        return time.time()-self._start_time

    def set_ignite_handlers(self):

        # Set a handler that sets the reporter scope on every iteration
        @self.engine.on(Events.ITERATION_STARTED)
        def set_reporter_on_iter(engine):
            self.observation = {}
            self.cm = self.reporter.scope(self.observation)
            self.cm.__enter__()

        @self.engine.on(Events.STARTED)
        def set_training_started(engine):
            # self._is_before_training = True
            self._start_time = time.time()
            self.start_extensions()
            # Make all the next
            # handlers to be executed after user defined ones
            @self.engine.on(Events.ITERATION_COMPLETED)
            def run_extensions_on_iter(engine):
                self.run_extensions()

            # This should be the last extension to be run
            @self.engine.on(Events.ITERATION_COMPLETED)
            def close_reporter_on_iter(engine):
                self.cm.__exit__(None, None, None)

    def start_extensions(self):
        exts = self._chainer_trainer._extensions
        extension_order = sorted(
            exts.keys(),
            key=lambda name: exts[name].priority, reverse=True)
        self.extensions = [(name, exts[name])
                           for name in extension_order]

        # invoke initializer of each extension
        for _, entry in self.extensions:
            initializer = getattr(entry.extension, 'initialize', None)
            finished = getattr(entry.trigger, 'finished', False)
            if initializer and not finished:
                initializer(self)

        # call extensions before training loop
        self.observation = {}
        if chainer.__version__ > "7.0.0b2":
            # call_before_training only works after 7.0.0b3
            with self.reporter.scope(self.observation):
                for name, entry in self.extensions:
                    if entry.call_before_training:
                        entry.extension(self)

    def run_extensions(self):
        for name, entry in self.extensions:
            if entry.trigger(self):
                ext = entry.extension
                self.cur_ext = (name, ext)
                entry.extension(self)

    def extend(self, extension):
        self.extensions.append(extension)

    def get_extension(self, class_name):
        return self._chainer_trainer.get_extension(class_name)

    def pre_run(self, data, max_epochs=1):
        # Method interception to capture the max_epochs
        # max_epochs is never saved in the Engine class
        self.max_epochs = max_epochs
        self.stop_trigger = trigger_module.get_trigger((max_epochs, 'epoch'))
        Engine.run(self.engine, data, max_epochs)

    def serialize(self, serializer):

        # Lets save torch objects using torch interface
        if isinstance(serializer, chainer.serializer.Serializer):
            name, ext = self.cur_ext
            if type(ext).__name__ == '_MultiNodeSnapshot':
                ext = ext.snapshot
            snap_path = ext.filename.format(self)
            snap_path = os.path.join(self.out, snap_path+'-torch')
            state = {'updater': {}}
            self.updater.serialize(serializer['updater'], state['updater'])
            torch.save(state, snap_path)
        elif isinstance(serializer, chainer.serializer.Deserializer):
            state = torch.load(self.torch_snapshot)
            self.updater.serialize(serializer['updater'], state['updater'])

        if hasattr(self.stop_trigger, 'serialize'):
            self.stop_trigger.serialize(serializer['stop_trigger'])

        s = serializer['extensions']
        t = serializer['extension_triggers']
        for name, entry in six.iteritems(self._chainer_trainer._extensions):
            if hasattr(entry.extension, 'serialize'):
                entry.extension.serialize(s[name])
            if hasattr(entry.trigger, 'serialize'):
                entry.trigger.serialize(t[name])


class ExtensionOptimizerAdapter(object):

    """Adapts torch optimizer interface
    to chainer one, so extensions are
    compatible
    It only access the optimizer param_groups
    TODO(ecastill) multiple param_groups
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        # Torch doesnt track the optimizer params
        # until calculations are performed
        # We need to delay the conversion.
        self.torch_model = None
        if isinstance(optimizer.target, chainer.Link):
            self.target = optimizer.target
        else:
            self.torch_model = optimizer.target
            self.target = cpm.TorchModule(optimizer.target)
            # There is no API in torch to know wether a model is on cuda
            param_tensor = next(optimizer.target.parameters())
            if param_tensor.is_cuda:
                self.target.to_gpu(param_tensor.device.index)

    def __getattr__(self, item):
        if item not in ('target', 'optimizer', 'torch_model'):
            return self.optimizer.param_groups[0][item]
        return super().__getattr__(item)

    def __setattr__(self, item, value):
        if item not in ('target', 'optimizer', 'torch_model'):
            self.optimizer.param_groups[0][item] = value
        else:
            super().__setattr__(item, value)

    def serialize(self, serializer, state):
        model_is_torch = self.torch_model is not None
        optimizer_is_torch = isinstance(self.optimizer, torch.optim.Optimizer)
        # if the model is or the optimizer is chainer use the chainer
        # serializers
        if not model_is_torch:
            self.target.serialize(serializer['model'])
        if not optimizer_is_torch:
            self.target.serialize(serializer['optimizer'])

        if isinstance(serializer, chainer.serializer.Serializer):
            if optimizer_is_torch:
                state['optimizer'] = self.optimizer.state_dict()
            if model_is_torch:
                state['model'] = self.torch_model.state_dict()
        elif isinstance(serializer, chainer.serializer.Deserializer):
            if optimizer_is_torch:
                self.optimizer.load_state_dict(state['optimizer'])
            if model_is_torch:
                self.torch_model.load_state_dict(state['model'])
