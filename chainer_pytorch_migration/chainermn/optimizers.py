import chainermn


class ChainerMNOptimizer(object):
    def __init__(self, optimizer):
        super(ChainerMNOptimizer, self).__setattr__(
            'optimizer', optimizer)

    def update(self, lossfun=None, *args, **kwds):
        """Used to fool chainermn optimizer wrappers"""
        self.optimizer.step()

    def setup(self, link):
        self.target = link
        return self

    def __getattr__(self, attr_name):
        return getattr(self.optimizer, attr_name)

    def __setattr__(self, attr_name, value):
        setattr(self.optimizer, attr_name, value)


def create_multi_node_optimizer(actual_optimizer, communicator,
                                double_buffering=False, zero_fill=True):
    return chainermn.optimizers.create_multi_node_optimizer(
            ChainerMNOptimizer(actual_optimizer), communicator,
            double_buffering, zero_fill)
