import ignite

from .extensions import add_trainer_extension, load_chainer_snapshot  # NOQA
from .collate import collate_to_array  # NOQA


def _get_version(version):
    # We compare up to the minor version (first two digits).
    # This is because it is highly unlikely that these numbers
    # will contain other character than digits.
    return [int(x) for x in version.split('.')[:2]]


if _get_version(ignite.__version__) < _get_version('0.3.0'):
    raise ImportError('Ignite version found {}. '
                      'Required is >=0.3.0'.format(ignite.__version__))
