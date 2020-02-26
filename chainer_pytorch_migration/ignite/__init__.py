import re

import ignite

from .extensions import add_trainer_extension, load_chainer_snapshot  # NOQA
from .collate import collate_to_array  # NOQA


def _get_version(version):
    # We compare up to the minor version (first two digits).
    # This is because it is highly unlikely that these numbers
    # will contain other character than digits.

    # Ignite versioning system is not explicitly documented.
    # However, it seems to be using semver, so the
    # major and minor ids can be only integers.
    # Some examples of versions are:
    # 0.1.0, 0.1.1, 0.3.0.dev20191007, 0.3.0.
    version_regexp = r'^[0-9]+\.[0-9]+\.[0-9]+(\.[0-9a-zA-Z]+)?$'
    if re.search(version_regexp, version):
        return [int(x) for x in version.split('.')[:2]]
    raise ValueError('Invalid version format')


if _get_version(ignite.__version__) < _get_version('0.3.0'):
    raise ImportError('Ignite version found {}. '
                      'Required is >=0.3.0'.format(ignite.__version__))
