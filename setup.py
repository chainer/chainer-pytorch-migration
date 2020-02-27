import setuptools


setuptools.setup(
    name='chainer_pytorch_migration',
    description='Chainer/PyTorch Migration Library',
    license='MIT License',
    version='0.0.2',
    install_requires=['chainer', 'numpy', 'torch'],
    extras_require={'test': ['pytest']},
    packages=[
        'chainer_pytorch_migration',
        'chainer_pytorch_migration.ignite',
        'chainer_pytorch_migration.chainermn',
    ],
)
