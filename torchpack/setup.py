from setuptools import setup, find_packages

from mtpack.version import __version__

setup(
    name='mtpack',
    version=__version__,
    packages=find_packages(exclude=['examples', 'tests']),
    install_requires=[
        'torch>=1.2',
        'torchvision>=0.4',
    ],
    url='https://github.com/synxlin/mini-torchpack/',
    license='MIT'
)
