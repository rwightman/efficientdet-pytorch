""" Setup
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

exec(open('effdet/version.py').read())
setup(
    name='effdet',
    version=__version__,
    description='EfficientDet for PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/rwightman/efficientdet-pytorch',
    author='Ross Wightman',
    author_email='hello@rwightman.com',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Note that this is a string of words separated by whitespace, not a list.
    keywords='pytorch pretrained efficientdet efficientnet bifpn object detection',
    packages=find_packages(exclude=['data']),
    install_requires=['torch >= 1.12.1', 'torchvision', 'timm >= 0.9.2', 'pycocotools>=2.0.2', 'omegaconf>=2.0'],
    python_requires='>=3.7',
)
