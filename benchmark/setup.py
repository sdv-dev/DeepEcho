#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

__version__ = '0.1.4.dev0'

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

install_requires = [
    'deepecho=={}'.format(__version__),
    'boto3>=1,<2',
    'humanfriendly>=8.2,<9',
    'numpy>=1.15.4,<2',
    'pandas>=1,<2',
    'scikit-learn>=0.21,<1',
    'sdmetrics>=0.0.2.dev0,<0.1',
    'sdv>=0.5.0,<0.7',
    'sktime>=0.4,<0.5',
    'tabulate>=0.8.3,<0.9',
    'tqdm>=4,<5',
    'tsfresh>=0.15,<1',

    # Compatibility issues
    'docutils<0.16,>=0.10',
]

kubernetes_requires = [
    'dask>=2.15.0,<3',
    'distributed>=2.15.2,<2.16',
    'kubernetes>=11.0.0,<11.1',
    'dask-kubernetes>=0.10.1,<0.11',
    'bokeh>=2.1.1,<3',
]

setup_requires = [
    'pytest-runner>=2.11.1',
]

setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description='Benchmarking for mixed-type multivariate time series modeling tools.',
    entry_points = {
        'console_scripts': [
            'deepecho-benchmark=deepecho.benchmark.__main__:main'
        ],
    },
    extras_require={
        'dev': kubernetes_requires,
        'kubernetes': kubernetes_requires
    },
    install_package_data=True,
    install_requires=install_requires,
    license='MIT license',
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='deepecho benchmark',
    name='deepecho-benchmark',
    packages=find_packages(),
    python_requires='>=3.6,<3.9',
    setup_requires=setup_requires,
    test_suite='tests',
    url='https://github.com/sdv-dev/DeepEcho',
    version=__version__,
    zip_safe=False,
)
