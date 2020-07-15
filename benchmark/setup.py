#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as readme_file:
    readme = readme_file.read()

install_requires = [
    'boto3>=1,<2',
    'numpy>=1.15.4,<2',
    'pandas>=0.22,<0.25',
    'scikit-learn>=0.21,<0.23',
    'sdmetrics>=0.0.2.dev0',
    'sdv>=0.3.5,<0.4',
    'sktime>=0.3,<0.4',
    'tqdm>=4,<5',

    # Compatibility issues
    'docutils<0.15,>=0.10',
    'matplotlib<3.2.2,>=2.2.2',
    'scipy<1.3,>=1.2',
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
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description='Benchmarking for mixed-type multivariate time series modeling tools.',
    entry_points = {
        'console_scripts': [
            'deepecho-benchmark=deepecho.benchmark.__main__:main'
        ],
    },
    install_package_data=True,
    install_requires=install_requires,
    license='MIT license',
    long_description=readme,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='deepecho benchmark',
    name='deepecho-benchmark',
    packages=find_packages(include=['deepecho.benchmark.*']),
    python_requires='>=3.5',
    setup_requires=setup_requires,
    test_suite='tests',
    url='https://github.com/sdv-dev/DeepEcho',
    version='0.0.2.dev0',
    zip_safe=False,
)
