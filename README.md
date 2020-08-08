<p align="left">
<img width=20% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“sdv-dev” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<p>
  <img width=65% src="docs/images/DeepEcho-Logo.png">
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/deepecho.svg)](https://pypi.python.org/pypi/deepecho)
[![Travis CI Shield](https://travis-ci.org/sdv-dev/DeepEcho.svg?branch=master)](https://travis-ci.org/sdv-dev/DeepEcho)
[![Coverage Status](https://codecov.io/gh/sdv-dev/DeepEcho/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/DeepEcho)
[![Downloads](https://pepy.tech/badge/deepecho)](https://pepy.tech/project/deepecho)

# DeepEcho

* License: [MIT](https://github.com/sdv-dev/DeepEcho/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
* Homepage: https://github.com/sdv-dev/DeepEcho

# Overview

**DeepEcho** is a Python library that implements methods for modeling mixed-type
multivariate time series. It provides multiple models and benchmarks to enable
rapid research and development.

# Install

## Requirements

**DeepEcho** has been developed and tested on [Python 3.6, 3.7 and 3.8](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system where **DeepEcho**
is run.

## Install with pip

The easiest and recommended way to install **DeepEcho** is using [pip](https://pip.pypa.io/en/stable/):

```bash
pip install deepecho
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

If you want to install from source or contribute to the project please read the
[Contributing Guide](CONTRIBUTING.rst).


# Quickstart

In this short quickstart, we show how to model a mixed-type multivariate time series
dataset and then generate synthetic data that resembles it.

We will start by loading the data and preparing the instance of our model.

```python3
from deepecho import PARModel
from deepecho.demo import load_demo

# Load demo data
data = load_demo()

# Define entity and context columns
entity_columns = ['id']
context_columns = ['season']

# Define data types for all the columns
data_types = {
    'season': 'categorical',
    'day_of_week': 'categorical',
    'total_sales': 'continuous',
    'nb_customers': 'count',
}

model = PARModel(cuda=False)
```

If we want to use different settings for our model, like increasing the number
of epochs or enabling CUDA, we can pass the arguments when creating the model:

```python  # keep this as python (without the 3) to avoid using it in test-readme
model = PARModel(epochs=1024, cuda=True)
```

Notice that for smaller datasets like the one used on this demo, CUDA usage
introduces more overhead than the gains it obtains from parallelization, so
the modeling in this case is more efficient without CUDA, even if it is available.

Once we have created our instance, we are ready to learn the data and generate
new synthetic data that resembles it:

```python3
# Learn a model from the data
model.fit(
    data=data,
    entity_columns=entity_columns,
    context_columns=context_columns,
    data_types=data_types,
)

# Sample new data
model.sample(num_entities=5)
```

The output will be a table with synthetic time series data with the same properties to
the demo data that we used as input.

# What's next?

For more details about **DeepEcho** and all its possibilities and features, please check and
run the [tutorials](tutorials).

If you want to see how we evaluate the performance of our models and their results, please
have a look at the [DeepEcho Benchmarking framework](benchmark)

There you can learn more about [how to contribute to DeepEcho](CONTRIBUTING.rst) in order to
help us developing new features or cool ideas.

# Related Projects

## SDV

[SDV](https://github.com/HDI-Project/SDV), for Synthetic Data Vault, is the end-user library for
synthesizing data in development under the [HDI Project](https://hdi-dai.lids.mit.edu/).
SDV allows you to easily model and sample relational datasets using DeepEcho thought a simple API.
Other features include anonymization of Personal Identifiable Information (PII) and preserving
relational integrity on sampled records.

## CTGAN

[CTGAN](https://github.com/sdv-dev/CTGAN) is a GAN based model for synthesizing tabular data.
It's also developed by the [MIT's Data to AI Lab](https://sdv-dev.github.io/) and is under
active development.
