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
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sdv-dev/DeepEcho/master?filepath=tutorials)
[![Slack](https://img.shields.io/badge/Slack%20Workspace-Join%20now!-36C5F0?logo=slack)](https://join.slack.com/t/sdv-space/shared_invite/zt-gdsfcb5w-0QQpFMVoyB2Yd6SRiMplcw)

# DeepEcho

* License: [MIT](https://github.com/sdv-dev/DeepEcho/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
* Homepage: https://github.com/sdv-dev/DeepEcho

# Overview

**DeepEcho** is a **Synthetic Data Generation** Python library for **mixed-type**, **multivariate
time series**. It provides:

1. Multiple models based both on **classical statistical modeling** of time series and the latest
   in **Deep Learning** techniques.
2. A robust [benchmarking framework](https://github.com/sdv-dev/SDGym) for evaluating these methods
   on multiple datasets and with multiple metrics.
3. Ability for **Machine Learning researchers** to submit new methods following our `model` and
   `sample` API and get evaluated.

## Try it out now!

If you want to quickly discover **DeepEcho**, simply click the button below and follow the tutorials!

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sdv-dev/DeepEcho/master?filepath=tutorials)

## Join our Slack Workspace

If you want to be part of the SDV community to receive announcements of the latest releases,
ask questions, suggest new features or participate in the development meetings, please join
our Slack Workspace!

[![Slack](https://img.shields.io/badge/Slack%20Workspace-Join%20now!-36C5F0?logo=slack)](https://join.slack.com/t/sdv-space/shared_invite/zt-gdsfcb5w-0QQpFMVoyB2Yd6SRiMplcw)

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

## Install with conda

**DeepEcho** can also be installed using [conda](https://docs.conda.io/en/latest/):

```bash
conda install -c sdv-dev -c pytorch -c conda-forge deepecho
```

This will pull and install the latest stable release from [Anaconda](https://anaconda.org/).


# Quickstart

In this short quickstart, we show how to learn a mixed-type multivariate time series
dataset and then generate synthetic data that resembles it.

We will start by loading the data and preparing the instance of our model.

```python3
from deepecho import PARModel
from deepecho.demo import load_demo

# Load demo data
data = load_demo()

# Define data types for all the columns
data_types = {
    'region': 'categorical',
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

Notice that for smaller datasets like the one used on this demo, CUDA usage introduces
more overhead than the gains it obtains from parallelization, so the process in this
case is more efficient without CUDA, even if it is available.

Once we have created our instance, we are ready to learn the data and generate
new synthetic data that resembles it:

```python3
# Learn a model from the data
model.fit(
    data=data,
    entity_columns=['store_id'],
    context_columns=['region'],
    data_types=data_types,
    sequence_index='date'
)

# Sample new data
model.sample(num_entities=5)
```

The output will be a table with synthetic time series data with the same properties to
the demo data that we used as input.

# What's next?

For more details about **DeepEcho** and all its possibilities and features, please check and
run the [tutorials](tutorials).

If you want to see how we evaluate the performance and quality of our models, please have a
look at the [SDGym Benchmarking framework](https://github.com/sdv-dev/SDGym).

Also, please feel welcome to visit [our contributing guide](CONTRIBUTING.rst) in order to help
us developing new features or cool ideas!

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
