<p align="left">
  <a href="https://dai.lids.mit.edu">
    <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab" />
  </a>
  <i>An Open Source Project from the <a href="https://dai.lids.mit.edu">Data to AI Lab, at MIT</a></i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/deepecho.svg)](https://pypi.python.org/pypi/deepecho)
[![Tests](https://github.com/sdv-dev/DeepEcho/workflows/Run%20Tests/badge.svg)](https://github.com/sdv-dev/DeepEcho/actions?query=workflow%3A%22Run+Tests%22+branch%3Amaster)
[![Downloads](https://pepy.tech/badge/deepecho)](https://pepy.tech/project/deepecho)
[![Coverage Status](https://codecov.io/gh/sdv-dev/DeepEcho/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/DeepEcho)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sdv-dev/DeepEcho/master?filepath=tutorials/timeseries_data)
[![Slack](https://img.shields.io/badge/Slack%20Workspace-Join%20now!-36C5F0?logo=slack)](https://join.slack.com/t/sdv-space/shared_invite/zt-gdsfcb5w-0QQpFMVoyB2Yd6SRiMplcw)

<img align="center" width=60% src="docs/images/DeepEcho-Logo.png">

* Website: https://sdv.dev
* Documentation: https://sdv.dev/SDV
* Repository: https://github.com/sdv-dev/DeepEcho
* License: [MIT](https://github.com/sdv-dev/DeepEcho/blob/master/LICENSE)
* Development Status: [Pre-Alpha](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)

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

[![Binder](https://mybinder.org/badge_logo.svg)](
https://mybinder.org/v2/gh/sdv-dev/DeepEcho/master?filepath=tutorials/timeseries_data)

## Join our Slack Workspace

If you want to be part of the SDV community to receive announcements of the latest releases,
ask questions, suggest new features or participate in the development meetings, please join
our Slack Workspace!

[![Slack](https://img.shields.io/badge/Slack%20Workspace-Join%20now!-36C5F0?logo=slack)](https://join.slack.com/t/sdv-space/shared_invite/zt-gdsfcb5w-0QQpFMVoyB2Yd6SRiMplcw)

# Install

**DeepEcho** is part of the **SDV** project and is automatically installed alongside it. For
details about this process please visit the [SDV Installation Guide](
https://sdv.dev/SDV/getting_started/install.html)

Optionally, **DeepEcho** can also be installed as a standalone library using the following commands:

**Using `pip`:**

```bash
pip install deepecho
```

**Using `conda`:**

```bash
conda install -c sdv-dev -c pytorch -c conda-forge deepecho
```

For more installation options please visit the [DeepEcho installation Guide](INSTALL.md)

# Quickstart

**DeepEcho** is included as part of [SDV](https://sdv.dev/SDV) to model and sample synthetic
time series. In most cases, usage through SDV is recommeded, since it provides additional
functionalities which are not available here. For more details about how to use DeepEcho
whithin SDV, please visit the corresponding User Guide:

* [SDV TimeSeries User Guide](https://sdv.dev/SDV/user_guides/timeseries/par.html)

## Standalone usage

**DeepEcho** can also be used as a standalone library.

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

# The Synthetic Data Vault

<p>
  <a href="https://sdv.dev">
    <img width=30% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/SDV-Logo-Color-Tagline.png?raw=true">
  </a>
  <p><i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a></i></p>
</p>

* Website: https://sdv.dev
* Documentation: https://sdv.dev/SDV
