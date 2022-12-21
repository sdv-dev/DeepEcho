<div align="center">
<br/>
<p align="center">
    <i>This repository is part of <a href="https://sdv.dev">The Synthetic Data Vault Project</a>, a project from <a href="https://datacebo.com">DataCebo</a>.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPi Shield](https://img.shields.io/pypi/v/deepecho.svg)](https://pypi.python.org/pypi/deepecho)
[![Tests](https://github.com/sdv-dev/DeepEcho/workflows/Run%20Tests/badge.svg)](https://github.com/sdv-dev/DeepEcho/actions?query=workflow%3A%22Run+Tests%22+branch%3Amaster)
[![Downloads](https://pepy.tech/badge/deepecho)](https://pepy.tech/project/deepecho)
[![Coverage Status](https://codecov.io/gh/sdv-dev/DeepEcho/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/DeepEcho)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sdv-dev/DeepEcho/master?filepath=tutorials/timeseries_data)
[![Slack](https://img.shields.io/badge/Slack%20Workspace-Join%20now!-36C5F0?logo=slack)](https://bit.ly/sdv-slack-invite)

<div align="left">
<br/>
<p align="center">
<a href="https://github.com/sdv-dev/DeepEcho">
<img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/DeepEcho-DataCebo.png"></img>
</a>
</p>
</div>

</div>

# Overview

**DeepEcho** is a **Synthetic Data Generation** Python library for **mixed-type**, **multivariate
time series**. It provides:

1. Multiple models based both on **classical statistical modeling** of time series and the latest
   in **Deep Learning** techniques.
2. A robust [benchmarking framework](https://github.com/sdv-dev/SDGym) for evaluating these methods
   on multiple datasets and with multiple metrics.
3. Ability for **Machine Learning researchers** to submit new methods following our `model` and
   `sample` API and get evaluated.

| Important Links                               |                                                                      |
| --------------------------------------------- | -------------------------------------------------------------------- |
| :computer: **[Website]**                      | Check out the SDV Website for more information about the project.    |
| :orange_book: **[SDV Blog]**                  | Regular publshing of useful content about Synthetic Data Generation. |
| :book: **[Documentation]**                    | Quickstarts, User and Development Guides, and API Reference.         |
| :octocat: **[Repository]**                    | The link to the Github Repository of this library.                   |
| :keyboard: **[Development Status]**           | This software is in its Pre-Alpha stage.                             |
| [![][Slack Logo] **Community**][Community]    | Join our Slack Workspace for announcements and discussions.          |
| [![][MyBinder Logo] **Tutorials**][Tutorials] | Run the SDV Tutorials in a Binder environment.                       |

[Website]: https://sdv.dev
[SDV Blog]: https://sdv.dev/blog
[Documentation]: https://sdv.dev/SDV
[Repository]: https://github.com/sdv-dev/DeepEcho
[License]: https://github.com/sdv-dev/DeepEcho/blob/master/LICENSE
[Development Status]: https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha
[Slack Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/slack.png
[Community]: https://bit.ly/sdv-slack-invite
[MyBinder Logo]: https://github.com/sdv-dev/SDV/blob/master/docs/images/mybinder.png
[Tutorials]: https://mybinder.org/v2/gh/sdv-dev/DeepEcho/master?filepath=tutorials

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
conda install -c pytorch -c conda-forge deepecho
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

---


<div align="center">
<a href="https://datacebo.com"><img align="center" width=40% src="https://github.com/sdv-dev/SDV/blob/master/docs/images/DataCebo.png"></img></a>
</div>
<br/>
<br/>

[The Synthetic Data Vault Project](https://sdv.dev) was first created at MIT's [Data to AI Lab](
https://dai.lids.mit.edu/) in 2016. After 4 years of research and traction with enterprise, we
created [DataCebo](https://datacebo.com) in 2020 with the goal of growing the project.
Today, DataCebo is the proud developer of SDV, the largest ecosystem for
synthetic data generation & evaluation. It is home to multiple libraries that support synthetic
data, including:

* 🔄 Data discovery & transformation. Reverse the transforms to reproduce realistic data.
* 🧠 Multiple machine learning models -- ranging from Copulas to Deep Learning -- to create tabular,
  multi table and time series data.
* 📊 Measuring quality and privacy of synthetic data, and comparing different synthetic data
  generation models.

[Get started using the SDV package](https://sdv.dev/SDV/getting_started/install.html) -- a fully
integrated solution and your one-stop shop for synthetic data. Or, use the standalone libraries
for specific needs.
