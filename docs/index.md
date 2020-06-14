[![Github Actions Shield](https://img.shields.io/github/workflow/status/sdv-dev/DeepEcho/Run%20Tests)](https://github.com/sdv-dev/DeepEcho/actions)
[![Coverage Status](https://codecov.io/gh/sdv-dev/DeepEcho/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/DeepEcho)

<br/>

# Overview
The **DeepEcho** library implements generative adversarial networks for modeling
mixed-type (continuous + categorical) multivariate time series. It implements 
a variety of different models (:py:mod:`deepecho.model`) and benchmarks 
(:py:mod:`deepecho.benchmark`) to enable rapid experimentation and development.

Each model supports a different set of features - i.e. timestamps, multiple 
entities, fixed length sequences, etc. - so you can select the model(s) that 
best fit your particular use case. Similarly, the complexity of the benchmarks 
range from simple simulated datasets to complex real world signals.


# Installation
To install the ``deepecho`` library from source, run the following:

```bash
git clone https://github.com/sdv-dev/DeepEcho
cd DeepEcho
make install-develop
```
