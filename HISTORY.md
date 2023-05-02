# History

## 0.4.1 - 2023-05-02

This release adds support for Pandas 2.0 and PyTorch 2.0!

### Maintenance

* Remove upper bound for pandas - Issue [#69](https://github.com/sdv-dev/DeepEcho/issues/69) by @frances-h
* Upgrade to Torch 2.0 - Issue [#70](https://github.com/sdv-dev/DeepEcho/issues/70) by @frances-h

## 0.4.0 - 2023-01-10

This release adds support for python 3.10 and 3.11. It also drops support for python 3.6.

### Maintenance

* Support Python 3.10 and 3.11 - Issue [#63](https://github.com/sdv-dev/DeepEcho/issues/63) by @pvk-developer
* DeepEcho Package Maintenance Updates - Issue [#62](https://github.com/sdv-dev/DeepEcho/issues/62) by @pvk-developer

## 0.3.0 - 2021-11-15

This release adds support for Python 3.9 and updates dependencies to ensure compatibility with the rest
of the SDV ecosystem.

* Add support for Python 3.9 - Issue [#41](https://github.com/sdv-dev/DeepEcho/issues/41) by @fealho
* Add pip check to CI workflows internal improvements - Issue [#39](https://github.com/sdv-dev/DeepEcho/issues/39) by @pvk-developer
* Add support for pylint>2.7.2 housekeeping - Issue [#33](https://github.com/sdv-dev/DeepEcho/issues/33) by @fealho
* Add support for torch>=1.8 housekeeping - Issue [#32](https://github.com/sdv-dev/DeepEcho/issues/32) by @fealho

## 0.2.1 - 2021-10-12

This release fixes a bug with how DeepEcho handles NaN values.

* Handling NaN's bug - Issue [#35](https://github.com/sdv-dev/DeepEcho/issues/35) by @fealho

## 0.2.0 - 2021-02-24

Maintenance release to update dependencies and ensure compatibility with the rest
of the SDV ecosystem libraries.

## 0.1.4 - 2020-10-16

Minor maintenance version to update dependencies and documentation, and
also make the demo data loading function parse dates properly.

## 0.1.3 - 2020-10-16

This version includes several minor improvements to the PAR model and the
way the sequences are generated:

* Sequences can now be generated without dropping the sequence index.
* The PAR model learns the min and max length of the sequence from the input data.
* NaN values are properly supported for both categorical and numerical columns.
* NaN values are generated for numerical columns only if there were NaNs in the input data.
* Constant columns can now be modeled.

## 0.1.2 - 2020-09-15

Add BasicGAN Model and additional benchmarking results.

## 0.1.1 - 2020-08-15

This release includes a few new features to make DeepEcho work on more types of datasets
as well as to making it easier to add new datasets to the benchmarking framework.

* Add `segment_size` and `sequence_index` arguments to `fit` method.
* Add `sequence_length` as an optional argument to `sample` and `sample_sequence` methods.
* Update the Dataset storage format to add `sequence_index` and versioning.
* Separate the sequence assembling process in its own `deepecho.sequences` module.
* Add function `make_dataset` to create a dataset from a dataframe and just a few column names.
* Add notebook tutorial to show how to create a datasets and use them.

## 0.1.0 - 2020-08-11

First release.

Included Features:

* PARModel
* Demo dataset and tutorials
* Benchmarking Framework
* Support and instructions for benchmarking on a Kubernetes cluster.
