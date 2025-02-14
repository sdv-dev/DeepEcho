# History

### v0.7.0 - 2025-02-13

### Maintenance

* Combine  `static_code_analysis.yml` with `release_notes.yml` - Issue [#142](https://github.com/sdv-dev/DeepEcho/issues/142) by @R-Palazzo
* Support Python 3.13 - Issue [#136](https://github.com/sdv-dev/DeepEcho/issues/136) by @rwedge
* Update codecov and add flag for integration tests - Issue [#135](https://github.com/sdv-dev/DeepEcho/issues/135) by @pvk-developer

## v0.6.1 - 2024-10-22

### Bugs Fixed

* Pip check fails with pip==24.2 on minimum tests - Issue [#130](https://github.com/sdv-dev/DeepEcho/issues/130) by @amontanez24
* Cap numpy to less than 2.0.0 until DeepEcho supports - Issue [#117](https://github.com/sdv-dev/DeepEcho/issues/117) by @gsheni

### Internal

* Add workflow to generate release notes - Issue [#126](https://github.com/sdv-dev/DeepEcho/issues/126) by @amontanez24

### Maintenance

* Add support for numpy 2.0.0 - Issue [#118](https://github.com/sdv-dev/DeepEcho/issues/118) by @R-Palazzo
* Only run unit and integration tests on oldest and latest python versions for macos - Issue [#110](https://github.com/sdv-dev/DeepEcho/issues/110) by @R-Palazzo

# 0.6.0 - 2024-04-10

This release adds support for Python 3.12!

### Maintenance

* Support Python 3.12 - Issue [#85](https://github.com/sdv-dev/DeepEcho/issues/85) by @fealho
* Transition from using setup.py to pyproject.toml to specify project metadata - Issue [#86](https://github.com/sdv-dev/DeepEcho/issues/86) by @R-Palazzo
* Remove bumpversion and use bump-my-version - Issue [#87](https://github.com/sdv-dev/DeepEcho/issues/87) by @R-Palazzo
* Add dependency checker - Issue [#96](https://github.com/sdv-dev/DeepEcho/issues/96) by @lajohn4747
* Add bandit workflow - Issue [#98](https://github.com/sdv-dev/DeepEcho/issues/98) by @R-Palazzo

### Bugs Fixed

* Fix make check candidate - Issue [#91](https://github.com/sdv-dev/DeepEcho/issues/91) by @R-Palazzo
* Fix minimum version workflow when pointing to github branch - Issue [#99](https://github.com/sdv-dev/DeepEcho/issues/99) by @R-Palazzo

## 0.5.0 - 2023-11-13

This release updates the PAR's model progress bar to show loss values and time elapsed (verbose option).

### New Features
* Update progress bar for PAR fitting - Issue [#80](https://github.com/sdv-dev/DeepEcho/issues/80) by @frances-h


## 0.4.2 - 2023-07-25

This release drops support for Python 3.7 and adds support for Python 3.11.

### Maintenance

* Add support for Python 3.11 - Issue [#74](https://github.com/sdv-dev/DeepEcho/issues/74) by @fealho
* Drop support for Python 3.7 - Issue [#75](https://github.com/sdv-dev/DeepEcho/issues/75) by @R-Palazzo

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
