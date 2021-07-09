# History

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
