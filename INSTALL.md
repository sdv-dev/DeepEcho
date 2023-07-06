# Installing DeepEcho

## Requirements

**DeepEcho** has been developed and tested on
[Python 3.8, 3.9 and 3.10](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](
https://virtualenv.pypa.io/en/latest/) is highly recommended in order to avoid
interfering with other software installed in the system where **DeepEcho** is run.

## Install with pip

The easiest and recommended way to install **DeepEcho** is using [pip](
https://pip.pypa.io/en/stable/):

```bash
pip install deepecho
```

This will pull and install the latest stable release from [PyPi](https://pypi.org/).

## Install with conda

**DeepEcho** can also be installed using [conda](https://docs.conda.io/en/latest/):

```bash
conda install -c sdv-dev -c pytorch -c conda-forge deepecho
```

This will pull and install the latest stable release from [Anaconda](https://anaconda.org/).

## Install from source

If you want to install **DeepEcho** from source you need to first clone the repository
and then execute the `make install` command inside the `stable` branch. Note that this
command works only on Unix based systems like GNU/Linux and macOS:

```bash
git clone https://github.com/sdv-dev/DeepEcho
cd DeepEcho
git checkout stable
make install
```

## Install for development

If you intend to modify the source code or contribute to the project you will need to
install it from the source using the `make install-develop` command. In this case, we
recommend you to branch from `master` first:

```bash
git clone git@github.com:sdv-dev/DeepEcho
cd DeepEcho
git checkout master
git checkout -b <your-branch-name>
make install-develp
```

For more details about how to contribute to the project please visit the [Contributing Guide](
CONTRIBUTING.rst).
