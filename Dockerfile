FROM daskdev/dask:2.8.0

RUN mkdir -p /workdir/deepecho && \
    mkdir -p /workdir/benchmark && \
    apt-get update && \
    apt-get install -y build-essential swig

COPY setup.py MANIFEST.in README.md HISTORY.md /workdir/deepecho/
COPY benchmark/setup.py benchmark/README.md /workdir/benchmark/
RUN pip install -e /workdir/deepecho[dev] -e /workdir/benchmark[dev]

COPY deepecho /workdir/deepecho/deepecho
COPY benchmark/deepecho /workdir/benchmark/deepecho
COPY tutorials /workdir/tutorials

WORKDIR /workdir/tutorials
CMD jupyter notebook --ip 0.0.0.0 --allow-root --NotebookApp.token=''
