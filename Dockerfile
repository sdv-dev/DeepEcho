FROM nvidia/cuda:10.1-base-ubuntu18.04

RUN mkdir -p /workdir/deepecho && \
    mkdir -p /workdir/benchmark && \
    apt-get update && \
    apt-get install -y build-essential swig python3-dev python3-pip

ENV TINI_VERSION v0.18.0
ADD https://github.com/krallin/tini/releases/download/v0.18.0/tini /usr/local/bin/tini
RUN chmod +x /usr/local/bin/tini
ENTRYPOINT ["/usr/local/bin/tini", "--"]

COPY setup.py MANIFEST.in README.md HISTORY.md /workdir/deepecho/
COPY benchmark/setup.py benchmark/README.md /workdir/benchmark/
RUN pip3 install numpy cython && pip3 install -e /workdir/deepecho[dev] -e /workdir/benchmark[dev]

COPY deepecho /workdir/deepecho/deepecho
COPY benchmark/deepecho /workdir/benchmark/deepecho
COPY tutorials /workdir/tutorials

WORKDIR /workdir/tutorials
CMD jupyter notebook --ip 0.0.0.0 --allow-root --NotebookApp.token=''
