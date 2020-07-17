# -*- coding: utf-8 -*-

"""DeepEcho Command Line Interface module."""

import argparse
import logging
import sys
import warnings

import tabulate

from deepecho.benchmark import get_datasets_list, run_benchmark


def _run(args):
    # Logger setup
    log_level = (3 - args.verbose) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    logging.basicConfig(level=log_level, format=fmt)
    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)

    # run
    results = run_benchmark(
        args.models,
        args.datasets,
        args.metrics,
        args.distributed,
        args.output_path,
    )

    if not args.output_path:
        print(tabulate.tabulate(
            results,
            tablefmt='github',
            headers=results.columns
        ))


def _datasets_list(args):
    del args  # Unused
    datasets = '\n  - '.join(get_datasets_list())
    print('Available DeepEcho Datasets:\n{}'.format(datasets))


def _get_parser():
    parser = argparse.ArgumentParser(description='DeepEcho Benchmark Command Line Interface')
    parser.set_defaults(action=None)
    action = parser.add_subparsers(title='action')
    action.required = True

    # Get Datasets List
    datasets_list = action.add_parser(
        'datasets-list', help='Get the list of available DeepEcho Datasets')
    datasets_list.set_defaults(action=_datasets_list)
    datasets_list.set_defaults(user=None)

    # Run action
    run = action.add_parser('run', help='Run the DeepEcho Benchmark')
    run.set_defaults(action=_run)
    run.set_defaults(user=None)

    run.add_argument('-v', '--verbose', action='count', default=0,
                     help='Be verbose. Use -vv for increased verbosity.')
    run.add_argument('-o', '--output-path', type=str, required=False,
                     help='Path to the CSV file where the report will be dumped')
    run.add_argument('-m', '--models', nargs='+',
                     help='Models/s to be benchmarked. Accepts multiple names.')
    run.add_argument('-d', '--datasets', nargs='+',
                     help='Datasets/s to be used. Accepts multiple names.')
    run.add_argument('-s', '--metrics', nargs='+',
                     choices=['sdmetrics', 'classification', 'detection'],
                     help='Metric/s to use. Accepts multiple names.')
    run.add_argument('-D', '--distributed', action='store_true',
                     help='Whether to distribute computation using dask.')

    return parser


def main():
    """DeepEcho Command Line Interface main function."""  # noqa: D403
    warnings.filterwarnings("ignore")

    # Parse args
    parser = _get_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    args.action(args)


if __name__ == '__main__':
    main()
