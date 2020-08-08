# -*- coding: utf-8 -*-

"""DeepEcho Command Line Interface module."""

import argparse
import logging
import sys

import tabulate

from deepecho.benchmark import get_datasets_list, run_benchmark


def _logging_setup(verbosity):
    # Logger setup
    log_level = (3 - verbosity) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    logging.basicConfig(level=log_level, format=fmt)
    logging.getLogger("botocore").setLevel(logging.ERROR)
    logging.getLogger("urllib3").setLevel(logging.CRITICAL)


def _run(args):
    _logging_setup(args.verbose)
    if args.datasets and len(args.datasets) == 1:
        try:
            num_datasets = int(args.datasets[0])
            args.datasets = get_datasets_list().head(num_datasets)['dataset'].tolist()
        except ValueError:
            pass

    if args.distributed and (args.threads or args.workers):
        # Start a local cluster of the indicated size
        # Import only if necessary
        from dask.distributed import Client, LocalCluster  # pylint: disable=C0415

        Client(LocalCluster(n_workers=args.workers, threads_per_worker=args.threads))

    # run
    results = run_benchmark(
        args.models,
        args.datasets,
        args.metrics,
        args.max_entities,
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
    _logging_setup(args.verbose)
    datasets = get_datasets_list(args.extended)

    print('Available DeepEcho Datasets:')
    print(tabulate.tabulate(
        datasets,
        tablefmt='github',
        headers=datasets.columns,
        showindex=False
    ))


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
    datasets_list.add_argument('-e', '--extended', action='store_true',
                               help='Add dataset details (Slow).')
    datasets_list.add_argument('-v', '--verbose', action='count', default=0,
                               help='Be verbose. Use -vv for increased verbosity.')

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
    run.add_argument('-M', '--max-entities', type=int,
                     help='Maximum number of entities to load per dataset.')
    run.add_argument('-s', '--metrics', nargs='+',
                     choices=['sdmetrics', 'classification', 'detection'],
                     help='Metric/s to use. Accepts multiple names.')
    run.add_argument('-D', '--distributed', action='store_true',
                     help='Whether to distribute computation using dask.')
    run.add_argument('-W', '--workers', type=int,
                     help='Number of workers to use when distributing locally.')
    run.add_argument('-T', '--threads', type=int,
                     help='Number of threads to use when distributing locally.')

    return parser


def main():
    """DeepEcho Command Line Interface main function."""  # noqa: D403
    # Parse args
    parser = _get_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    args.action(args)


if __name__ == '__main__':
    main()
