"""DeepEcho Benchmark CLI.

Usage:
    deepecho [--csv=<csv>]

Options:
    -h --help            Show this screen.
    --csv <csv>          The output path to the CSV file.
"""
import pandas as pd
from docopt import docopt

from deepecho.benchmark import Simple1, Simple2, Simple3
from deepecho.model import Model

BENCHMARKS = {
    "Simple1": Simple1(),
    "Simple2": Simple2(),
    "Simple3": Simple3()
}


def main():
    args = docopt(__doc__)

    results = []
    for benchmark_name, benchmark in BENCHMARKS.items():
        for model_cls in Model.__subclasses__():
            try:
                result = benchmark.evaluate(model_cls())
            except ValueError as e:
                result = {
                    "error": str(e)
                }
            result["benchmark"] = benchmark_name
            result["model"] = model_cls.__name__
            results.append(result)
            print(results[-1])
    df = pd.DataFrame(results)
    if args['--csv']:
        df.to_csv(args['--csv'], index=False)
    print(df)


if __name__ == '__main__':
    main()
