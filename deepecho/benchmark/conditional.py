from random import choice

import numpy as np
import pandas as pd

from deepecho import Dataset
from deepecho.benchmark.base import Benchmark


class Conditional1(Benchmark):
    """Variable-length conditional time-series.
    """

    def __init__(self, nb_entities=10):
        self.nb_entities = nb_entities

    def sample(self):
        """Create an instance of the Conditional1 dataset.

        Returns:
            A `Dataset` object containing a time-series dataset.
        """
        dataframes, entity_df = [], []
        for entity_id in range(self.nb_entities):
            mode, seq_len = choice([("short", 10), ("long", 20)])
            df = pd.DataFrame({
                "x": np.linspace(0.0, 1.0, num=seq_len),
                "y": np.linspace(0.0, 1.0, num=seq_len)
            })
            df["id"] = entity_id
            dataframes.append(df)
            entity_df.append({
                "id": entity_id,
                "mode": mode
            })
        df = pd.concat(dataframes)
        entity_df = pd.DataFrame(entity_df)
        return Dataset(df, entity_idx="id", entity_df=entity_df)

    def evaluate(self, model):
        """Evaluate the model on this benchmark.

        Args:
            model: A `Model` object.

        Returns:
            A `dict` object containing various metrics.
        """
        entity_df = pd.DataFrame({
            "id": list(range(10)),
            "mode": ["short"] * 5 + ["long"] * 5
        })
        model.fit(self.sample())
        dataset = model.sample(entity_df)
        df = dataset.entity_df.merge(dataset.df, on=dataset.entity_idx)

        short_lengths, long_lengths = [], []
        for _, df in df.groupby(dataset.entity_idx):
            if df["mode"].values[0] == "short":
                short_lengths.append(len(df))
            elif df["mode"].values[0] == "long":
                long_lengths.append(len(df))
            else:
                raise ValueError("Unexpected mode.")

        return {
            "deviation-from-short-length": abs(np.mean(short_lengths) - 10.0),
            "deviation-from-long-length": abs(np.mean(long_lengths) - 20.0),
        }


if __name__ == "__main__":
    model = Conditional1()
    print(model.evaluate(None))
