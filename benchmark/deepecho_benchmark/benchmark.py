import numpy as np
import pandas as pd
import sdmetrics
from sdv import Metadata
from tqdm import tqdm


class Benchmark():

    def __init__(self, df, key, context):
        self.df = df
        self.key = key
        self.context = context

    def evaluate(self, model):
        sequences, context_types, data_types = self._as_sequences()
        model.fit(sequences, context_types, data_types)

        synthetic_sequences = []
        for seq in tqdm(sequences, "Sampling"):
            synthetic_sequences.append({
                "context": seq["context"],
                "data": model.sample(seq["context"])
            })

        report = self._report(sequences, synthetic_sequences)
        print(report.details())
        return report.overall()

    def _report(self, sequences, synthetic_sequences):
        real_df = []
        for seq in sequences:
            real_df.append(pd.DataFrame(seq["data"]).T)
        real_df = pd.concat(real_df, axis=0)

        synthetic_df = []
        for seq in synthetic_sequences:
            synthetic_df.append(pd.DataFrame(seq["data"]).T)
        synthetic_df = pd.concat(synthetic_df, axis=0)
        synthetic_df = synthetic_df.astype(np.float64)

        metadata = Metadata()
        metadata.add_table('data', data=real_df)
        real_tables = {'data': real_df}
        synthetic_tables = {'data': synthetic_df}

        report = sdmetrics.evaluate(metadata, real_tables, synthetic_tables)
        return report

    def _as_sequences(self):
        sequences = []
        context_types = ["categorical"]
        data_types = None

        for _, sub_df in self.df.groupby(self.key):
            sequence = {}
            sub_df = sub_df.drop(self.key, axis=1)

            sequence["context"] = sub_df[self.context].iloc[0].tolist()
            sub_df = sub_df.drop(self.context, axis=1)

            sequence["data"] = []
            for column in sub_df.columns:
                sequence["data"].append(sub_df[column].values.tolist())

            data_types = []
            for column in sub_df.columns:
                if sub_df[column].dtype == np.float64:
                    data_types.append("continuous")
                else:
                    raise ValueError("idk")

            sequences.append(sequence)

        return sequences, context_types, data_types
