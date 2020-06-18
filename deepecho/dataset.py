import matplotlib.pyplot as plt
import numpy as np


def visualize(df):
    """Render a dataframe containing a single time series.

    This function renders the given time series data. Each feature is plotted
    horizontally over time; continuous features are rendered as lines while
    categorical features are rendered as points.

    Args:
        df: A pandas DataFrame.

    Returns:
        (plt.Figure): A Figure object.
    """
    fig, axs = plt.subplots(len(df.columns))

    for i, column in enumerate(df.columns):
        if df[column].dtype == np.float64:
            axs[i].scatter(range(len(df)), df[column])
        elif df[column].dtype == np.object:
            axs[i].scatter(range(len(df)), df[column])
        axs[i].set_ylabel(column)
        axs[i].xaxis.set_ticklabels([])

    return fig


class Dataset():
    """This represents a time-series dataset.

    By default, the DataFrame is expected to contain a single time series
    without an explicit time index (i.e. the observations are equally spaced
    and therefore don't have a timestamp). For example, consider a dataset
    which reports the average temperature in California at hourly intervals.

    If `time_idx` is set, then the corresponding column of the DataFrame is
    expected to contain a timestamp (i.e. a numerical value) indicating the
    time at which the observation occured. For example, consider a dataset
    containing github events (i.e. create, push, pull) with a timestamp.

    If `entity_idx` is set, then the corresponding column of the DataFrame is
    expected to contain a key which can be used to group the observations into
    multiple time series. For example, consider a dataset which reports the
    average temperature in every state - the state would be the enitity - at
    hourly intervals.

    If `fixed_length` is set, then the time series is expected to have the
    given length. If `entity_idx` is not set, then this is equal to the
    number of rows in the dataframe; however, if `entity_idx` is set, then
    each group of rows belonging to a particular entity should have the
    given number of rows.

    If `entity_df` is set, then `entity_idx` must be set. The `entity_idx`
    column of the `entity_df` table serves as a primary key; the remaining
    columns of the `entity_df` table can be used to model a conditional
    distribution for the corresponding entity in the `df` table.

    Attributes:
        df: A pandas.DataFrame object containing the raw data.
        time_idx: A string (or None) indicating a column in the dataframe.
        entity_idx: A string (or None) indicating a column in the dataframe.
        fixed_length: An integer (or None) indicating the number of time steps.
        entity_df: A pandas.DataFrame object representing the parent table.
    """

    def __init__(self, df, time_idx=None, entity_idx=None, fixed_length=None, entity_df=None):
        self.df = df
        self.time_idx = time_idx
        self.entity_idx = entity_idx
        self.fixed_length = fixed_length
        self.entity_df = entity_df

    def visualize(self):
        """Visualize the dataset.

        This plots the dataset. If it contains multiple entities, it only plots
        the time series for the first entity.

        Returns:
            (plt.Figure): A Figure object.
        """
        df = self.df
        if self.entity_idx:
            for _, df in df.groupby(self.entity_idx):
                df = df.drop(self.entity_idx, axis=1)
                break
        return visualize(df)
