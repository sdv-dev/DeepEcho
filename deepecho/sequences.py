"""Functions to manipulate sequences and assemble training examples."""


def segment_by_size(sequence, segment_size):
    """Segment the sequence in segments of the indicated size.

    If sequence length is not exactly divisible by the ``segment_size``,
    extra data points at the end will be discarded to ensure that all
    the segments have the same size.

    Args:
        sequence (pandas.DataFrame):
            Sequence to segment, passed as a multi-column ``pandas.DataFrame``.
        segment_size (int):
            Size of each segment, passed as an integer.

    Returns:
        list:
            List of ``pandas.DataFrames`` containing each segment, all
            of the indicated size.
    """
    sequences = []
    start = 0
    total = len(sequence)
    while start < total:
        end = start + segment_size
        segment = sequence.iloc[start:end].reset_index(drop=True)
        if len(segment) != segment_size:
            break

        sequences.append(segment)
        start = end

    return sequences


def segment_by_time(sequence, segment_size, sequence_index):
    """Segment the sequence in segments of the indicated time length.

    Segmentation will happen by time, which means that there is no guarantee
    that the outputed segments all contain the same number of data points.

    Args:
        sequence (pandas.DataFrame):
            Sequence to segment, passed as a multi-column ``pandas.DataFrame``.
        segment_size (pandas.Timedelta):
            Size of each segment, passed as a ``pandas.Timedelta`` object.
        sequence_index (pandas.Series):
            Data of the column that will be used as the time index for the
            segmentation.

    Returns:
        list:
            List of ``pandas.DataFrames`` containing each segment.
    """
    sequences = []
    start = sequence_index.iloc[0]
    max_time = sequence_index.iloc[-1]
    while start <= max_time:
        end = start + segment_size
        selected = (start <= sequence_index) & (sequence_index < end)
        sequences.append(sequence[selected.to_numpy()].reset_index(drop=True))
        start = end

    return sequences


def segment_sequence(sequence, segment_size, sequence_index, drop_sequence_index=True):
    """Segment the sequence in segments of the indicated time length or size.

    If a ``sequence_index`` is given, data will be sorted by it first.

    If ``segment_size`` is not passed, the whole sequence will be returned
    as a single segment.

    Args:
        sequence (pandas.DataFrame):
            Sequence to segment, passed as a multi-column ``pandas.DataFrame``.
        segment_size (int or pandas.Timedelta):
            Size of each segment, passed as an integer or as a``pandas.Timedelta``
            object.
        sequence_index (str):
            Name of the column that will be used as the time index for the
            segmentation. Required if a timedelta ``segment_size`` is passed.
        drop_sequence_index (bool):
            Whether to drop the sequence index after sorting. Defaults to ``True``.

    Returns:
        list:
            List of ``pandas.DataFrames`` containing each segment.
    """
    if sequence_index is not None:
        sequence = sequence.sort_values(sequence_index)
        sequence_index_values = sequence[sequence_index]
        if drop_sequence_index:
            del sequence[sequence_index]

    if segment_size is None:
        return [sequence]

    if isinstance(segment_size, int):
        return segment_by_size(sequence, segment_size)

    return segment_by_time(sequence, segment_size, sequence_index_values)


def _convert_to_dicts(segments, context_columns):
    sequences = []
    for segment in segments:
        if context_columns:
            context = segment[context_columns]
            if len(context.drop_duplicates()) > 1:
                raise ValueError('Context columns are not constant within each segment.')

            context = context.iloc[0].to_numpy()
            segment = segment.drop(context_columns, axis=1)
        else:
            context = []

        lists = [list(row) for _, row in segment.items()]
        sequences.append({'context': context, 'data': lists})

    return sequences


def assemble_sequences(
    data,
    entity_columns,
    context_columns,
    segment_size,
    sequence_index,
    drop_sequence_index=True,
):
    """Build sequences from the data, grouping first by entity and then segmenting by size.

    Input is a ``pandas.DataFrame`` containing all the data, lists of entity and context
    columns and instructions about how to segment each entity sequence.

    The process of building the sequences consists on:

    1. First group by the entity columns. If no entity columns are given this
       step is skipped.
    2. Then segment the data that corresponds to each data using the segment_size
       and sequence_index details.
    3. Finally build dictionaries out of each segment, containing two elements:
        * `context`: List of contextual values.
        * `data`: List containing one list for each data column, containing its values.

    Args:
        data (pandas.DataFrame):
            Data to assemble in sequences, containing entity columns, context columns
            and data columns.
        entity_columns (list):
            List with the names of the columns that form each entity_id.
        context_columns (list):
            List with the names of the columns that act as context for each entity or
            segment.
            Context values must be constant within each entity, if entity_columns are
            given, or within each segment otherwise.
        segment_size (int or pandas.Timedelta):
            Size of each segment, passed as an integer or as a``pandas.Timedelta``
            object.
        sequence_index (str):
            Name of the column that will be used as the time index for the
            segmentation. Required if a timedelta ``segment_size`` is passed.
        drop_sequence_index (bool):
            Whether to drop the sequence index after sorting. Defaults to ``True``.

    Raises:
        ValueError:
            If context columns are not constant within each entity or segment.

    Returns:
        list:
            List of ``pandas.DataFrames`` containing each segment.
    """
    if not entity_columns:
        segments = segment_sequence(data, segment_size, sequence_index, drop_sequence_index)
    else:
        segments = []
        groupby_columns = entity_columns[0] if len(entity_columns) == 1 else entity_columns
        for _, sequence in data.groupby(groupby_columns):
            sequence = sequence.drop(entity_columns, axis=1)
            if context_columns:
                if len(sequence[context_columns].drop_duplicates()) > 1:
                    raise ValueError('Context columns are not constant within each entity.')

            entity_segments = segment_sequence(
                sequence, segment_size, sequence_index, drop_sequence_index
            )
            segments.extend(entity_segments)

    return _convert_to_dicts(segments, context_columns)
