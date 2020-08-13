import pandas as pd
import pytest

from deepecho.sequences import (
    assemble_sequences, segment_by_size, segment_by_time, segment_sequence)


def test_segment_by_size():
    """The sequence is cut in sequences of the indicated lenght."""
    sequence = pd.DataFrame({
        'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'b': [9, 8, 7, 6, 5, 4, 3, 2, 1],
    })

    out = segment_by_size(sequence, 3)

    assert isinstance(out, list)
    assert len(out) == 3

    pd.testing.assert_frame_equal(pd.DataFrame({
        'a': [1, 2, 3],
        'b': [9, 8, 7],
    }), out[0])
    pd.testing.assert_frame_equal(pd.DataFrame({
        'a': [4, 5, 6],
        'b': [6, 5, 4],
    }), out[1])
    pd.testing.assert_frame_equal(pd.DataFrame({
        'a': [7, 8, 9],
        'b': [3, 2, 1],
    }), out[2])


def test_segment_by_time():
    """The sequence is cut in sequences of the indicated time lenght."""
    sequence = pd.DataFrame({
        'time': pd.date_range(start='2001-01-01', periods=9, freq='1d'),
        'a': [1, 2, 3, 4, 5, 6, 7, 8, 9],
        'b': [9, 8, 7, 6, 5, 4, 3, 2, 1],
    })

    segment_size = pd.to_timedelta('3d')
    out = segment_by_time(sequence, segment_size, 'time')

    assert isinstance(out, list)
    assert len(out) == 3

    pd.testing.assert_frame_equal(pd.DataFrame({
        'time': pd.to_datetime(['2001-01-01', '2001-01-02', '2001-01-03']),
        'a': [1, 2, 3],
        'b': [9, 8, 7],
    }), out[0])
    pd.testing.assert_frame_equal(pd.DataFrame({
        'time': pd.to_datetime(['2001-01-04', '2001-01-05', '2001-01-06']),
        'a': [4, 5, 6],
        'b': [6, 5, 4],
    }), out[1])
    pd.testing.assert_frame_equal(pd.DataFrame({
        'time': pd.to_datetime(['2001-01-07', '2001-01-08', '2001-01-09']),
        'a': [7, 8, 9],
        'b': [3, 2, 1],
    }), out[2])

def test_segment_sequence():
    """If no sequence index is given, segments are not ordered."""
    sequence = pd.DataFrame({
        'a': [1, 2, 3, 7, 8, 9, 4, 5, 6],
        'b': [9, 8, 7, 3, 2, 1, 6, 5, 4],
    })

    out = segment_sequence(sequence, 3, None)

    assert isinstance(out, list)
    assert len(out) == 3

    pd.testing.assert_frame_equal(pd.DataFrame({
        'a': [1, 2, 3],
        'b': [9, 8, 7],
    }), out[0])
    pd.testing.assert_frame_equal(pd.DataFrame({
        'a': [7, 8, 9],
        'b': [3, 2, 1],
    }), out[1])
    pd.testing.assert_frame_equal(pd.DataFrame({
        'a': [4, 5, 6],
        'b': [6, 5, 4],
    }), out[2])

def test_segment_sequence_sequence_index():
    """If a sequence index is given, segments are ordered."""
    sequence = pd.DataFrame({
        'a': [1, 2, 3, 7, 8, 9, 4, 5, 6],
        'b': [9, 8, 7, 3, 2, 1, 6, 5, 4],
    })

    out = segment_sequence(sequence, 3, 'a')

    assert isinstance(out, list)
    assert len(out) == 3

    pd.testing.assert_frame_equal(pd.DataFrame({
        'a': [1, 2, 3],
        'b': [9, 8, 7],
    }), out[0])
    pd.testing.assert_frame_equal(pd.DataFrame({
        'a': [4, 5, 6],
        'b': [6, 5, 4],
    }), out[1])
    pd.testing.assert_frame_equal(pd.DataFrame({
        'a': [7, 8, 9],
        'b': [3, 2, 1],
    }), out[2])

def test__assemble_sequences_no_entity_no_context():
    """If no entity_columns, segment the given data."""
    entity_columns = []
    context_columns = []

    data = pd.DataFrame({
        'a': [1, 2, 3, 4, 5, 6],
        'b': [9, 8, 7, 6, 5, 4],
    })
    out = assemble_sequences(data, entity_columns, context_columns, 3, None)

    assert isinstance(out, list)
    assert out == [
        {
            'context': [],
            'data': [[1, 2, 3], [9, 8, 7]],
        },
        {
            'context': [],
            'data': [[4, 5, 6], [6, 5, 4]],
        },
    ]

def test__assemble_sequences_no_entity_and_context():
    """If no entity columns, segment the given data adding context."""
    entity_columns = []
    context_columns = ['a']

    data = pd.DataFrame({
        'a': [1, 1, 1, 2, 2, 2],
        'b': [1, 2, 3, 4, 5, 6],
        'c': [9, 8, 7, 6, 5, 4],
    })
    out = assemble_sequences(data, entity_columns, context_columns, 3, None)

    assert isinstance(out, list)
    assert out == [
        {
            'context': [1],
            'data': [[1, 2, 3], [9, 8, 7]],
        },
        {
            'context': [2],
            'data': [[4, 5, 6], [6, 5, 4]],
        },
    ]

def test__assemble_sequences_entity_no_segment():
    """If entity columns , group by ."""
    entity_columns = ['a']
    context_columns = []

    data = pd.DataFrame({
        'a': [1, 1, 1, 2, 2, 2],
        'b': [1, 2, 3, 4, 5, 6],
        'c': [9, 8, 7, 6, 5, 4],
    })
    out = assemble_sequences(data, entity_columns, context_columns, None, None)

    assert isinstance(out, list)
    assert out == [
        {
            'context': [],
            'data': [[1, 2, 3], [9, 8, 7]],
        },
        {
            'context': [],
            'data': [[4, 5, 6], [6, 5, 4]],
        },
    ]

def test__assemble_sequences_entity_and_segment_size():
    """If entity columns and segment_size, group by and then segment."""
    entity_columns = ['a']
    context_columns = []

    data = pd.DataFrame({
        'a': [1, 1, 1, 1, 1, 1],
        'b': [1, 2, 3, 4, 5, 6],
        'c': [9, 8, 7, 6, 5, 4],
    })
    out = assemble_sequences(data, entity_columns, context_columns, 3, None)

    assert isinstance(out, list)
    assert out == [
        {
            'context': [],
            'data': [[1, 2, 3], [9, 8, 7]],
        },
        {
            'context': [],
            'data': [[4, 5, 6], [6, 5, 4]],
        },
    ]

def test__assemble_sequences_context_error():
    """If context is not constant within an entity, raise an error."""
    entity_columns = ['a']
    context_columns = ['b']

    data = pd.DataFrame({
        'a': [1, 1, 1, 1, 2, 2, 2, 2],
        'b': [1, 1, 2, 2, 3, 3, 4, 4],
        'c': [9, 8, 7, 6, 5, 4, 3, 2],
    })
    with pytest.raises(ValueError):
        assemble_sequences(data, entity_columns, context_columns, 2, None)

def test__assemble_sequences_entity_and_time_segment_size():
    """If entity columns and segment_size, group by and then segment."""
    entity_columns = ['a']
    context_columns = []

    data = pd.DataFrame({
        'a': [1, 1, 1, 1],
        'b': [1, 2, 3, 4],
        'c': [9, 8, 7, 6],
        'time': pd.date_range(start='2001-01-01', periods=4, freq='1d'),
    })
    out = assemble_sequences(data, entity_columns, context_columns, pd.to_timedelta('2d'), 'time')

    assert isinstance(out, list)
    assert out == [
        {
            'context': [],
            'data': [
                [1, 2],
                [9, 8],
                [pd.to_datetime('2001-01-01'), pd.to_datetime('2001-01-02')]
            ],
        },
        {
            'context': [],
            'data': [
                [3, 4],
                [7, 6],
                [pd.to_datetime('2001-01-03'), pd.to_datetime('2001-01-04')]
            ],
        },
    ]
