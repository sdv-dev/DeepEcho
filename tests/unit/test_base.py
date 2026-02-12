"""Unit tests for the ``base`` module."""

import pytest

from deepecho.models.base import _format_score


@pytest.mark.parametrize(
    'score, expected',
    [
        (0, '+00.00'),
        (1.233434, '+01.23'),
        (-0.93, '-00.93'),
        (0.01, '+00.01'),
        (-1.21, '-01.21'),
        (99.99, '+99.99'),
        (-99.99, '-99.99'),
        (150, '+99.99'),
        (-200, '-99.99'),
    ],
)
def test__format_score(score, expected):
    """Test the ``_format_score`` method."""
    result = _format_score(score)
    assert result == expected
    assert len(result) == 6
