"""Test the ``_utils`` module."""

import re
import sys
from unittest.mock import patch

import pytest
import torch

from deepecho.models._utils import _get_enable_gpu_value, _set_device, validate_and_set_device


def test__get_enable_gpu_values():
    """Test the ``_get_enable_gpu_value`` method."""
    # Setup
    expected_error = re.escape(
        'Cannot resolve the provided values of `cuda` and `enable_gpu` parameters. '
        'Please use only `enable_gpu`.'
    )
    expected_warning = re.escape(
        '`cuda` parameter is deprecated and will be removed in a future release. '
        'Please use `enable_gpu` instead.'
    )

    # Run
    enable_gpu_1 = _get_enable_gpu_value(False, None)
    enable_gpu_2 = _get_enable_gpu_value(True, None)
    with pytest.warns(FutureWarning, match=expected_warning):
        enable_gpu_3 = _get_enable_gpu_value(True, False)

    with pytest.raises(ValueError, match=expected_error):
        _get_enable_gpu_value(False, True)

    # Assert
    assert enable_gpu_1 is False
    assert enable_gpu_2 is True
    assert enable_gpu_3 is False


def test__set_device():
    """Test the ``_set_device`` method."""
    # Run
    device_1 = _set_device(False)
    device_2 = _set_device(True)

    # Assert
    if (
        sys.platform == 'darwin'
        and getattr(torch.backends, 'mps', None)
        and torch.backends.mps.is_available()
    ):
        expected_device_2 = torch.device('mps')
    elif torch.cuda.is_available():
        expected_device_2 = torch.device('cuda')
    else:
        expected_device_2 = torch.device('cpu')

    assert device_1 == torch.device('cpu')
    assert device_2 == expected_device_2


@patch('deepecho.models._utils._set_device')
@patch('deepecho.models._utils._get_enable_gpu_value')
def test_validate_and_set_device(mock_validate, mock_set_device):
    """Test the ``validate_and_set_device`` method."""
    # Setup
    mock_validate.return_value = True
    mock_set_device.return_value = torch.device('cuda')

    # Run
    device = validate_and_set_device(enable_gpu=True, cuda=None)

    # Assert
    mock_validate.assert_called_once_with(True, None)
    mock_set_device.assert_called_once_with(True)
    assert device == torch.device('cuda')
