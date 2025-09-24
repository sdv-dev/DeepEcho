import os

# Enable fallback so ops not implemented on MPS run on CPU
# https://github.com/pytorch/pytorch/issues/77764
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import platform
import warnings

import torch


def _validate_gpu_parameters(enable_gpu, cuda):
    """Validate both the `enable_gpu` and `cuda` parameters.

    The logic here is to:
    - Raise a warning if `cuda` is set because it's deprecated.
    - Raise an error if both parameters are set in a conflicting way.
    - Return the resolved `enable_gpu` value.
    """
    if cuda is not None:
        warnings.warn(
            '`cuda` parameter is deprecated and will be removed in a future release. '
            'Please use `enable_gpu` instead.',
            FutureWarning,
        )
        if not enable_gpu:
            raise ValueError(
                'Cannot resolve the provided values of `cuda` and `enable_gpu` parameters. '
                'Please use only `enable_gpu`.'
            )

        enable_gpu = cuda

    return enable_gpu


def _set_device(enable_gpu):
    """Set the torch device based on the `enable_gpu` parameter and system capabilities."""
    if enable_gpu:
        if (
            platform.machine() == 'arm64'
            and getattr(torch.backends, 'mps', None)
            and torch.backends.mps.is_available()
        ):
            device = 'mps'
        else:
            device = 'cpu'
    else:  # Linux/Windows
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return torch.device(device)


def validate_and_set_device(enable_gpu, cuda):
    """Validate the GPU parameters and set the torch device accordingly."""
    enable_gpu = _validate_gpu_parameters(enable_gpu, cuda)
    return _set_device(enable_gpu)
