import sys
import warnings

import torch


def _validate_gpu_parameter(enable_gpu, cuda):
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


def _set_device(enable_gpu, device=None):
    if device:
        return torch.device(device)

    if enable_gpu:
        if sys.platform == 'darwin':  # macOS
            if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        else:  # Linux/Windows
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = 'cpu'

    return torch.device(device)


def validate_and_set_device(enable_gpu, cuda):
    enable_gpu = _validate_gpu_parameter(enable_gpu, cuda)
    return _set_device(enable_gpu)
