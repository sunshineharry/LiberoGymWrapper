
import numpy as np
from typing import Optional
from typing import Union, Dict, Callable
import torch
import torch.nn as nn


def at_least_ndim(x: Union[np.ndarray, torch.Tensor, int, float], ndim: int, pad: int = 0):
    """ Add dimensions to the input tensor to make it at least ndim-dimensional.

    Args:
        x: Union[np.ndarray, torch.Tensor, int, float], input tensor
        ndim: int, minimum number of dimensions
        pad: int, padding direction. `0`: pad in the last dimension, `1`: pad in the first dimension

    Returns:
        Any of these 2 options

        - np.ndarray or torch.Tensor: reshaped tensor
        - int or float: input value

    Examples:
        >>> x = np.random.rand(3, 4)
        >>> at_least_ndim(x, 3, 0).shape
        (3, 4, 1)
        >>> x = torch.randn(3, 4)
        >>> at_least_ndim(x, 4, 1).shape
        (1, 1, 3, 4)
        >>> x = 1
        >>> at_least_ndim(x, 3)
        1
    """
    if isinstance(x, np.ndarray):
        if ndim > x.ndim:
            if pad == 0:
                return np.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return np.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, torch.Tensor):
        if ndim > x.ndim:
            if pad == 0:
                return torch.reshape(x, x.shape + (1,) * (ndim - x.ndim))
            else:
                return torch.reshape(x, (1,) * (ndim - x.ndim) + x.shape)
        else:
            return x
    elif isinstance(x, (int, float)):
        return x
    else:
        raise ValueError(f"Unsupported type {type(x)}")

class EmptyNormalizer:
    """ Empty Normalizer

    Does nothing to the input data.
    """

    def normalize(self, x: np.ndarray):
        return x

    def unnormalize(self, x: np.ndarray):
        return x

class MinMaxNormalizer(EmptyNormalizer):
    """ MinMax Normalizer

    Normalizes data from range [min, max] to [-1, 1].
    For those dimensions with zero range, the normalized value will be zero.

    Args:
        X: np.ndarray,
            dataset with shape (..., *x_shape)
        start_dim: int,
            the dimension to start normalization from, Default: -1
        X_max: Optional[np.ndarray],
            Maximum value for each dimension. If None, it will be calculated from X. Default: None
        X_min: Optional[np.ndarray],
            Minimum value for each dimension. If None, it will be calculated from X. Default: None

    Examples:
        >>> x_dataset = np.random.randn(100000, 3, 10)

        >>> x_min = np.random.randn(3, 10)
        >>> normalizer = MinMaxNormalizer(x_dataset, 1, X_min=x_min)
        >>> x = np.random.randn(1, 3, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)

        >>> x_max = np.random.randn(10)
        >>> normalizer = MinMaxNormalizer(x_dataset, 2, X_max=x_max)
        >>> x = np.random.randn(1, 10)
        >>> norm_x = normalizer.normalize(x)
        >>> unnorm_x = normalizer.unnormalize(norm_x)
    """

    def __init__(
            self, X: np.ndarray, start_dim: int = -1,
            X_max: Optional[np.ndarray] = None, X_min: Optional[np.ndarray] = None):
        total_dims = X.ndim
        if start_dim < 0:
            start_dim = total_dims + start_dim

        axes = tuple(range(start_dim))

        self.max = np.max(X, axis=axes) if X_max is None else X_max
        self.min = np.min(X, axis=axes) if X_min is None else X_min
        self.mask = np.ones_like(self.max)
        self.range = self.max - self.min
        self.mask[self.max == self.min] = 0.
        self.range[self.range == 0] = 1.

    def normalize(self, x: np.ndarray):
        ndim = x.ndim
        x = (x - at_least_ndim(self.min, ndim, 1)) / at_least_ndim(self.range, ndim, 1)
        x = x * 2 - 1
        x = x * at_least_ndim(self.mask, ndim, 1)
        return x

    def unnormalize(self, x: np.ndarray):
        ndim = x.ndim
        x = (x + 1) / 2
        x = x * at_least_ndim(self.mask, ndim, 1)
        x = x * at_least_ndim(self.range, ndim, 1) + at_least_ndim(self.min, ndim, 1)
        return x
    
import numba
@numba.jit(nopython=True)
def create_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0, pad_after: int = 0,
        debug: bool = True) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0  # episode start index
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]  # episode end index
        episode_length = end_idx - start_idx  # episode length

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert (start_offset >= 0)
                assert (end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices
