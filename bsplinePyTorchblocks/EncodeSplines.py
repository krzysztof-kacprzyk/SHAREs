# EncodeSplines
import torch

from .BSpline import *
from .helper import *


class EncodeSplines(object):
    """Transformer (class) for computing the B-spline basis functions.

    # Shape
        The final result is an array with a new axis:
        - `N x D -> N x D x n_bases`
        - `N x L x D -> N x L x D x n_bases`
        N: number of samples
        D: number of channels, or say depth
        L: length / number of features
        n_bases: number of basis functions

    # Args
        n_bases: int; Number of basis functions.
        degree: int; 2 for quadratic, 3 for qubic splines
        share_knots: bool; if True, the spline knots are
            shared across all the features (last-dimension)

    # Methods
        **fit**: Calculate the knot placement from the values ranges.
        **transform**: Obtain the transformed values
        **fit_transform**: fit and transform.
    """

    def __init__(self, n_bases=5, degree=3, share_knots=False):
        self.n_bases = n_bases
        self.degree = degree
        self.share_knots = share_knots
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, x):
        """Calculate the knot placement from the values ranges.
        # Args
            x: torch.tensor, either N x D or N x L x D dimensional.
        """
        assert x.ndim > 1
        self.data_min_ = torch.amin(x, dim=tuple(range(x.ndim - 1)))
        self.data_max_ = torch.amax(x, dim=tuple(range(x.ndim - 1)))

        if self.share_knots:
            self.data_min_[:] = torch.amin(self.data_min_)
            self.data_max_[:] = torch.amax(self.data_max_)

    def transform(self, x, warn=True):
        """Obtain the transformed values"""
        # 1. split across last dimension
        # 2. re-use ranges
        # 3. Merge
        array_list = [
            encodeSplines(
                x[..., i].reshape((-1, 1)),
                n_bases=self.n_bases,
                spline_order=self.degree,
                warn=warn,
                start=self.data_min_[i],
                end=self.data_max_[i],
            ).reshape(x[..., i].shape + (self.n_bases,))
            for i in range(x.shape[-1])
        ]
        return torch.stack(array_list, axis=-2)

    def fit_transform(self, x):
        """Fit and transform."""
        self.fit(x)
        return self.transform(x)

def encodeSplines(x, n_bases=5, spline_order=3, start=None, end=None, warn=True):
    """Function for the class `EncodeSplines`.
    Expansion by generating B-spline basis functions for each x
    and each n (spline-index) with `scipy.interpolate.splev`,
    based on the pre-placed equidistant knots on [start, end] range.

    # Arguments
        x: a torch.tensor of positions
        n_bases int: Number of spline bases.
        spline_order: 2 for quadratic, 3 for qubic splines
        start, end: range of values. If None, they are inferred from the data
        as minimum and maximum value.
        warn: Show warnings.

    # Returns
        `torch.tensor` of shape `(x.shape[0], x.shape[1], channels, n_bases)`
    """

    if len(x.shape) == 1:
        x = x.reshape((-1, 1))

    if start is None:
        start = torch.amin(x)  # should be np.nanmin
    else:
        if x.min() < start:
            if warn:
                print(
                    "WARNING, x.min() < start for some elements. Truncating them to start: x[x < start] = start"
                )
            x = _trunc(x, minval=start)
    if end is None:
        end = torch.amax(x)  # should be np.nanmax
    else:
        if x.max() > end:
            if warn:
                print(
                    "WARNING, x.max() > end for some elements. Truncating them to end: x[x > end] = end"
                )
            x = _trunc(x, maxval=end)
    bs = BSpline(start, end, n_bases=n_bases, spline_order=spline_order)

    # concatenate x to long
    assert len(x.shape) == 2
    n_rows = x.shape[0]
    n_cols = x.shape[1]

    x_long = x.reshape((-1,))

    # shape = (n_rows * n_cols, n_bases)
    x_feat = bs.predict(x_long, add_intercept=False)

    x_final = x_feat.reshape((n_rows, n_cols, n_bases))
    return x_final
