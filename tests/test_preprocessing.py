"""Property tests for the RankGauss transformer — the invariance guarantee, asserted bitwise."""
import numpy as np
import pytest

from xrfm.preprocessing import RankGaussTransform

WARPS = {
    "exp": lambda v: np.exp(1.5 * v),
    "cubic": lambda v: v ** 3 + 0.01 * v,
    "arcsinh": lambda v: np.arcsinh(8.0 * v),
    "affine": lambda v: 3.7 * v - 2.0,
}


def _data(seed=0, n=80, d=5):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    X[:, d - 1] = rng.randint(0, 4, n)      # tie-heavy (categorical-like) column
    return X


@pytest.mark.parametrize("warp", list(WARPS))
@pytest.mark.parametrize("with_freq", [False, True])
def test_invariant_channels_are_bitwise_invariant(warp, with_freq):
    """Rank (+frequency) channels are EXACTLY unchanged by strictly increasing marginal maps."""
    X = _data()
    Xw = np.column_stack([WARPS[warp](X[:, j]) for j in range(X.shape[1])])
    t = lambda: RankGaussTransform(with_z=False, with_freq=with_freq)
    assert np.array_equal(t().fit_transform(X), t().fit_transform(Xw))


@pytest.mark.parametrize("warp", ["exp", "cubic"])
def test_z_channel_breaks_invariance(warp):
    """The documented trade: adding the standardized channel is NOT warp-invariant."""
    X = _data()
    Xw = np.column_stack([WARPS[warp](X[:, j]) for j in range(X.shape[1])])
    t = lambda: RankGaussTransform(with_z=True, with_freq=False)
    assert not np.array_equal(t().fit_transform(X), t().fit_transform(Xw))


def test_ties_share_one_rank():
    """Tie-aware ranks: equal inputs must map to equal outputs (the label-leak guard)."""
    X = np.array([[1.0], [1.0], [1.0], [2.0], [2.0], [3.0]])
    Z = RankGaussTransform(with_z=False).fit_transform(X)
    assert Z[0, 0] == Z[1, 0] == Z[2, 0]
    assert Z[3, 0] == Z[4, 0]
    assert Z[0, 0] < Z[3, 0] < Z[5, 0]


def test_sort_based_ranks_would_leak():
    """Demonstrates the hazard this module avoids: argsort-based ranks separate tied values,
    encoding row order (== the label, on class-sorted data)."""
    col = np.array([0.0] * 5 + [1.0] * 5)          # class-sorted, all ties within class
    sort_ranks = col.argsort().argsort()            # the buggy encoding
    assert len(np.unique(sort_ranks[:5])) == 5      # 5 distinct ranks for 5 identical values
    Z = RankGaussTransform(with_z=False).fit_transform(col.reshape(-1, 1))
    assert len(np.unique(Z[:5, 0])) == 1            # tie-aware: one rank, no row-order signal


def test_transform_on_unseen_values_is_monotone():
    """Inductive use: searchsorted extension keeps order on out-of-sample values."""
    t = RankGaussTransform(with_z=False).fit(_data())
    probe = np.array([[-5.0, 0, 0, 0, 1], [0.0, 0, 0, 0, 1], [5.0, 0, 0, 0, 1]])
    z = t.transform(probe)[:, 0]
    assert z[0] <= z[1] <= z[2]


def test_nan_handling():
    X = _data(); X[3, 1] = np.nan
    Z = RankGaussTransform(with_z=True, with_freq=True).fit_transform(X)
    assert np.isfinite(Z).all()
