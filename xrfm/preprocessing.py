"""Tie-aware RankGauss preprocessing for xRFM.

Drop-in alternative to the README's StandardScaler + one-hot recommendation, aimed at
small-n / heavy-tailed / monotone-warped marginals:

  - rank channel: Phi^{-1}((rank + 1/2) / n) with tie-aware 'min' ranks
    (rank(v) = #{strictly smaller values}); exactly invariant to any strictly increasing
    per-column transformation of the inputs, end-to-end through kernel evaluation,
    bandwidth selection, AGOP iterations, and model selection.
  - optional z channel: standardized raw values (breaks exact invariance; carries real
    signal mainly on regression tasks — make the trade consciously).
  - optional frequency channel: z-scored per-value counts (invariant; encodes categorical
    structure without one-hot dimension blow-up).

WARNING (the tie hazard this module exists to prevent): computing ranks via
argsort().argsort() or any sort-based rank assigns TIED values distinct, row-order-dependent
ranks. On class-sorted or otherwise structured tables this silently leaks row position — which
can be the label — into every high-tie (e.g. categorical) column. We measured AUC 0.09 vs 0.99
on kr-vs-kp from this defect alone. Tie-aware counts (searchsorted, side='left') are both the
statistically correct and the provably invariant choice.

Evidence (xRFM 0.4.5, 256-row supports, 189 OpenML datasets from CC18/CTR23/Grinsztajn/
TabZilla/AMLB, tuned via the package's own random search): replacing standardization-only
inputs with rank+z+frequency channels improves mean skill by +0.007 (104/83 wins, Wilcoxon
p = 0.018). Caveats: that baseline omits one-hot expansion (so it understates the native
pipeline on categorical-heavy data), and at 256 rows the xRFM tree does not split, so the
comparison exercises the leaf RFM rather than the tree-partitioned method. Under adversarial strictly-increasing marginal warps the gap is far larger: the
rank channel is exactly invariant (99% of tasks move < 1e-3 in skill, worst case 2e-3) while
standardized inputs lose 0.038 mean and up to 0.39 worst-case skill.

Example
-------
>>> from xrfm import xRFM, RankGaussTransform
>>> tf = RankGaussTransform(with_z=True, with_freq=True).fit(X_train)
>>> model = xRFM(...).fit((tf.transform(X_train), y_train), (tf.transform(X_val), y_val))
"""
import numpy as np

try:
    from scipy.special import erfinv as _erfinv
except ImportError:  # pragma: no cover
    _erfinv = None


def _phi_inv(u):
    if _erfinv is not None:
        return np.sqrt(2.0) * _erfinv(2.0 * u - 1.0)
    import torch
    return (np.sqrt(2.0) * torch.erfinv(torch.as_tensor(2.0 * u - 1.0))).numpy()


class RankGaussTransform:
    """Sklearn-style transformer: tie-aware rank-gauss (+ optional z / frequency channels).

    Parameters
    ----------
    with_z : bool, default=True
        Append the standardized raw channel (breaks exact monotone-warp invariance;
        adds signal mainly on regression).
    with_freq : bool, default=False
        Append the z-scored value-frequency channel (invariant; useful for categoricals).
    """

    def __init__(self, with_z=True, with_freq=False):
        self.with_z = with_z
        self.with_freq = with_freq

    def fit(self, X, y=None):
        X = np.asarray(X, np.float64)
        self.n_, d = X.shape
        med = np.zeros(d)
        for j in range(d):  # element-selecting median keeps imputation inside observed values
            fin = np.sort(X[np.isfinite(X[:, j]), j])
            med[j] = fin[(len(fin) - 1) // 2] if len(fin) else 0.0
        self.med_ = med
        Xf = np.where(np.isfinite(X), X, med[None, :])
        self.sorted_ = [np.sort(Xf[:, j]) for j in range(d)]
        self.mu_, sd = Xf.mean(0), Xf.std(0)
        self.sd_ = np.where(sd < 1e-6, 1e-6, sd)
        if self.with_freq:
            self.fmap_ = []
            for j in range(d):
                v, c = np.unique(Xf[:, j], return_counts=True)
                self.fmap_.append((v, c / self.n_))
            fr = self._freq(Xf)
            self.fmu_, fsd = fr.mean(0), fr.std(0)
            self.fsd_ = np.where(fsd < 1e-6, 1e-6, fsd)
        return self

    def _freq(self, X):
        out = np.zeros_like(X)
        for j, (v, f) in enumerate(self.fmap_):
            idx = np.searchsorted(v, X[:, j])
            idx = np.clip(idx, 0, len(v) - 1)
            hit = v[idx] == X[:, j]
            out[:, j] = np.where(hit, f[idx], 0.0)
        return out

    def transform(self, X):
        X = np.asarray(X, np.float64)
        Xf = np.where(np.isfinite(X), X, self.med_[None, :])
        parts = []
        R = np.zeros_like(Xf)
        for j, srt in enumerate(self.sorted_):
            r = np.searchsorted(srt, Xf[:, j], side="left")  # tie-aware, step-function extension
            R[:, j] = np.clip((r + 0.5) / self.n_, 1e-6, 1 - 1e-6)
        parts.append(_phi_inv(R))
        if self.with_z:
            parts.append((Xf - self.mu_) / self.sd_)
        if self.with_freq:
            parts.append((self._freq(Xf) - self.fmu_) / self.fsd_)
        return np.concatenate(parts, axis=1).astype(np.float32)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
