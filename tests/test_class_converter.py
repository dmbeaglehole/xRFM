import torch
import pytest

from xrfm.rfm_src.class_conversion import ClassificationConverter


def make_labels(counts):
    """Build labels tensor according to class counts."""
    K = len(counts)
    return torch.repeat_interleave(torch.arange(K, dtype=torch.long), torch.tensor(counts, dtype=torch.long))


@pytest.mark.parametrize("counts", [
    [7, 3],             # K=2
    [5, 2, 3],          # K=3
    [5, 1, 4, 2, 3],    # K=5
])
def test_simplex_equidistance(counts):
    labels = make_labels(counts)
    K = len(counts)
    conv = ClassificationConverter(mode='prevalence', labels=labels, n_classes=K)

    # Codes for each class index 0..K-1
    class_ids = torch.arange(K, dtype=torch.long)
    codes = conv.labels_to_numerical(class_ids)  # shape: (K, K-1)
    assert codes.shape == (K, K - 1), "Codes should live in K-1 dimensions."

    # Pairwise squared distances
    diffs = codes.unsqueeze(1) - codes.unsqueeze(0)  # (K, K, K-1)
    dists2 = (diffs ** 2).sum(dim=-1)               # (K, K)

    # Exclude diagonal and check all off-diagonal distances are (nearly) equal
    off_diag = ~torch.eye(K, dtype=torch.bool)
    off_vals = dists2[off_diag]
    assert torch.isfinite(off_vals).all()
    tol = 1e-6
    assert (off_vals.max() - off_vals.min()) < tol, "Class codes are not equidistant."


@pytest.mark.parametrize("counts", [
    [7, 3],           # K=2
    [5, 2, 3],        # K=3
    [5, 1, 4, 2, 3],  # K=5
])
def test_encode_decode_roundtrip(counts):
    labels = make_labels(counts)
    K = len(counts)
    conv = ClassificationConverter(mode='prevalence', labels=labels, n_classes=K)

    # Encode each class exactly once and decode back
    class_ids = torch.arange(K, dtype=torch.long)
    codes = conv.labels_to_numerical(class_ids)            # (K, K-1)
    probs = conv.numerical_to_probas(codes, eps=0.0)       # allow exact 0/1 without clamping
    # Should be (close to) one-hot
    eye = torch.eye(K)
    assert torch.allclose(probs, eye, atol=1e-6, rtol=0), "Round-trip did not recover one-hot probabilities."


@pytest.mark.parametrize("counts", [
    [7, 3],           # K=2
    [5, 2, 3],        # K=3
    [5, 1, 4, 2, 3],  # K=5
])
def test_zero_maps_to_empirical_prior(counts):
    labels = make_labels(counts)
    K = len(counts)
    conv = ClassificationConverter(mode='prevalence', labels=labels, n_classes=K)

    prior = torch.tensor(counts, dtype=torch.float32)
    prior = prior / prior.sum()

    # Decode the origin; do multiple rows to also test batch handling
    zeros = torch.zeros(3, K - 1)   # three samples at the origin in R^{K-1}
    probs = conv.numerical_to_probas(zeros, eps=0.0)  # exact mapping, no clamp
    assert probs.shape == (3, K)
    # Every row should equal the empirical prior
    assert torch.allclose(probs, prior.expand_as(probs), atol=1e-6, rtol=0), "Origin did not map to empirical prior."
