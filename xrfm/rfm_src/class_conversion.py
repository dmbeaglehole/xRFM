from typing import Literal
import torch

class ClassificationConverter:
    def __init__(self, mode: Literal['zero_one', 'prevalence'], labels: torch.Tensor, n_classes: int):
        """
        Args:
            mode:
              - 'zero_one': binary -> {0,1}; multiclass -> one-hot.
              - 'prevalence': encode classes to a regular simplex in R^{K-1}, then shift so the
                empirical class prior maps to the origin (so a prediction of 0 decodes to the prior).
            labels: tensor of shape (n_samples,) with dtype torch.long
            n_classes: number of classes K (K >= 2)
        """
        assert mode in ['zero_one', 'prevalence']
        assert n_classes >= 2, "n_classes must be at least 2."
        self.mode = mode
        self.labels = labels
        self.n_classes = n_classes

        # Basics
        self._device = labels.device
        self._dtype = torch.float32

        # Empirical prior π (length-K)
        counts = torch.bincount(labels, minlength=n_classes).float()
        total = counts.sum().clamp_min(1.0)
        self._prior = counts / total

        # Precompute regression coding/decoding pieces
        self._C = None     # K x (K-1) codes (rows = class codes)
        self._invA = None  # (K x K) inverse for decoding: A = [C^T; 1^T]
        if self.mode == 'prevalence':
            K = self.n_classes
            I = torch.eye(K, dtype=self._dtype, device=self._device)

            # Orthonormal basis of the sum-zero subspace via QR on [e_i - e_K] (K x (K-1))
            M = I[:, :-1] - I[:, [-1]]       # columns span the sum-zero subspace
            Q, _ = torch.linalg.qr(M, mode='reduced')  # Q is K x (K-1), columns orthonormal

            # Rows of Q are the vertices of a regular simplex (equal pairwise distances) centered at 0 (for uniform prior)
            C0 = Q

            # Shift so the empirical prior maps to the origin
            mu = self._prior @ C0            # shape: (K-1,)
            self._C = C0 - mu                # row-wise shift, K x (K-1)

            # Precompute inverse for decoding
            A = torch.cat([self._C.T, torch.ones(1, K, dtype=self._dtype, device=self._device)], dim=0)  # K x K
            self._invA = torch.linalg.inv(A)

    def labels_to_numerical(self, labels: torch.Tensor) -> torch.Tensor:
        if self.mode == 'prevalence':
            # Unified: works for K=2 as well (then output is (N,1))
            return self._C[labels]
        else:  # 'zero_one'
            if self.n_classes == 2:
                return labels.float().unsqueeze(-1)     # (N,1) with values {0,1}
            else:
                return torch.nn.functional.one_hot(labels, num_classes=self.n_classes).float()

    def numerical_to_probas(self, num: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        if self.mode == 'prevalence':
            # Ensure shape (N, K-1)
            if num.ndim == 1:
                num = num.unsqueeze(-1)
            N = num.shape[0]
            ones = torch.ones((N, 1), dtype=self._dtype, device=num.device)
            B = torch.cat([num, ones], dim=1)                 # (N, K)
            pi = B @ self._invA.T                             # (N, K)
            pi = torch.clamp(pi, eps, 1 - eps)
            pi = pi / pi.sum(dim=1, keepdim=True)
            return pi
        else:
            # Original simple path
            if num.shape[1] == 1:
                num = torch.cat([1 - num, num], dim=1)
            num = torch.clamp(num, eps, 1 - eps)
            num = num / num.sum(dim=1, keepdim=True)
            return num

    def numerical_to_labels(self, num: torch.Tensor) -> torch.Tensor:
        # Always decode via probabilities for consistency
        probs = self.numerical_to_probas(num)
        return probs.argmax(dim=-1)
