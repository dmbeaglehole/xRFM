import torch
from typing import Optional


class FeatureSelector:
    """
    Utility to keep track of which features remain after pre-selection.

    The selector keeps the indices of the retained features in their original order,
    applies the projection before fitting/predicting, and can inflate matrices back
    to the full dimensionality.
    """

    def __init__(self, active_indices: torch.Tensor, original_dim: int):
        if active_indices.ndim != 1:
            raise ValueError("active_indices must be a 1D tensor of feature indices.")
        if active_indices.numel() == 0:
            raise ValueError("FeatureSelector requires at least one active feature.")

        self.original_dim = int(original_dim)
        self.active_indices = torch.as_tensor(active_indices, dtype=torch.long, device='cpu')
        self.active_dim = int(self.active_indices.numel())

    @staticmethod
    def _extract_diag_weights(agop: torch.Tensor) -> torch.Tensor:
        if agop is None:
            raise ValueError("AGOP tensor is required for feature selection.")
        if agop.dim() == 1:
            diag = agop
        elif agop.dim() == 2 and agop.shape[0] == agop.shape[1]:
            diag = torch.diagonal(agop)
        else:
            raise ValueError("Unsupported AGOP tensor shape for feature selection.")
        return torch.abs(diag)

    @classmethod
    def from_agop(cls, agop: torch.Tensor, fraction: float, original_dim: int) -> "FeatureSelector":
        """
        Build a selector by keeping the minimum number of coordinates whose diagonal
        mass of the AGOP covers ``fraction`` of the total weight.
        """
        diag_weights = cls._extract_diag_weights(agop)
        total_weight = torch.sum(diag_weights)

        if total_weight <= 0:
            keep_indices = torch.arange(diag_weights.shape[0], device=diag_weights.device)
        else:
            sorted_values, sorted_indices = torch.sort(diag_weights, descending=True)
            cumulative = torch.cumsum(sorted_values, dim=0)
            target_weight = fraction * total_weight
            reach = torch.nonzero(cumulative >= target_weight, as_tuple=False)
            if reach.numel() == 0:
                keep_count = int(sorted_values.numel())
            else:
                keep_count = int(reach[0].item()) + 1
            keep_count = min(max(keep_count, 1), sorted_values.numel())
            keep_indices = sorted_indices[:keep_count]

        # Keep features in their original order to preserve column semantics.
        keep_indices = torch.sort(keep_indices).values
        return cls(keep_indices.cpu(), original_dim=original_dim)

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        """
        Project ``X`` onto the retained features. If ``X`` is already reduced to
        the active dimensionality, it is returned unchanged.
        """
        if X.shape[1] == self.active_dim:
            return X
        if X.shape[1] != self.original_dim:
            raise ValueError(
                f"Expected feature dimension {self.original_dim} before selection, "
                f"got {X.shape[1]}."
            )
        indices = self.active_indices.to(X.device)
        return X.index_select(1, indices)

    def inflate_agop(self, agop: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Expand a reduced AGOP matrix back to the original dimensionality by
        inserting zeros for the dropped coordinates.
        """
        if agop is None:
            return None
        if agop.dim() == 1:
            full = torch.zeros(self.original_dim, dtype=agop.dtype, device=agop.device)
            indices = self.active_indices.to(agop.device)
            full[indices] = agop
            return full
        if agop.dim() == 2 and agop.shape[0] == agop.shape[1] == self.active_dim:
            full = torch.zeros(
                self.original_dim, self.original_dim, dtype=agop.dtype, device=agop.device
            )
            indices = self.active_indices.to(agop.device)
            full[indices[:, None], indices] = agop
            return full
        if agop.dim() == 2 and agop.shape[0] == agop.shape[1] == self.original_dim:
            # Already full size
            return agop
        raise ValueError("Unsupported AGOP shape for inflation.")
