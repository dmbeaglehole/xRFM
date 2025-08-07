from typing import List, Literal

import torch
import sklearn
import time

from sklearn.metrics import roc_auc_score, mean_squared_error


class Metric:
    name: str
    display_name: str
    should_maximize: bool
    task_types: List[Literal['reg', 'class']]
    required_quantities: List[Literal['y_true', 'y_pred', 'y_pred_proba', 'agop', 'topk']]

    def __init__(self):
        assert self.compute == Metric.compute  # should not be overridden

    def compute(self, **kwargs) -> float:
        for q in self.required_quantities:
            if q not in kwargs:
                raise ValueError(f'Need to pass parameter {q} for metric {self.name}')

        return self._compute(**kwargs)

    def _compute(self, **kwargs) -> float:
        raise NotImplementedError()

    @staticmethod
    def from_name(name: str) -> 'Metric':
        all_metrics = [MSE]  # todo: populate
        all_metrics_dict = {m.name: m for m in all_metrics}
        return all_metrics_dict[name]()


class MSE(Metric):
    name = 'mse'
    display_name = 'MSE'
    should_maximize = False
    task_types = ['reg']
    required_quantities = ['y_true', 'y_pred']

    def _compute(self, **kwargs) -> float:
        return mean_squared_error(kwargs['y_true'], kwargs['y_pred'])


class Metrics:
    def __init__(self, names: List[str]):
        self.names = names
        self.metrics = [Metric.from_name(name) for name in names]
        self.required_quantities = list(set.union(*[set(m.required_quantities) for m in self.metrics]))

    def compute(self, **kwargs):
        for q in self.required_quantities:
            if q not in kwargs:
                raise ValueError(f'Need to pass parameter {q} for metric {self.name}')

        return {m.name: m.compute(**kwargs) for m in self.metrics}


def compute_metric(metrics: List[str], y_true: torch.Tensor, y_pred: torch.Tensor, **kwargs):
    """
    Evaluate model performance using specified metrics.

    Parameters
    ----------
    samples : torch.Tensor
        Input samples of shape (n_samples, n_features)
    targets : torch.Tensor
        Target values of shape (n_samples, n_outputs)
    metrics : list of str
        List of metrics to compute. Supported metrics:
        - 'accuracy': Classification accuracy
        - 'mse': Mean squared error
        - 'f1': F1 score
        - 'auc': Area under ROC curve
        - 'top_agop_vector_auc': AUC using top AGOP eigenvector projection
        - 'top_agop_vector_pearson_r': Pearson correlation of targets with top AGOP eigenvector projection
        - 'top_agop_vectors_ols_auc': AUC using OLS regression on top AGOP eigenvectors

    Returns
    -------
    dict
        Dictionary mapping metric names to their computed values
    """

    out_metrics = {}
    if 'accuracy' in metrics:
        preds = self.predict_proba(samples.to(self.device)).to(targets.device)
        preds_ = torch.argmax(preds, dim=-1)
        targets_ = torch.argmax(targets, dim=-1)
        num_classes = preds.shape[-1]

        if num_classes == 2:
            out_metrics['accuracy'] = accuracy(preds_, targets_, task="binary").item()
        else:
            out_metrics['accuracy'] = accuracy(preds_, targets_, task="multiclass", num_classes=num_classes).item()

    if 'mse' in metrics:
        preds = self.predict(samples.to(self.device)).to(targets.device)
        out_metrics['mse'] = (targets - preds).pow(2).mean()

    if 'f1' in metrics:
        preds = self.predict_proba(samples.to(self.device)).to(targets.device)
        if targets.shape[1] == 1:
            # assume binary classification
            targets = torch.cat([1 - targets, targets], dim=1)
        out_metrics['f1'] = f1_score(preds, targets, num_classes=preds.shape[-1]).item()

    if 'auc' in metrics:
        preds = self.predict_proba(samples.to(self.device))
        if targets.shape[1] == 1:
            # assume binary classification
            targets = torch.cat([1 - targets, targets], dim=1)
        out_metrics['auc'] = roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy(), multi_class='ovr')

    if 'top_agop_vector_auc' in metrics:
        assert len(targets.shape) == 1 or targets.shape[
            1] == 1, "Top AGOP Vector AUC is only defined for binary classification"
        _, U = torch.lobpcg(self.agop, k=1)
        top_eigenvector = U[:, 0]
        projections = samples @ top_eigenvector
        projections = projections.reshape(targets.shape)
        plus_auc = roc_auc_score(targets.cpu().numpy(), torch.sigmoid(projections).cpu().numpy())
        minus_auc = roc_auc_score(targets.cpu().numpy(), torch.sigmoid(-projections).cpu().numpy())
        out_metrics['top_agop_vector_auc'] = max(plus_auc, minus_auc)

    if 'top_agop_vector_pearson_r' in metrics:
        assert len(targets.shape) == 1 or targets.shape[
            1] == 1, "Top AGOP Vector Pearson R is only defined for binary classification"
        _, U = torch.lobpcg(self.agop, k=1)
        top_eigenvector = U[:, 0]
        projections = samples @ top_eigenvector
        projections = projections.reshape(-1, 1)
        targets = targets.reshape(-1, 1)
        out_metrics['top_agop_vector_pearson_r'] = \
        torch.abs(torch.corrcoef(torch.cat((projections, targets), dim=-1).T))[0, 1].item()

    if 'top_agop_vectors_ols_auc' in metrics:
        top_k = self.top_k
        print(f"Computing Top AGOP Vectors OLS AUC for {top_k} eigenvectors")
        start_time = time.time()
        _, U = torch.lobpcg(self.agop, k=top_k)
        end_time = time.time()
        print(f"Time taken to compute top {top_k} eigenvectors: {end_time - start_time} seconds")

        top_eigenvectors = U[:, :top_k]
        projections = samples @ top_eigenvectors
        projections = projections.reshape(-1, top_k)

        start_time = time.time()
        XtX = projections.T @ projections
        Xty = projections.T @ targets
        end_time = time.time()
        print(f"Time taken to compute XtX and Xty: {end_time - start_time} seconds")

        start_time = time.time()
        betas = torch.linalg.pinv(XtX) @ Xty
        end_time = time.time()
        print(f"Time taken to solve OLS: {end_time - start_time} seconds")

        start_time = time.time()
        preds = torch.sigmoid(projections @ betas).reshape(targets.shape)
        end_time = time.time()
        print(f"Time taken to compute OLS predictions: {end_time - start_time} seconds")

        out_metrics['top_agop_vectors_ols_auc'] = roc_auc_score(targets.cpu().numpy(), preds.cpu().numpy(),
                                                                multi_class='ovr')

    return out_metrics