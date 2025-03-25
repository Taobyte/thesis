import torch
import lightning as L
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate(model: L.LightningModule, preds: torch.Tensor, y: torch.Tensor):
    metric_mae = mae(preds, y)
    metric_mse = mse(preds, y)
    metric_mape = mape(preds, y)
    model.log("metric_mae", metric_mae)
    model.log("metric_mse", metric_mse)
    model.log("metric_mape", metric_mape)


def mae(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return mean_absolute_error(targets.cpu().numpy(), preds.cpu().numpy())


def mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return mean_squared_error(targets.cpu().numpy(), preds.cpu().numpy())


def rmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return mean_squared_error(targets.cpu().numpy(), preds.cpu().numpy(), squared=False)


def nrmse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return rmse(preds, targets) / torch.mean(targets).item()


def mape(preds: torch.Tensor, targets: torch.Tensor) -> float:
    eps = 1e-8
    return torch.mean(torch.abs((targets - preds) / (targets + eps))).item()


def smape(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (
        200.0
        * torch.mean(
            torch.abs(targets - preds) / (torch.abs(targets) + torch.abs(preds))
        ).item()
    )


def mase(preds: torch.Tensor, targets: torch.Tensor) -> float:
    pass


def r2_score(preds: torch.Tensor, targets: torch.Tensor):
    target_mean = torch.mean(targets)
    ss_tot = torch.sum((targets - target_mean) ** 2)
    ss_res = torch.sum((targets - preds) ** 2)
    return 1 - ss_res / ss_tot
