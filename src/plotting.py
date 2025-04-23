import numpy as np
import torch
import matplotlib.pyplot as plt
import wandb
import plotly.io as pio
import plotly.graph_objects as go
from lightning.pytorch.loggers import WandbLogger
from typing import Tuple


metric_names = ["MSE", "abs_target_mean", "cross_correlation"]

name_to_title = {"MSE": "MSE", "abs_target_mean": "MAE", "cross_correlation": "Pearson"}

types = ["best", "worst", "median"]


def find_best_worst_median_idx(
    metrics: list[float], best_is_highest: bool = False
) -> Tuple[int, int, int]:
    # return min_idx, max_idx, median_idx
    arr = np.array(metrics)
    sorted_indices = np.argsort(arr)
    if best_is_highest:
        best_idx = sorted_indices[0]
        worst_idx = sorted_indices[-1]
    else:
        best_idx = sorted_indices[-1]
        worst_idx = sorted_indices[0]

    median_idx = sorted_indices[len(sorted_indices) // 2]

    return [worst_idx, best_idx, median_idx]


def plot_example(
    lookback,
    target,
    prediction,
    metric_name: str,
    metric_value: float,
    type: str,
    use_heart_rate: bool = False,
    freq: int = 25,
    yaxis_name: str = "Heartrate",
    batch_idx: int = 0,
):
    if use_heart_rate:
        t_lookback = np.arange(-(len(lookback) - 1), 1) * 2
        t_future = np.arange(0, len(target)) * 2
    else:
        t_lookback = np.linspace(-len(lookback) // freq, 0, len(lookback))
        t_future = np.linspace(0, len(target) // freq, len(target))
    fig = go.Figure()

    # Lookback history (blue)
    fig.add_trace(
        go.Scatter(
            x=t_lookback,
            y=lookback,
            mode="lines",
            name="Lookback",
            line=dict(color="#0000ff"),
        )
    )

    # Ground truth future (black)
    fig.add_trace(
        go.Scatter(
            x=t_future,
            y=target,
            mode="lines",
            name="Ground Truth",
            line=dict(color="#ffa000"),
        )
    )

    # Model prediction future (purple)
    fig.add_trace(
        go.Scatter(
            x=t_future,
            y=prediction,
            mode="lines",
            name="Prediction",
            line=dict(
                color="#ff0069",
            ),
        )
    )

    title = f"Batch {batch_idx} | {type} {metric_name}: {metric_value}"

    fig.update_layout(
        title=title,
        xaxis_title="Seconds",
        yaxis_title=yaxis_name,
        legend=dict(x=0.01, y=0.99, borderwidth=1),
    )

    # plt.close(fig)
    return fig


def plot_prediction_wandb(
    x: torch.Tensor,
    y: torch.Tensor,
    preds: torch.Tensor,
    wandb_logger: WandbLogger,
    metrics: dict,
    batch_idx: int,
    use_heart_rate: bool = False,
    freq: int = 25,
    yaxis_name: str = "Heartrate",
):
    look_back_window = x.cpu().detach()
    target = y.cpu().detach()
    prediction = preds.cpu().detach()

    # all_figs = []
    for metric_name in metric_names:
        indices = find_best_worst_median_idx(
            metrics[metric_name], True if metric_name == "cross_correlation" else False
        )

        for type, idx in zip(types, indices):
            metric_value = metrics[metric_name][idx]
            fig = plot_example(
                look_back_window[idx][:, 0],
                target[idx][:, 0],
                prediction[idx][:, 0],
                name_to_title[metric_name],
                metric_value,
                type,
                use_heart_rate,
                freq,
                yaxis_name,
                batch_idx,
            )
            # all_figs.append(fig)
            wandb_logger.experiment.log({"test/predictions": wandb.Image(fig)})

        metric = metrics[metric_name]
        wandb_logger.experiment.log(
            {
                f"test/batch_{batch_idx}/best_{metric_name}": metric[indices[1]],
                f"test/batch_{batch_idx}/worst_{metric_name}": metric[indices[0]],
                f"test/batch_{batch_idx}/median_{metric_name}": metric[2],
            }
        )
    """
    with open(f"all_predictions_batch_{batch_idx}.html", "w") as f:
        for fig in all_figs:
            f.write(pio.to_html(fig, full_html=False, include_plotlyjs="cdn"))
    """


if __name__ == "__main__":
    # test plotting functionality
    x = torch.randn((3, 96, 2))
    y = torch.randn((3, 32, 2))
    preds = torch.randn((3, 32, 2))

    metrics = {
        "cross_correlation": [0.1, 0.7, 0.3],
        "MSE": [0.9, 3.3, 11],
        "abs_target_mean": [0.3, 2, 0.87],
    }

    plot_prediction_wandb(
        x, y, preds, None, metrics, 0, False, freq=32, yaxis_name="PPG"
    )
