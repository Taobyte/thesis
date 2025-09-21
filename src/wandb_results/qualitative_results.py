import json
import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots
from typing import Tuple, Union
from numpy.typing import NDArray
from collections import defaultdict
from pathlib import Path

from src.wandb_results.utils import get_runs
from src.constants import (
    MODELS,
    dataset_to_name,
    model_to_name,
    model_colors,
)


# STYLING

TITLE_SIZE = 28
SUBTITLE_SIZE = 20
AXIS_TITLE_SIZE = 16
TICK_SIZE = 8
LEGEND_SIZE = 16
EXO_ENDO_LEGEND_SIZE = 10
GT_LINE_WIDTH = 3.8
PRED_LINE_WIDTH = 2.4
PRED_OPACITY = 0.8
GT_MARKER_SIZE = 6

best_model = {"dalia": "xgboost", "wildppg": "timexer", "ieee": "linear"}


def load_predictions(
    dataset: str,
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    experiment_name: str,
    metric: str = "MAE",
    artifacts_path: str = "C:/Users/cleme/ETH/Master/Thesis/ns-forecast/artifacts",
):
    runs = get_runs(
        dataset,
        look_back_window,
        prediction_window,
        models,
        experiment_name=experiment_name,
        predictions=True,
    )

    loaded_preds = defaultdict(dict)

    for run in runs:
        # download artifcats
        config = run.config
        model = config["model"]["name"]
        print(f"Processing {model_to_name[model]}")
        raw_artifact = next(
            (a for a in run.logged_artifacts() if "predictions" in a.name), None
        )
        if raw_artifact is None:
            print(f"No predictions for run {run.name}")
            continue
        else:
            art_dir = Path(artifacts_path) / str(raw_artifact.name).replace(":", "-")

            if not os.path.exists(art_dir):
                raw_artifact.download()

            json_path = art_dir / "test_predictions_metrics.json"
            npz_path = art_dir / "test_predictions.npz"
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            metrics = obj[metric]
            np_arrays = np.load(npz_path, allow_pickle=True)
            loaded_preds[model]["metrics"] = metrics
            loaded_preds[model]["arrays"] = np_arrays

    return loaded_preds


def get_index_and_gt(
    loaded_preds, dataset: str, plot_type: str, window: int
) -> Tuple[int, NDArray[np.float32], NDArray[np.float32]]:
    best_model_metrics = loaded_preds[best_model[dataset]]["metrics"]
    sorted_idx = np.argsort(best_model_metrics)
    if plot_type == "worst":
        worst_idx = sorted_idx[-1]
    elif plot_type == "best":
        worst_idx = sorted_idx[0]
    else:
        worst_idx = sorted_idx[len(sorted_idx) // 2]

    print(f"worst value: {best_model_metrics[worst_idx]}")

    x = np.arange(window) * 2
    any_preds = loaded_preds[best_model[dataset]]["arrays"]
    gt_series = any_preds["gt_series"]
    n_windows_per_series = [len(s) - window + 1 for s in gt_series]
    cum_lengths = np.cumsum([0] + n_windows_per_series)
    gt_idx = np.searchsorted(cum_lengths, worst_idx, side="right") - 1
    window_pos = worst_idx - cum_lengths[gt_idx]
    gt = gt_series[gt_idx][window_pos : window_pos + window, 0]

    return worst_idx, gt, x


def get_index_and_gt_exo(
    loaded_endo,
    loaded_exo,
    dataset: str,
    plot_type: str,
    window: int,
    use_imprv: bool = False,
) -> Tuple[int, NDArray[np.float32], NDArray[np.float32]]:
    endo_metrics = np.array(loaded_endo[best_model[dataset]]["metrics"])
    exo_metrics = np.array(loaded_exo[best_model[dataset]]["metrics"])
    diff = exo_metrics - endo_metrics
    imprv = diff / endo_metrics

    fig = go.Figure()
    fig.add_histogram(
        x=imprv,
        nbinsx=30,
        opacity=0.75,
    )
    fig.show()
    sorted_idx = np.argsort(imprv if use_imprv else diff)
    if plot_type == "worst":
        worst_idx = sorted_idx[-1]
    elif plot_type == "best":
        worst_idx = sorted_idx[0]
    else:
        worst_idx = sorted_idx[len(sorted_idx) // 2]

    print(f"worst value: {diff[worst_idx]}")

    x = np.arange(window) * 2
    exo_preds = loaded_exo[best_model[dataset]]["arrays"]
    gt_series = exo_preds["gt_series"]
    n_windows_per_series = [len(s) - window + 1 for s in gt_series]
    cum_lengths = np.cumsum([0] + n_windows_per_series)
    gt_idx = np.searchsorted(cum_lengths, worst_idx, side="right") - 1
    window_pos = worst_idx - cum_lengths[gt_idx]
    gt = gt_series[gt_idx][window_pos : window_pos + window, :]

    return worst_idx, gt, x


def plot_predictions(
    datasets: list[str],
    look_back_window: list[int] = [20],
    prediction_window: list[int] = [10],
    models: list[str] = MODELS,
    experiment_name: str = "endo_exo",
    plot_type: str = "worst",
):
    assert len(look_back_window) == 1
    assert len(prediction_window) == 1
    lbw = look_back_window[0]
    pw = prediction_window[0]

    window = lbw + pw

    fig = make_subplots(
        rows=1 if experiment_name == "endo_only" else 2,
        cols=len(datasets),
        shared_xaxes=True,
        subplot_titles=[f"<b>{dataset_to_name[d]}</b>" for d in datasets],
        horizontal_spacing=0.03,
        vertical_spacing=0.1,
        row_heights=1.0 if experiment_name == "endo_only" else [0.8, 0.2],
    )

    for col_idx, dataset in enumerate(datasets, start=1):
        loaded_preds = load_predictions(
            dataset, models, look_back_window, prediction_window, experiment_name
        )

        worst_idx, gt, x = get_index_and_gt(loaded_preds, dataset, plot_type, window)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=gt,
                mode="lines+markers",  # dots + line
                name="Ground truth",
                legendgroup=f"{dataset}-gt",
                showlegend=(col_idx == 1),
                line=dict(width=GT_LINE_WIDTH, color="blue"),
                marker=dict(size=GT_MARKER_SIZE, symbol="circle"),
            ),
            row=1,
            col=col_idx,
        )

        if experiment_name == "endo_exo":
            activity = gt_series[gt_idx][window_pos : window_pos + window, 1]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=activity,
                    mode="lines",  # dots + line
                    name="Activity",
                    legendgroup=f"{dataset}-act",
                    showlegend=(col_idx == 1),
                    line=dict(width=GT_LINE_WIDTH, color="red"),
                ),
                row=2,
                col=col_idx,
            )

        # Add a vertical boundary at LBW
        fig.add_vline(
            x=(lbw - 1) * 2,
            line_dash="dash",
            line_width=1,
            line_color="black",
            row=1,
            col=col_idx,
        )

        for model in loaded_preds.keys():
            # load model artifact & plot prediction
            arrays = loaded_preds[model]["arrays"]
            preds = arrays["preds"]
            pred = preds[worst_idx][:, 0]

            y_pred_full = np.full(window, np.nan, dtype=float)
            y_pred_full[lbw:] = pred[:pw]
            y_pred_full[lbw - 1] = gt[lbw - 1]

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y_pred_full,
                    mode="lines",
                    name=f"{model_to_name[model]}",
                    legendgroup=f"{model_to_name[model]}",
                    showlegend=(col_idx == 1),  # one legend entry across subplots
                    line=dict(width=PRED_LINE_WIDTH, color=model_colors[model]),
                    opacity=PRED_OPACITY,
                ),
                row=1,
                col=col_idx,
            )

    fig.update_layout(
        template="seaborn",
        height=400,  # tweak as you like
        width=1200,  # tweak as you like
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,  # push legend below plot
            xanchor="center",
            x=0.5,
            font=dict(size=LEGEND_SIZE),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=80, r=30, t=90, b=130),
        font=dict(size=16),  # base font (ticks inherit unless overridden)
    )

    # Axis titles & tick label sizes
    fig.update_xaxes(
        title_text="Time (s)",
        title_font=dict(size=AXIS_TITLE_SIZE),
        tickfont=dict(size=TICK_SIZE),
        row=1 if experiment_name == "endo_only" else 2,
    )
    fig.update_yaxes(
        title_text="HR",
        title_font=dict(size=AXIS_TITLE_SIZE),
        tickfont=dict(size=TICK_SIZE),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="IMU",
        title_font=dict(size=AXIS_TITLE_SIZE),
        tickfont=dict(size=TICK_SIZE),
        row=2,
        col=1,
    )

    for col in range(2, len(datasets) + 1):
        fig.update_yaxes(
            tickfont=dict(size=TICK_SIZE),
            col=col,
        )

    # Make all subplot titles bigger
    for a in fig.layout.annotations:
        a.font = dict(size=SUBTITLE_SIZE)

    fig.show()


def _plot_trace(
    fig,
    loaded_preds,
    model: str,
    exo_type: str,
    gt: NDArray[np.float32],
    x: NDArray[np.float32],
    window: int,
    worst_idx: int,
    col_idx: int,
    lbw: int,
    pw: int,
) -> None:
    assert exo_type in ["Endo", "Exo"]
    # load model artifact & plot prediction
    arrays = loaded_preds[model]["arrays"]
    preds = arrays["preds"]
    pred = preds[worst_idx][:, 0]

    y_pred_full = np.full(window, np.nan, dtype=float)
    y_pred_full[lbw:] = pred[:pw]
    y_pred_full[lbw - 1] = gt[lbw - 1]

    mode = "lines+markers" if exo_type == "Exo" else "lines"
    dash = "solid" if exo_type == "Exo" else "dash"

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y_pred_full,
            mode=mode,
            name=f"{model_to_name[model]} {exo_type}",
            legendgroup=f"{model_to_name[model]} {exo_type}",
            showlegend=True,
            line=dict(
                width=PRED_LINE_WIDTH,
                color=model_colors[model],
                dash=dash,
            ),
            opacity=PRED_OPACITY,
        ),
        row=1,
        col=col_idx,
    )


def plot_best_exo_improvement(
    datasets: list[str],
    look_back_window: list[int] = [20],
    prediction_window: list[int] = [10],
    models: list[str] = MODELS,
    metric: str = "MAE",
    plot_type: str = "worst",
    use_imprv: bool = True,
):
    assert len(look_back_window) == 1
    assert len(prediction_window) == 1
    lbw = look_back_window[0]
    pw = prediction_window[0]

    window = lbw + pw

    fig = make_subplots(
        rows=2,
        cols=len(datasets),
        shared_xaxes=True,
        subplot_titles=[f"<b>{dataset_to_name[d]}</b>" for d in datasets],
        horizontal_spacing=0.03,
        vertical_spacing=0.10,
        row_heights=[0.8, 0.2],
    )

    for col_idx, dataset in enumerate(datasets, start=1):
        loaded_exo = load_predictions(
            dataset,
            [best_model[dataset]],
            look_back_window,
            prediction_window,
            "endo_exo",
        )
        loaded_endo = load_predictions(
            dataset,
            [best_model[dataset]],
            look_back_window,
            prediction_window,
            "endo_only",
        )

        idx, gt, x = get_index_and_gt_exo(
            loaded_exo=loaded_exo,
            loaded_endo=loaded_endo,
            dataset=dataset,
            plot_type=plot_type,
            window=window,
            use_imprv=use_imprv,
        )

        hr = gt[:, 0]

        fig.add_trace(
            go.Scatter(
                x=x,
                y=hr,
                mode="lines+markers",  # dots + line
                name="Ground truth",
                legendgroup=f"{dataset}-gt",
                showlegend=(col_idx == 1),
                line=dict(width=GT_LINE_WIDTH, color="blue"),
                marker=dict(size=GT_MARKER_SIZE, symbol="circle"),
            ),
            row=1,
            col=col_idx,
        )

        activity = gt[:, 1]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=activity,
                mode="lines",  # dots + line
                name="Activity",
                legendgroup=f"{dataset}-act",
                showlegend=(col_idx == 1),
                line=dict(width=GT_LINE_WIDTH, color="red"),
            ),
            row=2,
            col=col_idx,
        )

        # Add a vertical boundary at LBW
        fig.add_vline(
            x=(lbw - 1) * 2,
            line_dash="dash",
            line_width=1,
            line_color="black",
            # row=1,
            col=col_idx,
        )

        _plot_trace(
            fig,
            loaded_endo,
            best_model[dataset],
            "Endo",
            hr,
            x,
            window,
            idx,
            col_idx,
            lbw,
            pw,
        )
        _plot_trace(
            fig,
            loaded_exo,
            best_model[dataset],
            "Exo",
            hr,
            x,
            window,
            idx,
            col_idx,
            lbw,
            pw,
        )

    fig.update_layout(
        template="seaborn",
        height=400,  # tweak as you like
        width=1200,  # tweak as you like
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.4,  # push legend below plot
            xanchor="center",
            x=0.5,
            font=dict(size=EXO_ENDO_LEGEND_SIZE),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=80, r=30, t=90, b=130),
        font=dict(size=16),  # base font (ticks inherit unless overridden)
    )

    # Axis titles & tick label sizes
    fig.update_xaxes(
        title_text="Time (s)",
        title_font=dict(size=AXIS_TITLE_SIZE),
        tickfont=dict(size=TICK_SIZE),
        row=2,
    )
    fig.update_yaxes(
        title_text="HR",
        title_font=dict(size=AXIS_TITLE_SIZE),
        tickfont=dict(size=TICK_SIZE),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="IMU",
        title_font=dict(size=AXIS_TITLE_SIZE),
        tickfont=dict(size=TICK_SIZE),
        row=2,
        col=1,
    )

    for col in range(2, len(datasets) + 1):
        fig.update_yaxes(
            tickfont=dict(size=TICK_SIZE),
            col=col,
        )

    # Make all subplot titles bigger
    for a in fig.layout.annotations:
        a.font = dict(size=SUBTITLE_SIZE)

    fig.show()
