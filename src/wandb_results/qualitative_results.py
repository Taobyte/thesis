import json
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots
from tqdm import tqdm
from typing import Tuple, Union
from collections import defaultdict
from pathlib import Path

from src.wandb_results.utils import get_runs
from src.constants import (
    MODELS,
    BASELINES,
    DL,
    dataset_to_name,
    model_to_name,
    model_colors,
    dataset_colors,
)


def plot_predictions(
    datasets: list[str],
    look_back_window: list[int] = [20],
    prediction_window: list[int] = [10],
    models: list[str] = MODELS,
    experiment_name: str = "endo_exo",
    metric: str = "MAE",
    artifacts_path: str = "C:/Users/cleme/ETH/Master/Thesis/ns-forecast/artifacts",
):
    assert len(look_back_window) == 1
    assert len(prediction_window) == 1
    lbw = look_back_window[0]
    pw = prediction_window[0]

    window = lbw + pw

    fig = make_subplots(
        rows=1,
        cols=len(datasets),
        shared_xaxes=True,
        subplot_titles=[dataset_to_name[d] for d in datasets],
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    for col_idx, dataset in enumerate(datasets, start=1):
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
            raw_artifact = next(
                (a for a in run.logged_artifacts() if "predictions" in a.name), None
            )
            if raw_artifact is None:
                print(f"No predictions for run {run.name}")
                continue
            else:
                art_dir = Path(artifacts_path) / str(raw_artifact.name).replace(
                    ":", "-"
                )

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

        import pdb

        x = np.arange(window) * 2
        any_preds = loaded_preds[models[0]]["arrays"]
        gt_series = any_preds["gt_series"]
        gt = gt_series[0][:window, 0]
        pdb.set_trace()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=gt,
                mode="lines",
                name="Ground truth",
                legendgroup=f"{dataset}-gt",
                showlegend=True,
                line=dict(width=2),
            ),
            row=1,
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

        for model in models:
            # load model artifact & plot prediction
            arrays = loaded_preds[model]["arrays"]
            metrics = loaded_preds[model]["metrics"]
            gt_series = arrays["gt_series"]
            preds = arrays["preds"]
            pred = preds[0][:, 0]

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
                    showlegend=True
                    if col_idx == 1
                    else False,  # one legend entry across subplots
                ),
                row=1,
                col=col_idx,
            )

    fig.update_xaxes(title_text="Time (s)", row=1, col=col_idx)
    fig.update_yaxes(
        title_text="HR (BPM)" if col_idx == 1 else None, row=1, col=col_idx
    )

    fig.show()
