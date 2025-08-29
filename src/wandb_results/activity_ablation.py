import pandas as pd
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from typing import List, DefaultDict
from collections import defaultdict

from src.wandb_results.utils import get_metrics, get_runs
from src.constants import (
    MODELS,
    model_to_name,
    model_colors,
    model_to_abbr,
    dataset_to_name,
)


def visualize_exo_difference(
    datasets: list[str],
    models: list[str] = MODELS,
    look_back_window: list[int] = [30],
    prediction_window: list[int] = [3],
    metric: str = "MSE",
    start_time: str = "2025-8-28",
    use_std: bool = False,
) -> None:
    lbw = look_back_window[0]
    pw = prediction_window[0]

    dataset_length = len(datasets)

    cols: DefaultDict[str, list[str]] = defaultdict(list)
    dataset_col: List[str] = []
    for dataset in datasets:
        name = dataset_to_name[dataset]
        dataset_col.append(
            rf"\multirow{{4}}{{*}}{{\rotatebox[origin=c]{{90}}{{{name}}}}}"
        )
        dataset_col.extend([""] * 3)  # the 3 rows under the multirow
    metric_col = ["EO", "EE", "DI", "IM"] * dataset_length

    cols["D"] = dataset_col
    cols["M"] = metric_col

    for dataset in datasets:
        print(f"Processing {dataset}...")

        runs_exo = get_runs(
            dataset,
            look_back_window,
            prediction_window,
            models,
            experiment_name="endo_exo",
            start_time=start_time,
        )
        exo_metrics, exo_mean, exo_std = get_metrics(runs_exo)

        runs_endo = get_runs(
            dataset,
            look_back_window,
            prediction_window,
            models,
            experiment_name="endo_only",
            start_time=start_time,
        )
        endo_metrics, endo_mean, endo_std = get_metrics(runs_endo)

        diff_mean: List[float] = []
        diff_std: List[float] = []
        labels: List[str] = []
        colors: List[str] = []

        for m in models:
            abbr_name = model_to_abbr[m]
            try:
                diffs: List[float] = []
                imprv: List[float] = []
                for fold_nr, seed_dict in exo_metrics[m][lbw][pw].items():
                    for seed, metrics_dict in seed_dict.items():
                        endo_val = endo_metrics[m][lbw][pw][fold_nr][seed][metric]
                        exo_val = metrics_dict[metric]
                        diffs.append(endo_val - exo_val)
                        imprv.append((endo_val - exo_val) / (endo_val + 1e-8))

                if not diffs:
                    continue

                diff_mean.append(float(np.mean(diffs)))
                diff_std.append(float(np.std(diffs, ddof=1)))
                labels.append(model_to_name.get(m, m))
                colors.append(model_colors.get(m, "#444"))  # fallback if missing

                # add values to col for table
                model_endo_mean = endo_mean[m][lbw][pw][metric]
                model_endo_std = endo_std[m][lbw][pw][metric]
                model_exo_mean = exo_mean[m][lbw][pw][metric]
                model_exo_std = exo_std[m][lbw][pw][metric]
                mean_diff = np.mean(diffs)
                std_diff = np.std(diffs)
                mean_imprv = np.mean(imprv)
                std_imprv = np.std(imprv)
                if use_std:
                    values = [
                        f"{model_endo_mean:.2f} $\\pm$ {model_endo_std:.2f}",
                        f"{model_exo_mean:.2f} $\\pm$ {model_exo_std:.2f}",
                        f"{mean_diff:.2f} $\\pm$ {std_diff:.2f}",
                        f"{mean_imprv:.2f} $\\pm$ {std_imprv:.2f}",
                    ]
                else:
                    values = [
                        f"{model_endo_mean:.2f}",
                        f"{model_exo_mean:.2f}",
                        f"{mean_diff:.2f}",
                        f"{mean_imprv:.2f}",
                    ]

                cols[abbr_name].extend(values)

            except KeyError as e:
                print(f"[{dataset}] skipped {m} due to missing keys: {e}")

        fig = go.Figure()

        for x, y, err, c in zip(labels, diff_mean, diff_std, colors):
            fig.add_trace(
                go.Scatter(
                    x=[x],
                    y=[y],
                    mode="markers",
                    hoverinfo="skip",
                    showlegend=False,
                    marker=dict(
                        color=c,
                        size=16,
                        line=dict(color=c, width=2),
                    ),
                    error_y=dict(
                        type="data",
                        array=[err],
                        visible=True,
                        thickness=2,
                        width=8,
                        color=c,
                    ),
                    name="",
                )
            )

        # zero line (single color for the whole figure)
        fig.add_hline(y=0, line_dash="dash", line_width=2.5)

        fig.update_layout(
            template="plotly_white",
            title=None,
            font=dict(size=18),
            yaxis_title=f"Δ{metric} = {metric}(endo_only) − {metric}(endo+exo)",
        )
        fig.update_xaxes(title=None, tickangle=-30, tickfont=dict(size=18))
        fig.update_yaxes(tickfont=dict(size=18))

        fig.add_hline(y=0, line_dash="dash", line_width=2.5)

        fig.update_layout(
            template="plotly_white",
            margin=dict(l=50, r=20, t=20, b=60),
            font=dict(size=18),
            yaxis_title=f"Delta {metric} = {metric}(endo only) - {metric}(endo+exo)",
        )
        fig.update_xaxes(
            tickangle=-30,
            tickfont=dict(family="Arial Black, Arial, sans-serif", size=18),
        )
        fig.update_yaxes(tickfont=dict(size=18))

        fig.show()

    df = pd.DataFrame(cols)

    latex_str = df.to_latex(
        index=False,
        escape=False,
        header=True,
        # column_format=column_format,
        bold_rows=False,
        # float_format="%.3f",
    )
    print(latex_str)
