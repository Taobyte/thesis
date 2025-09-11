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


def process_diff(
    dataset: str,
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    start_time: str,
    metric: str = "MSE",
):
    assert len(look_back_window) == 1
    lbw = look_back_window[0]

    runs_exo = get_runs(
        dataset,
        look_back_window,
        prediction_window,
        models,
        experiment_name="endo_exo",
        start_time=start_time,
    )
    exo_metrics, _, _ = get_metrics(runs_exo)

    runs_endo = get_runs(
        dataset,
        look_back_window,
        prediction_window,
        models,
        experiment_name="endo_only",
        start_time=start_time,
    )
    endo_metrics, _, _ = get_metrics(runs_endo)
    cols = defaultdict(list)
    for pw in prediction_window:
        for m in models:
            diffs: List[float] = []
            imprv: List[float] = []
            for fold_nr, seed_dict in exo_metrics[m][lbw][pw].items():
                for seed, metrics_dict in seed_dict.items():
                    if metric not in endo_metrics[m][lbw][pw][fold_nr][seed]:
                        print(
                            f"metric {metric} not in endo_metric {m} {fold_nr} {seed}"
                        )
                        continue
                    if metric not in metrics_dict:
                        print(
                            f"metric {metric} not in metric_dict for {m} {fold_nr} {seed}"
                        )
                        continue
                    endo_val = endo_metrics[m][lbw][pw][fold_nr][seed][metric]
                    exo_val = metrics_dict[metric]
                    if not isinstance(endo_val, float) or not isinstance(
                        exo_val, float
                    ):
                        print(f"NOT FLOAT for {m} fold {fold_nr} seed {seed}")
                        continue
                    diffs.append(endo_val - exo_val)
                    imprv.append(100 * (endo_val - exo_val) / (endo_val + 1e-8))

            cols[pw].append(float(np.mean(imprv)))

    df = pd.DataFrame(
        cols, index=models
    )  # keys in cols become columns; rows align to models
    df.index.name = "model"
    return df


# --- style knobs (tweak here) ---
SUBPLOT_TITLE_SIZE = 18  # larger subplot titles
LINE_WIDTH = 4.0  # thicker median lines
MARKER_SIZE = 9  # larger point markers
LEGEND_Y = -0.18  # how far below the plot the legend sits


def _add_models(fig, x, df: pd.DataFrame, col: int, showlegend: bool):
    for m, row in df.iterrows():
        color = model_colors[m]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=row.values,
                mode="lines+markers",
                line=dict(width=LINE_WIDTH, color=color),
                marker=dict(size=MARKER_SIZE),
                name=m,
                showlegend=showlegend,
            ),
            row=2,
            col=col,
        )


def _rgba(color_rgb: str, alpha: float = 0.18) -> str:
    # "rgb(r,g,b)" -> "rgba(r,g,b,alpha)"
    return color_rgb.replace("rgb", "rgba").rstrip(")") + f",{alpha})"


def _add_band(fig, x, ql, qu, color, name, row, col, showlegend):
    # IQR band (fill between q25 and q75)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=ql,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=qu,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor=_rgba(color, 0.18),
            name=f"{name} IQR",
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def _add_median(fig, x, median, color, name, row, col, showlegend):
    fig.add_trace(
        go.Scatter(
            x=x,
            y=median,
            mode="lines+markers",
            line=dict(width=LINE_WIDTH, color=color),
            marker=dict(size=MARKER_SIZE),
            name=f"{name} median",
            showlegend=showlegend,
        ),
        row=row,
        col=col,
    )


def horizon_exo_difference(
    datasets: list[str],
    models: list[str] = MODELS,
    look_back_window: list[int] = [30],
    prediction_window: list[int] = [1, 3, 5, 10, 20],
    metric: str = "MSE",
    start_time: str = "2025-08-28",
    use_std: bool = False,
):
    models.remove("msar")  # we remove msar due to bad performance
    N_BASELINES = len(models) // 2

    BASELINE_COLOR = "rgb(31,119,180)"  # blue
    DL_COLOR = "rgb(214,39,40)"  # red

    # make subplot titles bold
    dataset_names = [f"<b>{dataset_to_name[d]}</b>" for d in datasets]

    fig = make_subplots(
        rows=2,
        cols=len(datasets),
        shared_yaxes=False,
        subplot_titles=dataset_names,
        horizontal_spacing=0.08,
    )

    # categorical, equally spaced x-ticks
    x_labels = [str(h) for h in prediction_window]

    for j, dataset in enumerate(datasets, start=1):
        df = process_diff(
            dataset=dataset,
            models=models,
            look_back_window=look_back_window,
            prediction_window=prediction_window,
            start_time=start_time,
            metric=metric,
        )

        # ensure column order matches desired horizons
        df = df.reindex(columns=prediction_window)

        baseline_df = df.iloc[:N_BASELINES, :]
        dl_df = df.iloc[N_BASELINES:, :]

        # robust summaries across models (per horizon)
        baseline_median = (
            baseline_df.median(axis=0, skipna=True).reindex(prediction_window).values
        )
        baseline_q25 = (
            baseline_df.quantile(0.25, axis=0, interpolation="linear")
            .reindex(prediction_window)
            .values
        )
        baseline_q75 = (
            baseline_df.quantile(0.75, axis=0, interpolation="linear")
            .reindex(prediction_window)
            .values
        )

        dl_median = dl_df.median(axis=0, skipna=True).reindex(prediction_window).values
        dl_q25 = (
            dl_df.quantile(0.25, axis=0, interpolation="linear")
            .reindex(prediction_window)
            .values
        )
        dl_q75 = (
            dl_df.quantile(0.75, axis=0, interpolation="linear")
            .reindex(prediction_window)
            .values
        )

        # Only show legend once
        showlegend = j == 1

        # Baselines band + median
        _add_band(
            fig,
            x_labels,
            baseline_q25,
            baseline_q75,
            BASELINE_COLOR,
            "Baselines",
            1,
            j,
            showlegend,
        )
        _add_median(
            fig,
            x_labels,
            baseline_median,
            BASELINE_COLOR,
            "Baselines",
            1,
            j,
            showlegend,
        )

        # DL band + median
        _add_band(fig, x_labels, dl_q25, dl_q75, DL_COLOR, "DL", 1, j, showlegend)
        _add_median(fig, x_labels, dl_median, DL_COLOR, "DL", 1, j, showlegend)

        # add all models two second row
        _add_models(fig, x_labels, df, j, showlegend)

        # equally spaced categorical x
        fig.update_xaxes(
            type="category",
            categoryorder="array",
            categoryarray=x_labels,
            title_text="Horizon (steps)",
            row=1,
            col=j,
        )

        fig.update_yaxes(
            title_text="Relative Improvement",
            row=1,
            col=j,
        )

    # --- layout: bold & larger titles, legend bottom center ---
    fig.update_annotations(
        font=dict(size=SUBPLOT_TITLE_SIZE)
    )  # boost subplot title size
    fig.update_layout(
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=LEGEND_Y,
            yanchor="top",
        ),
        margin=dict(b=120),
    )

    fig.update_xaxes(title_font=dict(size=14))
    fig.update_yaxes(title_font=dict(size=14))

    fig.show()


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

    # models.remove("msar")

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
        imprv_mean: List[float] = []
        imprv_std: List[float] = []
        labels: List[str] = []
        colors: List[str] = []

        for m in models:
            abbr_name = model_to_abbr[m]
            diffs: List[float] = []
            imprv: List[float] = []
            for fold_nr, seed_dict in exo_metrics[m][lbw][pw].items():
                for seed, metrics_dict in seed_dict.items():
                    if metric not in endo_metrics[m][lbw][pw][fold_nr][seed]:
                        print(
                            f"metric {metric} not in endo_metric {m} {fold_nr} {seed}"
                        )
                        continue
                    if metric not in metrics_dict:
                        print(
                            f"metric {metric} not in metric_dict for {m} {fold_nr} {seed}"
                        )
                        continue
                    endo_val = endo_metrics[m][lbw][pw][fold_nr][seed][metric]
                    exo_val = metrics_dict[metric]
                    if not isinstance(endo_val, float) or not isinstance(
                        exo_val, float
                    ):
                        print(f"NOT FLOAT for {m} fold {fold_nr} seed {seed}")
                        continue
                    diffs.append(endo_val - exo_val)
                    imprv.append(100 * (endo_val - exo_val) / (endo_val + 1e-8))

            if not diffs:
                continue

            diff_mean.append(float(np.mean(diffs)))
            diff_std.append(float(np.std(diffs, ddof=1)))
            imprv_mean.append(float(np.mean(imprv)))
            imprv_std.append(float(np.std(imprv)))
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

        fig = go.Figure()

        for x, y, err, c in zip(labels, imprv_mean, imprv_std, colors):
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
