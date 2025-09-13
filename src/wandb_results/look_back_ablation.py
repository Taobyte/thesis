import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots
from tqdm import tqdm
from typing import Tuple, Union
from collections import defaultdict

from src.wandb_results.utils import get_metrics, get_runs
from src.constants import (
    MODELS,
    BASELINES,
    DL,
    dataset_to_name,
    model_to_name,
    model_colors,
    dataset_colors,
)


TITLE_SIZE = 22 * 2
AXIS_TITLE_SIZE = 40
TICK_SIZE = 32
LEGEND_SIZE = 40
LINE_WIDTH = 8
MARKER_SIZE = 16
LINE_OPACITY = 0.9


# Configure which direction is "better" for each metric
METRIC_GOAL = {
    "MASE": "min",
    "MAE": "min",
    "MSE": "min",
    "SMAPE": "min",
    "DA": "max",
}


def _metric_goal(metric_name: str) -> str:
    # fallback: minimize if unknown
    return METRIC_GOAL.get(metric_name, "min")


def _best_from_nested(metric_dict_model: dict, metric_key: str, goal: str):
    """
    Given metric_dict_model = metrics[model] = {lbw: {pw: {metric_key: value, ...}}}
    return (best_value, best_lbw, best_pw)
    """
    best_val = None
    best_lbw = None
    best_pw = None

    for lbw, pw_dict in metric_dict_model.items():
        for pw, mvals in pw_dict.items():
            if metric_key not in mvals:
                continue
            val = mvals[metric_key]
            if val is None or (
                isinstance(val, float) and (np.isnan(val) or np.isinf(val))
            ):
                continue
            if best_val is None:
                best_val, best_lbw, best_pw = val, lbw, pw
            else:
                if goal == "min" and val < best_val:
                    best_val, best_lbw, best_pw = val, lbw, pw
                if goal == "max" and val > best_val:
                    best_val, best_lbw, best_pw = val, lbw, pw

    return best_val, best_lbw, best_pw


def _std_at(std_dict_model: dict, lbw, pw, metric_key: str):
    try:
        return std_dict_model[lbw][pw].get(metric_key, None)
    except Exception:
        return None


def summarize_one_dataset(
    dataset_label: str,
    models: list,
    metric_key: str,
    exo_mean_dict: dict,
    exo_std_dict: dict,
    endo_mean_dict: dict,
    endo_std_dict: dict,
    show_pw=True,
    diff_mode="best_vs_best",
):
    """
    Build a wide table (one row) for a single dataset:
      columns = model names
      cell = "{score ± std} | lbw=..[, pw=..] | Δ=exo-endo"
    diff_mode:
      - "best_vs_best": Δ = exo_best - endo_best (each at its own best lbw/pw)
      - "same_hypers": Δ computed at the exo best lbw/pw (falls back if missing)
    """
    goal = _metric_goal(metric_key)
    cols: dict[str, list[Union[float, str]]] = defaultdict(list)
    cols["Dataset"] = [
        rf"\multirow{{6}}{{*}}{{\rotatebox[origin=c]{{90}}{{{dataset_to_name[dataset_label]}}}}}"
    ] + [""] * 5
    cols["Metrics"] = ["MSE", "MAE", "DIR", "MASE", "IMPR", "LBW"]
    for model in models:
        exo_best, exo_lbw, exo_pw = _best_from_nested(
            exo_mean_dict[model], metric_key, goal
        )
        endo_best, endo_lbw, endo_pw = _best_from_nested(
            endo_mean_dict[model], metric_key, goal
        )

        # Choose how to compute Δ
        if diff_mode == "same_hypers":
            if exo_lbw is None:
                import pdb

                pdb.set_trace()
            endo_same = endo_mean_dict[model][exo_lbw][exo_pw][metric_key]
            delta = endo_same - exo_best
            denom = endo_same
        else:
            delta = endo_best - exo_best
            denom = endo_best

        # best = exo_best if delta < 0 else endo_best
        best_lbw = exo_lbw if delta < 0 else endo_lbw
        mase = (
            exo_mean_dict[model][exo_lbw][exo_pw]["MASE"]
            if delta >= 0
            else endo_mean_dict[model][endo_lbw][endo_pw]["MASE"]
        )

        mse = (
            exo_mean_dict[model][exo_lbw][exo_pw]["MSE"]
            if delta >= 0
            else endo_mean_dict[model][endo_lbw][endo_pw]["MSE"]
        )

        mae = (
            exo_mean_dict[model][exo_lbw][exo_pw]["MAE"]
            if delta >= 0
            else endo_mean_dict[model][endo_lbw][endo_pw]["MAE"]
        )

        dir = (
            exo_mean_dict[model][exo_lbw][exo_pw]["DIRACC"]
            if delta >= 0
            else endo_mean_dict[model][endo_lbw][endo_pw]["DIRACC"]
        )

        imprv = 100 * (delta / denom)

        cols[f"{model_to_name[model]}"] = [mse, mae, dir, mase, imprv, best_lbw]

    df = pd.DataFrame.from_dict(cols)
    return df


def summarize_all_datasets(
    dataset_labels: list,
    models: list,
    metric_key: str,
    per_dataset_metrics: list,  # list of tuples: (exo_mean, exo_std, endo_mean, endo_std)
    diff_mode="best_vs_best",
) -> pd.DataFrame:
    """
    dataset_labels: ["DaLiA", "WildPPG", "Capture24"]  (same order as your loop)
    per_dataset_metrics: [(exo_mean_dict, exo_std_dict, endo_mean_dict, endo_std_dict), ...]
    """
    tables = []
    for label, (exo_m, exo_s, endo_m, endo_s) in zip(
        dataset_labels, per_dataset_metrics
    ):
        tables.append(
            summarize_one_dataset(
                label,
                models=models,
                metric_key=metric_key,
                exo_mean_dict=exo_m,
                exo_std_dict=exo_s,
                endo_mean_dict=endo_m,
                endo_std_dict=endo_s,
                diff_mode=diff_mode,
            )
        )
    big = pd.concat(tables, axis=0)

    return big


def find_best_model(metrics: dict, lbw: int, pw: int, metric: str) -> Tuple[str, float]:
    best_model = ""
    best_value = -np.inf if metric == "DIRACC" else np.inf

    for model_name, values in metrics.items():
        if metric not in values[str(lbw)][str(pw)]:
            print(f"{metric} is not in dict for {model_name} lbw {lbw} pw {pw}")
            model_value = -np.inf if metric == "DIRACC" else np.inf
        else:
            model_value = values[str(lbw)][str(pw)][metric]

        if metric == "DIRACC":
            if model_value > best_value:
                best_model = model_name
                best_value = model_value
        else:
            if model_value < best_value:
                best_model = model_name
                best_value = model_value

    return best_model, best_value


def ablation_delta_plot(
    datasets: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    experiment: str = "endo_only",
    start_time: str = "2025-6-05",
    save_html: bool = False,
    models: list[str] = [],
):
    metric = "NRMSE"  # METRICS

    num_datasets = len(datasets)
    n_metrics = 1
    if len(models) == 0:
        models = MODELS

    lbw_ablation = True
    if len(look_back_window) == 1:
        assert len(prediction_window) > 1
        lbw = look_back_window[0]
        x = sorted([int(pw) for pw in prediction_window])
        lbw_ablation = False
    elif len(prediction_window) == 1:
        assert len(look_back_window) > 1
        pw = prediction_window[0]
        x = sorted([int(lbw) for lbw in look_back_window])
        lbw_ablation = True
    else:
        raise ValueError(
            "Invalid input: either look_back_window or prediction_window must have length > 1, but not both."
        )

    x_vals = x
    x_pos = list(range(len(x_vals)))
    x_text = [str(v) for v in x_vals]

    fig = make_subplots(
        rows=n_metrics,
        cols=1,
        row_titles=[metric],
        shared_xaxes=True,
        shared_yaxes=True if metric in ["ND", "NRMSE"] else False,
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )
    dataset_perf: dict[str, float] = {}
    for b, dataset in tqdm(enumerate(datasets), total=num_datasets):
        runs = get_runs(
            dataset,
            look_back_window,
            prediction_window,
            models,
            start_time,
            experiment_name=experiment,
        )

        mean_dict, std_dict = get_metrics(runs)

        baseline_means = {
            model_name: values
            for model_name, values in mean_dict.items()
            if model_name in BASELINES
        }

        dl_means = {
            model_name: values
            for model_name, values in mean_dict.items()
            if model_name in DL
        }

        factor = 100 if metric in ["SMAPE", "ND", "NRMSE"] else 1
        diffs: list[float] = []
        for ablation_x in x:
            if lbw_ablation:
                best_baseline, bb_value = find_best_model(
                    baseline_means, ablation_x, pw, metric
                )
                best_dl, bdl_value = find_best_model(dl_means, ablation_x, pw, metric)
            else:
                best_baseline, bb_value = find_best_model(
                    baseline_means, lbw, ablation_x, metric
                )
                best_dl, bdl_value = find_best_model(dl_means, lbw, ablation_x, metric)

            diff = (bdl_value - bb_value) * factor
            diffs.append(diff)
        dataset_perf[dataset] = diffs

    for dataset_name, diffs in dataset_perf.items():
        fig.add_trace(
            go.Scatter(
                x=x_pos,
                y=diffs,
                mode="lines+markers",
                line=dict(color=dataset_colors[dataset_name], width=LINE_WIDTH),
                marker=dict(size=MARKER_SIZE),
                opacity=LINE_OPACITY,
                showlegend=True,
                name=dataset_to_name[dataset_name],
            ),
            row=1,
            col=1,
        )

    fig.update_layout(template="seaborn")

    fig.update_xaxes(
        title_text="Lookback Window" if lbw_ablation else "Prediction Window",
        tickmode="array",
        tickvals=x_pos,
        ticktext=x_text,
        title_font=dict(size=AXIS_TITLE_SIZE),
        tickfont=dict(size=TICK_SIZE),
        row=1,
        col=1,
    )

    # FONT SIZE & STYLE
    fig.update_layout(
        font=dict(size=TICK_SIZE),
        legend=dict(font=dict(size=LEGEND_SIZE)),
        margin=dict(t=90, b=120),
    )

    fig.update_xaxes(
        title_font=dict(size=AXIS_TITLE_SIZE), tickfont=dict(size=TICK_SIZE)
    )
    fig.update_yaxes(
        title_font=dict(size=AXIS_TITLE_SIZE), tickfont=dict(size=TICK_SIZE)
    )

    # LEGEND POSITION
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            # title_text="Datasets",
            itemsizing="trace",
        ),
        margin=dict(b=30),
    )

    if save_html:
        plot_name = f"{dataset}_{use_heart_rate}_{use_dynamic_features}_{'_'.join(models)}_{'_'.join([str(lbw) for lbw in look_back_window])}_{'_'.join([str(pw) for pw in prediction_window])}"
        pio.write_html(
            fig, file=f"./plots/ablations/look_back/{plot_name}.html", auto_open=True
        )
        fig.write_image(
            f"./plots/ablations/look_back/{plot_name}.pdf",
            width=1920,  # width in pixels
            height=1080,
            scale=2,
        )
        print(f"Successfully saved {plot_name}")
    else:
        fig.show()


def _series_for(
    mean_dict, std_dict, model, lbw_ablation, x_vals, lbw, pw, metric, factor
):
    means, stds = [], []
    for xv in x_vals:
        key_lbw = str(xv) if lbw_ablation else str(lbw)
        key_pw = str(pw) if lbw_ablation else str(xv)
        try:
            m = mean_dict[model][int(key_lbw)][int(key_pw)][metric] * factor
            s = std_dict[model][int(key_lbw)][int(key_pw)][metric] * factor
        except KeyError:
            m, s = np.nan, 0.0
        means.append(m)
        stds.append(s)
    return means, stds


def _alpha_fill(color_str, alpha=0.18):
    # works for 'rgba(r,g,b,a)' or '#RRGGBB' or 'rgb(r,g,b)'
    c = color_str
    if isinstance(c, tuple):  # (r,g,b) 0-255
        r, g, b = c
        return f"rgba({r},{g},{b},{alpha})"
    if c.startswith("rgba"):
        head, body = c.split("(", 1)
        rgb = body.split(")")[0].split(",")[:3]
        r, g, b = [x.strip() for x in rgb]
        return f"rgba({r},{g},{b},{alpha})"
    if c.startswith("rgb("):
        r, g, b = c[4:-1].split(",")
        return f"rgba({r.strip()},{g.strip()},{b.strip()},{alpha})"
    if c.startswith("#") and len(c) == 7:
        r = int(c[1:3], 16)
        g = int(c[3:5], 16)
        b = int(c[5:7], 16)
        return f"rgba({r},{g},{b},{alpha})"
    return "rgba(0,0,0,0.18)"


def visualize_look_back_window_difference(
    datasets: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    start_time: str = "2025-6-05",
    save_html: bool = False,
    use_std: bool = False,
    models: list[str] = MODELS,
):
    num_datasets = len(datasets)

    lbw_ablation = True
    if len(look_back_window) == 1:
        assert len(prediction_window) > 1
        lbw = look_back_window[0]
        x = sorted([int(pw) for pw in prediction_window])
        lbw_ablation = False
    elif len(prediction_window) == 1:
        assert len(look_back_window) > 1
        pw = prediction_window[0]
        x = sorted([int(lbw) for lbw in look_back_window])
        lbw_ablation = True
    else:
        raise ValueError(
            "Invalid input: either look_back_window or prediction_window must have length > 1, but not both."
        )

    x_vals = x
    x_pos = list(range(len(x_vals)))
    x_text = [str(v) for v in x_vals]

    row_names: list[str] = []
    for dataset in datasets:
        row_names.append(f"BL {dataset_to_name[dataset]}")
        row_names.append(f"DL {dataset_to_name[dataset]}")
    readable_model_names = [model_to_name[m] for m in models]
    metric = "MASE"

    all_metrics = []

    fig = make_subplots(
        rows=2 * num_datasets,
        cols=len(models) // 2,
        subplot_titles=readable_model_names,
        row_titles=row_names,
        shared_xaxes=True,
        shared_yaxes=False,
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )
    for b, dataset in tqdm(enumerate(datasets), total=num_datasets):
        exo_runs = get_runs(
            dataset,
            look_back_window,
            prediction_window,
            models,
            start_time,
            experiment_name="endo_exo",
        )
        endo_runs = get_runs(
            dataset,
            look_back_window,
            prediction_window,
            models,
            start_time,
            experiment_name="endo_only",
        )

        _, exo_mean_dict, exo_std_dict = get_metrics(exo_runs)
        _, endo_mean_dict, endo_std_dict = get_metrics(endo_runs)

        all_metrics.append((exo_mean_dict, exo_std_dict, endo_mean_dict, endo_std_dict))

        factor = 100 if metric == "SMAPE" else 1

        for m, model in enumerate(models):
            row = 2 * b + m // 7 + 1
            col = m % 7 + 1
            base_color = model_colors[model]

            # --- build series for both settings ---
            exo_means, exo_stds = _series_for(
                exo_mean_dict,
                exo_std_dict,
                model,
                lbw_ablation,
                x,
                lbw if not lbw_ablation else None,
                pw if lbw_ablation else None,
                metric,
                factor,
            )
            endo_means, endo_stds = _series_for(
                endo_mean_dict,
                endo_std_dict,
                model,
                lbw_ablation,
                x,
                lbw if not lbw_ablation else None,
                pw if lbw_ablation else None,
                metric,
                factor,
            )

            # bands
            exo_upper = [a + b for a, b in zip(exo_means, exo_stds)]
            exo_lower = [a - b for a, b in zip(exo_means, exo_stds)]
            endo_upper = [a + b for a, b in zip(endo_means, endo_stds)]
            endo_lower = [a - b for a, b in zip(endo_means, endo_stds)]

            # Legend: show only once to avoid 28 entries
            show_legend = row == 1 and col == 1

            # --- endo_exo line (solid) ---
            fig.add_trace(
                go.Scatter(
                    x=x_pos,
                    y=exo_means,
                    mode="lines+markers",
                    name="Endogenous & Exogenous",
                    line=dict(color=base_color, width=LINE_WIDTH),
                    marker=dict(size=MARKER_SIZE),
                    opacity=LINE_OPACITY,
                    showlegend=show_legend,
                    legendgroup="endo_exo",
                ),
                row=row,
                col=col,
            )
            if use_std:
                fig.add_trace(
                    go.Scatter(
                        x=x_pos + x_pos[::-1],
                        y=exo_upper + exo_lower[::-1],
                        fill="toself",
                        fillcolor=_alpha_fill(base_color, 0.18),
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup="endo_exo",
                    ),
                    row=row,
                    col=col,
                )

            # --- endo_only line (dashed) ---
            fig.add_trace(
                go.Scatter(
                    x=x_pos,
                    y=endo_means,
                    mode="lines+markers",
                    name="Endogenous Only",
                    line=dict(color=base_color, width=LINE_WIDTH, dash="dash"),
                    marker=dict(size=MARKER_SIZE),
                    opacity=LINE_OPACITY,
                    showlegend=show_legend,
                    legendgroup="endo_only",
                ),
                row=row,
                col=col,
            )
            if use_std:
                fig.add_trace(
                    go.Scatter(
                        x=x_pos + x_pos[::-1],
                        y=endo_upper + endo_lower[::-1],
                        fill="toself",
                        fillcolor=_alpha_fill(base_color, 0.10),  # slightly lighter
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                        legendgroup="endo_only",
                    ),
                    row=row,
                    col=col,
                )

    # figure dimensions
    subplot_size = 500  # pixels per subplot
    cols = len(models) // 2
    rows = 2 * num_datasets

    total_width = cols * subplot_size
    total_height = rows * subplot_size

    fig.update_layout(
        width=total_width,
        height=total_height,
    )

    # x and y-axis ticks
    fig.update_xaxes(
        tickvals=x_pos,
        ticktext=x_text,
        showticklabels=True,
        tickfont=dict(size=16),  # label text size
        ticks="outside",  # draw tick marks outside
        ticklen=10,  # tick mark length (px)
        tickwidth=3,  # tick mark thickness (px)
        row="all",
        col="all",
    )

    fig.update_yaxes(
        showticklabels=True,
        tickfont=dict(size=28),  # label text size
        ticks="outside",
        ticklen=10,
        tickwidth=3,
        nticks=4,  # fewer ticks (or use tickmode="linear", dtick=... )
        row="all",
        col="all",
    )

    fig.update_xaxes(row=1, col=1, scaleanchor="y", scaleratio=1)

    # larger + bold annotations
    for i, ann in enumerate(fig.layout.annotations):
        if ann.text in readable_model_names:
            fig.layout.annotations[i].update(
                text=f"<b>{ann.text}</b>",
                font=dict(size=28, family="Arial"),
            )
        if ann.text in row_names:
            fig.layout.annotations[i].update(
                text=f"<b>{ann.text}</b>",
                font=dict(size=32, family="Arial"),
            )

    # legend position + size
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.12,
            xanchor="center",
            x=0.5,
            font=dict(size=28),
            itemsizing="constant",
        )
    )

    if save_html:
        plot_name = f"{dataset}_{use_heart_rate}_{use_dynamic_features}_{'_'.join(models)}_{'_'.join([str(lbw) for lbw in look_back_window])}_{'_'.join([str(pw) for pw in prediction_window])}"
        pio.write_html(
            fig, file=f"./plots/ablations/look_back/{plot_name}.html", auto_open=True
        )
        fig.write_image(
            f"./plots/ablations/look_back/{plot_name}.pdf",
            width=1920,  # width in pixels
            height=1080,
            scale=2,
        )
        print(f"Successfully saved {plot_name}")
    else:
        fig.show()

    table_metric = "MASE"  # or "MAE", "DA", "SMAPE", "MSE"

    def print_latex_table(models: list[str]):
        table = summarize_all_datasets(
            dataset_labels=datasets,
            models=models,
            metric_key=table_metric,
            per_dataset_metrics=all_metrics,
            diff_mode="same_hypers",
        )

        n_columns = len(table.columns)

        latex_str = table.to_latex(
            index=False,
            escape=False,
            header=True,
            column_format="|".join(["c"] * n_columns),
            bold_rows=False,
            float_format="%.3f",
        )
        print(latex_str)

    print_latex_table(BASELINES)
    print_latex_table(DL)
