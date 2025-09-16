import pandas as pd
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from collections import defaultdict

from src.wandb_results.utils import get_metrics, get_runs, model_to_lbw
from src.constants import (
    MODELS,
    BASELINES,
    DL,
    model_to_abbr,
    dataset_to_name,
)


def exo_norm_ablation_table(
    datasets: list[str],
    models: list[str] = MODELS,
    look_back_window: list[int] = [5, 10, 20, 30, 60],
    prediction_window: list[int] = [1, 3, 5, 10, 20],
    metric: str = "MSE",
    start_time: str = "2025-8-28",
    baselines: bool = False,
    dls: bool = False,
) -> None:
    if baselines:
        models = BASELINES
    elif dls:
        models = DL

    cols = defaultdict(list)

    for dataset in datasets:
        cols["Dataset"].extend(
            [
                rf"\multirow{{15}}{{*}}{{\rotatebox[origin=c]{{90}}{{{dataset_to_name[dataset]}}}}}"
            ]
            + [""] * (len(prediction_window) * 3 - 1)
        )
        cols["Metric"].extend(["No Norm", "Norm", "Imprv"] * len(prediction_window))
        runs_exo = get_runs(
            dataset,
            look_back_window,
            prediction_window,
            models,
            experiment_name="endo_exo",
            start_time=start_time,
            local_norm_endo_only=False,
        )
        _, exo_mean, _ = get_metrics(runs_exo)

        runs_endo = get_runs(
            dataset,
            look_back_window,
            prediction_window,
            models,
            experiment_name="endo_exo",
            start_time=start_time,
            local_norm_endo_only=True,
        )
        _, endo_mean, _ = get_metrics(runs_endo)

        for pw in prediction_window:
            cols["PW"].extend([rf"\multirow{{3}}{{*}}{{{pw}}}"] + [""] * 2)
            for model in models:
                lbw = model_to_lbw(dataset, model)
                if (
                    metric in endo_mean[model][lbw][pw]
                    and metric in exo_mean[model][lbw][pw]
                ):
                    exo_val = exo_mean[model][lbw][pw][metric]
                    endo_val = endo_mean[model][lbw][pw][metric]
                    imprv = 100 * (endo_val - exo_val) / endo_val
                else:
                    print(f"did not find {model} {lbw} {pw} {metric}")
                    exo_val = np.nan
                    endo_val = np.nan
                    imprv = np.nan
                abbr = model_to_abbr[model]
                cols[abbr].extend([endo_val, exo_val, imprv])

    df = pd.DataFrame.from_dict(cols)
    order = ["Dataset", "PW", "Metric"] + [model_to_abbr[m] for m in models]
    df = df[order]

    latex_str = df.to_latex(
        index=False,
        escape=False,
        header=True,
        column_format="|".join(["c"] * len(df.columns)),
        bold_rows=False,
        float_format="%.3f",
    )
    print(latex_str)


def exo_norm_ablation_heatmap(
    datasets: list[str],
    models: list[str] = MODELS,
    look_back_window: list[int] = [5, 10, 20, 30, 60],
    prediction_window: list[int] = [1, 3, 5, 10, 20],
    metric: str = "MAE",
    start_time: str = "2025-08-28",
    use_imprv: bool = False,
    width_per_panel: int = 420,
):
    model_list = [m for m in models if m != "msar"]

    dfs = []
    zs = []
    for dataset in datasets:
        runs_exo = get_runs(
            dataset,
            look_back_window,
            prediction_window,
            model_list,
            experiment_name="endo_exo",
            start_time=start_time,
            local_norm_endo_only=False,
        )
        _, exo_mean, _ = get_metrics(runs_exo)

        runs_endo = get_runs(
            dataset,
            look_back_window,
            prediction_window,
            model_list,
            experiment_name="endo_exo",
            start_time=start_time,
            local_norm_endo_only=True,
        )
        _, endo_mean, _ = get_metrics(runs_endo)

        heat_cols: dict[int, list] = defaultdict(list)
        for pw in prediction_window:
            for model in model_list:
                lbw = model_to_lbw(dataset, model)
                if (metric in endo_mean[model][lbw][pw]) and (
                    metric in exo_mean[model][lbw][pw]
                ):
                    exo_val = exo_mean[model][lbw][pw][metric]
                    endo_val = endo_mean[model][lbw][pw][metric]
                    delta = exo_val - endo_val
                    imprv = 100.0 * (endo_val - exo_val) / (endo_val + 1e-12)
                else:
                    delta = np.nan
                    imprv = np.nan
                heat_cols[pw].append(imprv if use_imprv else delta)

        df = pd.DataFrame.from_dict(heat_cols)
        df.index = [model_to_abbr[m] for m in model_list]
        dfs.append((dataset, df))
        zs.append(df.to_numpy(dtype=float))

    Z_all = np.concatenate([z.flatten() for z in zs])
    Z_all = Z_all[~np.isnan(Z_all)]
    vmax = np.nanpercentile(np.abs(Z_all), 90) if Z_all.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    vmin = -vmax

    n = len(datasets)
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=[f"{dataset_to_name[d]}" for d, _ in dfs],
        horizontal_spacing=0.06,
    )

    xlabels = [str(pw) for pw in prediction_window]
    color_title = (
        f"Δ {metric} (%)  ↑ better"
        if use_imprv
        else (
            f"Δ {metric}  (↑ worse, ↓ better)"
            if metric.upper() == "MSE"
            else f"Δ {metric}  (↑ worse, ↓ better)"
        )
    )

    for j, (dataset, df) in enumerate(dfs, start=1):
        z = df.to_numpy(dtype=float)
        y = df.index.tolist()
        text = np.where(np.isnan(z), "", np.round(z, 1).astype(str))

        fig.add_trace(
            go.Heatmap(
                z=z,
                x=xlabels,
                y=y,
                text=text,
                texttemplate="%{text}",
                textfont={"size": 10},
                coloraxis="coloraxis",
                hovertemplate=(
                    "Model=%{y}<br>PW=%{x}<br>"
                    + ("Δ%=%{z:.1f}" if use_imprv else f"Δ {metric}=%{{z:.2f}}")
                    + "<extra></extra>"
                ),
            ),
            row=1,
            col=j,
        )

        fig.update_yaxes(autorange="reversed", row=1, col=j)
        fig.update_xaxes(title_text="Prediction window (steps)", row=1, col=j)
        if j == 1:
            fig.update_yaxes(title_text="Model", row=1, col=j)

    fig.update_layout(
        coloraxis=dict(
            colorscale="RdBu",
            cmin=vmin,
            cmax=vmax,
            cmid=0.0,
            colorbar=dict(title=color_title),
        ),
        width=max(900, width_per_panel * n),
        height=max(360, 26 * len(model_list) + 160),
        margin=dict(l=90, r=30, t=60, b=60),
    )

    fig.show()
