import pandas as pd
import numpy as np
import wandb
import plotly.graph_objects as go
import plotly.io as pio

from matplotlib import colors
from plotly.subplots import make_subplots
from typing import List

from src.wandb_results.utils import get_metrics, get_runs
from src.constants import (
    model_to_name,
    metric_to_name,
    dataset_to_name,
    MODELS,
    METRICS,
)


def plot_normalization_table(
    datasets: list[str],
    prediction_window: list[int] = [3],
    start_time: str = "2025-8-21",
    models: list[str] = [],
):
    assert len(prediction_window) == 1
    pw = prediction_window[0]
    metric = "ND"
    if len(models) == 0:
        models = MODELS

    cols: dict[str, list[float]] = dict()

    model_col: List[str] = []
    for model_name in models:
        model_name = model_to_name[model_name]
        model_col.append(model_name)

    cols["Models"] = model_col

    for dataset in datasets:
        print(f"DATASET: {dataset}")
        for experiment, name in zip(
            [
                "no_norm",
                "global_z_norm",
                "min_max_norm",
                "local_z_norm",
                "difference",
            ],
            ["NN", "GZ", "MM", "LZ", "DF"],
        ):
            print(f"EXPERIMENT: {experiment}")
            col: List[float] = []
            for model_name in models:
                conditions = [
                    {"state": "finished"},
                    {"created_at": {"$gte": start_time}},
                    {"tags": dataset},  # dataset tag must be present
                    {"tags": experiment},  # experiment tag must be present
                    {"tags": model_name},
                ]
                filters = {"$and": conditions}

                api = wandb.Api()
                runs = api.runs("c_keusch/thesis", filters=filters)
                print(f"{model_name} length of runs {len(runs)}")

                if len(runs) % 3 == 0 and len(runs) > 0:
                    mean_dict, _ = get_metrics(runs)
                    lbw = list(mean_dict[model_name].keys())[0]
                    if metric in mean_dict[model_name][lbw][str(pw)]:
                        mean = mean_dict[model_name][lbw][str(pw)][metric]
                    else:
                        mean = np.nan

                else:
                    print(f"{model_name} {dataset} {experiment} no 3 runs found")
                    mean = np.nan
                col.append(mean)
            cols[f"{dataset} {name}"] = col

    df = pd.DataFrame(cols)

    latex_str = df.to_latex(
        index=False,
        escape=False,
        header=True,
        # column_format=column_format,
        bold_rows=False,
        float_format=lambda x: "-" if pd.isna(x) else f"{100 * x:.2f}",
    )
    print(latex_str)


def visualize_normalization_difference(
    datasets: list[str],
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    use_dynamic_features: bool,
    use_static_features: bool,
    start_time: str = "2025-6-05",
    save_html: bool = False,
):
    for dataset in datasets:
        for pw in prediction_window:
            fig = make_subplots(
                rows=len(models),
                cols=4,
                column_titles=[metric_to_name[metric] for metric in METRICS],
                row_titles=[model_to_name[model] for model in models],
                shared_xaxes=False,
            )

            for n, normalization in enumerate(["global", "local", "none"]):
                runs = get_runs(
                    dataset,
                    models,
                    look_back_window,
                    prediction_window,
                    use_heart_rate,
                    use_dynamic_features,
                    use_static_features,
                    normalization,
                    start_time,
                )

                mean_dict, std_dict = get_metrics(runs)
                for m, model in enumerate(models):
                    for j, metric in enumerate(METRICS):
                        look_back_windows = sorted(list(mean_dict[model].keys()))
                        x = [int(lbw) for lbw in look_back_windows]

                        means = [
                            mean_dict[model][lbw][str(pw)][metric]
                            for lbw in look_back_windows
                        ]
                        stds = [
                            std_dict[model][lbw][str(pw)][metric]
                            for lbw in look_back_windows
                        ]
                        upper = [m + s for m, s in zip(means, stds)]
                        lower = [m - s for m, s in zip(means, stds)]

                        color = model_colors[n]

                        row = m + 1
                        col = j + 1

                        # Mean line
                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=means,
                                mode="lines+markers",
                                name=normalization,
                                line=dict(color=color),
                                showlegend=(j == 0) and (row == 1),
                                legendgroup=normalization,
                                # legendgrouptitle_text=dataset_name,
                            ),
                            row=row,
                            col=col,
                        )

                        # Std deviation band (fill between)
                        fig.add_trace(
                            go.Scatter(
                                x=x + x[::-1],
                                y=upper + lower[::-1],
                                fill="toself",
                                fillcolor=color.replace("1.0", "0.2")
                                if "rgba" in color
                                else f"rgba({','.join(str(int(c * 255)) for c in colors.to_rgb(color))},0.2)",
                                line=dict(color="rgba(255,255,255,0)"),
                                hoverinfo="skip",
                                showlegend=False,
                                name=normalization,
                                legendgroup=normalization,
                            ),
                            row=row,
                            col=col,
                        )
                        fig.update_xaxes(
                            title_text="Lookback Window",
                            tickmode="array",
                            tickvals=look_back_windows,
                            row=row,
                            col=col,
                        )

            fig.update_layout(
                title={
                    "text": f"<b>Normalization Ablation {dataset_to_name[dataset]} | PW = {pw}</b>",
                    "x": 0.5,
                    "xanchor": "center",
                    "font": dict(size=50, family="Arial", color="black"),
                },
                template="plotly_white",
                height=2000,
                width=2000,
            )

            if save_html:
                plot_name = f"{dataset_to_name[dataset]}_{pw}"
                pio.write_html(
                    fig,
                    file=f"./plots/ablations/normalization/{plot_name}.html",
                    auto_open=True,
                )
            else:
                fig.show()
