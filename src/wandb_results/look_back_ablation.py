import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots
from matplotlib import colors
from tqdm import tqdm
from typing import List

from src.wandb_results.utils import get_metrics, get_runs
from src.constants import (
    METRICS,
    MODELS,
    BASELINES,
    DL,
    dataset_to_name,
    metric_to_name,
    model_to_name,
    model_colors,
)


def visualize_look_back_window_difference(
    datasets: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    experiment: str = "endo_only",
    start_time: str = "2025-6-05",
    save_html: bool = False,
    use_std: bool = False,
    models: list[str] = [],
):
    num_datasets = len(datasets)
    n_metrics = len(METRICS)

    if len(models) == 0:
        models = MODELS

    readable_metric_names = [metric_to_name[m] for m in METRICS]
    readable_dataset_names = [dataset_to_name[d] for d in datasets]
    subplot_titles = readable_metric_names + [""] * ((num_datasets - 1) * n_metrics)
    row_titles: list[str] = []
    for name in readable_dataset_names:
        row_titles.append(f"{name} Individual")
        row_titles.append(f"{name} Aggregated")

    fig = make_subplots(
        rows=num_datasets * 2,  # we also plot the aggregated results per baseline or dl
        cols=n_metrics,
        subplot_titles=subplot_titles,
        row_titles=row_titles,
        shared_xaxes=True,
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    for b, dataset in tqdm(enumerate(datasets), total=num_datasets):
        runs = get_runs(
            dataset,
            models,
            look_back_window,
            prediction_window,
            True,
            "all",
            start_time,
            experiment_name=experiment,
        )

        mean_dict, std_dict = get_metrics(runs)

        for i, pw in enumerate(prediction_window):
            for j, metric in enumerate(METRICS):
                assert set(mean_dict.keys()) == set(models), (
                    f"Models froms runs: {mean_dict.keys()} | Models from cmd {models}"
                )
                baseline_means: List[float] = []
                dl_means: List[float] = []
                row = 2 * b + 1
                col = j + 1
                for m, model in enumerate(models):
                    model_name = model_to_name[model]

                    look_back_windows = sorted(list(mean_dict[model].keys()), key=int)
                    x = [int(lbw) for lbw in look_back_windows]

                    means = [
                        mean_dict[model][lbw][str(pw)][metric]
                        for lbw in look_back_windows
                    ]
                    if model in BASELINES:
                        baseline_means.append(means)
                    elif model in DL:
                        dl_means.append(means)

                    stds = [
                        std_dict[model][lbw][str(pw)][metric]
                        for lbw in look_back_windows
                    ]
                    upper = [m + s for m, s in zip(means, stds)]
                    lower = [m - s for m, s in zip(means, stds)]

                    color = model_colors[m]

                    # Mean line
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=means,
                            mode="lines+markers",
                            name=model_name,
                            line=dict(color=color),
                            showlegend=(j == 0) and row == 1,
                            legendgroup=model_name,
                            legendgrouptitle_text="Individual Perf.",
                        ),
                        row=row,
                        col=col,
                    )

                    # Std deviation band (fill between)
                    if use_std:
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
                                name=model_name,
                                legendgroup=model_name,
                            ),
                            row=row,
                            col=col,
                        )

                baseline_means_mean = np.mean(
                    np.stack(baseline_means), axis=0
                ).tolist()  # (len(look_back_window),)
                dl_means_mean = np.mean(np.stack(dl_means), axis=0).tolist()
                baseline_means_std = np.std(np.stack(baseline_means), axis=0).tolist()
                # (len(look_back_window),)
                dl_means_std = np.std(np.stack(dl_means), axis=0).tolist()

                for name, color, y, stds in zip(
                    ["Baselines", "DL"],
                    [model_colors[-1], model_colors[-2]],
                    [baseline_means_mean, dl_means_mean],
                    [baseline_means_std, dl_means_std],
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=y,
                            mode="lines+markers",
                            name=name,
                            showlegend=(j == 0) and row == 1,
                            line=dict(color=color),
                            legendgroup=name,
                            legendgrouptitle_text="Baselines vs DL",
                        ),
                        row=row + 1,
                        col=col,
                    )

                    upper = [m + s for m, s in zip(y, stds)]
                    lower = [m - s for m, s in zip(y, stds)]

                    fig.add_trace(
                        go.Scatter(
                            x=x + x[::-1],
                            y=upper + lower[::-1],
                            fill="toself",
                            fillcolor=f"rgba({','.join(str(int(c * 255)) for c in colors.to_rgb(color))},0.2)",
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            showlegend=False,
                            name=name,
                            legendgroup=name,
                            legendgrouptitle_text="Baselines vs DL",
                        ),
                        row=row + 1,
                        col=col,
                    )

                if row == 2 * len(datasets):
                    fig.update_xaxes(
                        title_text="Lookback Window",
                        tickmode="array",
                        tickvals=look_back_windows,
                        row=row,
                        col=col,
                    )

    num_rows = num_datasets
    fig.update_layout(
        height=num_rows * 800,
        template="plotly_white",
    )

    fig.update_layout(legend=dict(font=dict(size=16)))

    for annotation in fig["layout"]["annotations"]:
        if annotation["text"] in row_titles:
            annotation["x"] = -0.02
            annotation["xanchor"] = "right"  # Align text to the right of x position
            annotation["font"] = dict(size=20, color="black", family="Arial")
            annotation["text"] = f"<b>{annotation['text']}</b>"  # Make text bold

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
