import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots
from matplotlib import colors
from tqdm import tqdm

from src.wandb_results.utils import get_metrics, get_runs
from src.constants import (
    METRICS,
    MODELS,
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
    subplot_titles = subplot_titles = readable_metric_names + [""] * (
        (num_datasets - 1) * n_metrics
    )
    row_titles = readable_dataset_names

    fig = make_subplots(
        rows=num_datasets,
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
                for m, model in enumerate(models):
                    row = b + 1
                    col = j + 1

                    model_name = model_to_name[model]

                    look_back_windows = sorted(list(mean_dict[model].keys()), key=int)
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
                            # legendgrouptitle_text=dataset_name,
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
                    if row == len(datasets):
                        fig.update_xaxes(
                            title_text="Lookback Window",
                            tickmode="array",
                            tickvals=look_back_windows,
                            row=row,
                            col=col,
                        )

    num_rows = num_datasets
    fig.update_layout(
        height=num_rows * 400,
        template="plotly_white",
    )

    fig.update_layout(
        legend=dict(
            font=dict(
                size=16  # <-- Change this value to increase/decrease size
            )
        )
    )

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
