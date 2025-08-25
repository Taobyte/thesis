import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots
from matplotlib import colors
from tqdm import tqdm
from typing import List, Tuple

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
    dataset_colors,
)


TITLE_SIZE = 22 * 2
AXIS_TITLE_SIZE = 40
TICK_SIZE = 32
LEGEND_SIZE = 40
LINE_WIDTH = 8
MARKER_SIZE = 15
LINE_OPACITY = 0.9


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
    metrics = ["SMAPE"]  # METRICS
    ALL_MODELS = models
    IEEE_MODELS = [
        "linear",
        # "xgboost",
        "kalmanfilter",
        "patchtst",
        # "timexer",
        "nbeatsx",
    ]
    DALIA_MODELS = ["xgboost", "mole", "simpletm", "timexer"]  #  "mlp", "nbeatsx"
    use_specific_models = 0
    plot_aggregated_results = 0

    num_datasets = len(datasets)
    n_metrics = len(metrics)
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

    readable_dataset_names = [dataset_to_name[d] for d in datasets]
    subplot_titles = readable_dataset_names

    fig = make_subplots(
        rows=n_metrics + plot_aggregated_results,
        cols=num_datasets,
        subplot_titles=subplot_titles,
        # row_titles=row_titles,
        shared_xaxes=True,
        horizontal_spacing=0.03,
        vertical_spacing=0.03,
    )

    for b, dataset in tqdm(enumerate(datasets), total=num_datasets):
        if use_specific_models:
            if dataset == "dalia":
                models = DALIA_MODELS
            elif dataset == "ieee":
                models = IEEE_MODELS
            else:
                models = ALL_MODELS

        runs = get_runs(
            dataset,
            look_back_window,
            prediction_window,
            models,
            start_time,
            experiment_name=experiment,
        )

        mean_dict, std_dict = get_metrics(runs)

        for j, metric in enumerate(metrics):
            assert set(mean_dict.keys()) == set(models), (
                f"Models froms runs: {mean_dict.keys()} | Models from cmd {models}"
            )
            factor = 100 if metric == "SMAPE" else 1
            baseline_means: List[float] = []
            dl_means: List[float] = []
            row = j + 1
            col = b + 1
            for m, model in enumerate(models):
                model_name = model_to_name[model]

                means: list[float] = []
                stds: list[float] = []
                for ablation_x in x:
                    if lbw_ablation:
                        if metric in mean_dict[model][str(ablation_x)][str(pw)]:
                            means.append(
                                factor
                                * mean_dict[model][str(ablation_x)][str(pw)][metric]
                            )
                            stds.append(
                                std_dict[model][str(ablation_x)][str(pw)][metric]
                            )
                        else:
                            print(
                                f"{model} {str(ablation_x)} {str(pw)} {metric} does not exist"
                            )
                    else:
                        means.append(
                            factor * mean_dict[model][str(lbw)][str(ablation_x)][metric]
                        )
                        stds.append(std_dict[model][str(lbw)][str(ablation_x)][metric])

                if model in BASELINES:
                    baseline_means.append(means)
                elif model in DL:
                    dl_means.append(means)

                upper = [m + s for m, s in zip(means, stds)]
                lower = [m - s for m, s in zip(means, stds)]

                fig.add_trace(
                    go.Scatter(
                        x=x_pos,
                        y=means,
                        mode="lines+markers",
                        name=model_name,
                        line=dict(color=model_colors[model], width=LINE_WIDTH),
                        marker=dict(size=MARKER_SIZE),
                        opacity=LINE_OPACITY,
                        showlegend=(row == 1) and col == 1,
                        legendgroup=model_name,
                    ),
                    row=row,
                    col=col,
                )

                # Std deviation band (fill between)
                if use_std:
                    fig.add_trace(
                        go.Scatter(
                            x=x_pos + x_pos[::-1],
                            y=upper + lower[::-1],
                            fill="toself",
                            fillcolor=color.replace("1.0", "0.2")
                            if "rgba" in color
                            else f"rgba({','.join(str(int(c * 255)) for c in colors.to_rgb(color))},0.2)",
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            showlegend=False,
                            name=model_name,
                        ),
                        row=row,
                        col=col,
                    )

            if plot_aggregated_results:
                baseline_means_mean = np.mean(
                    np.stack(baseline_means), axis=0
                ).tolist()  # (len(look_back_window),)
                dl_means_mean = np.mean(np.stack(dl_means), axis=0).tolist()
                baseline_means_std = np.std(np.stack(baseline_means), axis=0).tolist()
                # (len(look_back_window),)
                dl_means_std = np.std(np.stack(dl_means), axis=0).tolist()

                for name, color, y, stds in zip(
                    ["Baselines", "DL"],
                    [model_colors["baseline"], model_colors["dl"]],
                    [baseline_means_mean, dl_means_mean],
                    [baseline_means_std, dl_means_std],
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=x_pos,
                            y=y,
                            mode="lines+markers",
                            name=name,
                            showlegend=(j == 0) and row == 1,
                            line=dict(color=color),
                            legendgroup=name,
                            # legendgrouptitle_text="Baselines vs DL",
                        ),
                        row=row + 1,
                        col=col,
                    )

                    upper = [m + s for m, s in zip(y, stds)]
                    lower = [m - s for m, s in zip(y, stds)]

                    fig.add_trace(
                        go.Scatter(
                            x=x_pos + x_pos[::-1],
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

            fig.update_xaxes(
                title_text="Lookback Window" if lbw_ablation else "Prediction Window",
                tickmode="array",
                tickvals=x_pos,
                ticktext=x_text,
                title_font=dict(size=AXIS_TITLE_SIZE),
                tickfont=dict(size=TICK_SIZE),
                row=row,
                col=col,
            )

    # LEGEND POSITION
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            title_text="Model",
            itemsizing="trace",
        ),
        margin=dict(b=30),
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

    fig.update_layout(legend=dict(font=dict(size=16)))

    for ann in fig["layout"]["annotations"]:
        if ann["text"] in readable_dataset_names:
            ann["font"] = dict(size=TITLE_SIZE, color="black", family="Arial")
            ann["text"] = f"<b>{ann['text']}</b>"

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
