import os
import ast
import time
import pandas as pd
import numpy as np
import wandb
import argparse
import plotly.graph_objects as go
import plotly.io as pio

from tqdm import tqdm
from matplotlib import colors
from plotly.subplots import make_subplots
from collections import defaultdict
from itertools import product
from typing import Tuple
from collections import OrderedDict

from utils import create_group_run_name
from src.constants import (
    test_metrics,
    model_colors,
    metric_to_name,
    model_to_name,
    dataset_to_name,
)


def get_metrics(runs: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the run metrics for the wandb training runs in the runs list.
    Returns two dataframes, the first storing the mean and the second the standard deviation for
    each metric MSE, MAE, Cross Correlation and Directional Accuracy.
    """

    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    model_run_counts = defaultdict(int)
    for run in tqdm(runs):
        name_splitted = run.name.split("_")
        model_name = name_splitted[4]
        model_run_counts[model_name] += 1
        look_back_window = name_splitted[-2]
        prediction_window = name_splitted[-1]
        summary = run.summary._json_dict
        filtered_summary = {k: summary[k] for k in summary if k in test_metrics}
        metrics[model_name][look_back_window][prediction_window].append(
            filtered_summary
        )

    for k, v in model_run_counts.items():
        print(f"Model {k}: {v}")

    processed_metrics_mean = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    processed_metrics_std = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for model, v in metrics.items():
        for lbw, w in v.items():
            for pw, z in w.items():
                metric_list = defaultdict(list)
                for metric_dict in z:
                    for metric_name, metric_value in metric_dict.items():
                        metric_list[metric_name].append(metric_value)

                mean = {
                    metric_name: np.mean(v) for metric_name, v in metric_list.items()
                }
                std = {metric_name: np.std(v) for metric_name, v in metric_list.items()}
                processed_metrics_mean[model][lbw][pw] = mean
                processed_metrics_std[model][lbw][pw] = std

    return processed_metrics_mean, processed_metrics_std


def get_runs(
    dataset: str,
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    use_dynamic_features: bool,
    use_static_features: bool,
    normalization: str = "global",
    start_time: str = "2025-6-12",
    window_statistic: str = None,
):
    if normalization is None:
        normalizations = ["global", "local", "none"]
    else:
        normalizations = [normalization]
    group_names = []
    for lbw, pw in product(look_back_window, prediction_window):
        for normalization in normalizations:
            group_name, run_name, _ = create_group_run_name(
                dataset,
                "",  # doesn't matter
                use_heart_rate,
                lbw,
                pw,
                use_dynamic_features,
                use_static_features,
                fold_nr=-1,  # does not matter, we only want group_name
                fold_datasets=[],  # does not matter
                normalization=normalization,
            )

            group_names.append(group_name)

    conditions = [
        {"group": {"$in": group_names}},
        {"state": "finished"},
        {"created_at": {"$gte": start_time}},
        {"config.model.name": {"$in": models}},
    ]

    if window_statistic:
        conditions.append(
            {"config.dataset.datamodule.window_statistic": {"$eq": window_statistic}}
        )

    filters = {"$and": conditions}

    api = wandb.Api()
    runs = api.runs("c_keusch/thesis", filters=filters)
    print(f"Found {len(runs)} runs.")

    assert len(runs) % 3 == 0, "Attention, length of runs is not divisible by 3!"
    assert len(runs) > 0, "No runs were found!"
    return runs


def add_model_mean_std_to_fig(
    model: str,
    model_name: str,
    model_color: str,
    mean_dict: dict,
    std_dict: dict,
    fig: go.Figure,
    dataset: str,
    row_idx: int = None,
    ablation: bool = False,
    row_delta: int = 0,
    col_delta: int = 0,
):
    look_back_windows = sorted(list(mean_dict[model].keys()), key=int)
    prediction_windows = [
        sorted(list(mean_dict[model][look_back_windows[0]].keys()))[row_delta // 2]
    ]

    assert len(prediction_windows) == 1
    assert set(test_metrics) == set(
        mean_dict[model][look_back_windows[0]][prediction_windows[0]]
    )

    x = [int(lbw) for lbw in look_back_windows]
    mse_upper = {"dalia": 10, "wildppg": 200, "ieee": 100}
    mae_upper = {"dalia": 5, "wildppg": 20, "ieee": 10}
    y_axis_ranges = {
        test_metrics[0]: [0, mse_upper[dataset]],
        test_metrics[1]: [0, mae_upper[dataset]],
        test_metrics[2]: [-1, 1],
        test_metrics[3]: [0, 1],
    }

    for pw in prediction_windows:
        for i, metric in enumerate(test_metrics):
            means = [mean_dict[model][lbw][pw][metric] for lbw in look_back_windows]
            stds = [std_dict[model][lbw][pw][metric] for lbw in look_back_windows]

            upper = [m + s for m, s in zip(means, stds)]
            lower = [m - s for m, s in zip(means, stds)]

            if row_idx is None:
                row, col = divmod(i, 2)
                row += 1 + row_delta
                col += 1 + col_delta
            else:
                col = i + 1  # plotly indexing starts at 1 not 0
                row = row_idx

            color = model_color
            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=means,
                    mode="lines+markers",
                    name=f"{dataset_to_name[dataset]} {model_name}"
                    if not ablation
                    else model_name[: model_name.find("Activity") + 8],
                    line=dict(color=color),
                    showlegend=(i == 0) and (row == 1),
                    legendgroup=f"{dataset_to_name[dataset]} {model_name}"
                    if not ablation
                    else model_name[: model_name.find("Activity") + 8],
                    # legendgrouptitle_text=model_name if i == 0 else None,
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
                    name=model_name,
                    legendgroup=model_name,
                ),
                row=row,
                col=col,
            )
            # Set y-axis range for this subplot
            # ig.update_yaxes(range=y_axis_ranges[metric], row=row, col=col)
            # Set x-axis to look_back_window values
            fig.update_xaxes(
                title_text="Lookback Window",
                tickmode="array",
                tickvals=look_back_windows,
                row=row,
                col=col,
            )


def dynamic_feature_ablation(
    datasets: list[str],
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    use_static_features: bool,
    start_time: str = "2025-6-24",
    normalization: str = "global",
    save_html: bool = False,
    window_statistic: str = None,
):
    n_models = len(models)
    current_time = int(time.time())

    for dataset in datasets:
        dynamic_runs = get_runs(
            dataset,
            models,
            look_back_window,
            prediction_window,
            use_heart_rate,
            True,
            use_static_features,
            normalization,
            start_time,
            window_statistic=window_statistic,
        )

        no_dynamic_runs = get_runs(
            dataset,
            models,
            look_back_window,
            prediction_window,
            use_heart_rate,
            False,
            use_static_features,
            normalization,
            start_time,
            window_statistic=window_statistic,
        )
        dynamic_mean_dict, dynamic_std_dict = get_metrics(dynamic_runs)
        no_dynamic_mean_dict, no_dynamic_std_dict = get_metrics(no_dynamic_runs)
        for p, pw in enumerate(prediction_window):
            fig = make_subplots(
                rows=n_models,
                cols=4,
                column_titles=[metric_to_name[metric] for metric in test_metrics],
                row_titles=[model_to_name[model] for model in models],
                shared_xaxes=False,
            )
            for i, model in tqdm(enumerate(models), total=len(models)):
                add_model_mean_std_to_fig(
                    model,
                    f"Activity {model_to_name[model]}",
                    model_colors[2],
                    dynamic_mean_dict,
                    dynamic_std_dict,
                    fig,
                    dataset,
                    i + 1,
                    True,
                    row_delta=2 * p,
                )

                add_model_mean_std_to_fig(
                    model,
                    f"No Activity {model_to_name[model]}",
                    model_colors[0],
                    no_dynamic_mean_dict,
                    no_dynamic_std_dict,
                    fig,
                    dataset,
                    i + 1,
                    True,
                    row_delta=2 * p,
                )

            fig.update_layout(
                title_text=f"Activity Ablation | Dataset {dataset_to_name[dataset]} | Prediction Windows {pw} | Window Statistic {window_statistic}",
                height=n_models * 400,
                width=1200,
                template="plotly_white",
            )

            fig.update_xaxes(title_text="Lookback Window")
            fig.update_yaxes(title_text="Metric Value")

            if save_html:
                plot_name = f"{current_time}_{dataset}_{window_statistic}_lbw_{'_'.join([str(lbw) for lbw in look_back_window])}_pw_{pw}_{'_'.join(models)}"
                base_dir = f"./plots/ablations/{window_statistic}"
                os.makedirs(base_dir, exist_ok=True)
                pio.write_html(
                    fig,
                    file=f"{base_dir}/{plot_name}.html",
                    auto_open=True,
                )
            else:
                fig.show()


def visualize_look_back_window_difference(
    datasets: list[str],
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    use_dynamic_features: bool,
    use_static_features: bool,
    normalization: str,
    start_time: str = "2025-6-05",
    save_html: bool = False,
):
    num_datasets = len(datasets)
    num_preds = len(prediction_window)
    n_metrics = 4
    offset = num_preds + 1

    readable_metric_names = [metric_to_name[m] for m in test_metrics]
    block_titles = n_metrics * [""] + (num_preds * readable_metric_names)
    subplot_titles = num_datasets * block_titles

    mse_upper = {"dalia": 10, "wildppg": 200, "ieee": 100}
    mae_upper = {"dalia": 5, "wildppg": 20, "ieee": 10}

    fig = make_subplots(
        rows=num_datasets * offset,
        cols=n_metrics,
        subplot_titles=subplot_titles,
        # shared_xaxes=True,
        # horizontal_spacing=0.1,
        # vertical_spacing=0.1,
    )
    for b, dataset in tqdm(enumerate(datasets), total=num_datasets):
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

        dataset_name = dataset_to_name[dataset]
        y_axis_ranges = {
            test_metrics[0]: [0, mse_upper[dataset]],
            test_metrics[1]: [0, mae_upper[dataset]],
            test_metrics[2]: [-1, 1],
            test_metrics[3]: [0, 1],
        }

        mean_dict, std_dict = get_metrics(runs)

        for i, pw in enumerate(prediction_window):
            for j, metric in enumerate(test_metrics):
                for m, model in enumerate(models):
                    row = b * offset + i + 2
                    col = j + 1

                    model_name = model_to_name[model]

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

                    color = model_colors[m]

                    # Mean line
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=means,
                            mode="lines+markers",
                            name=f"{dataset_name} {model_name}",
                            line=dict(color=color),
                            showlegend=(j == 0)
                            and (row in [2 + k * offset for k in range(num_datasets)]),
                            legendgroup=f"{dataset_name} {model_name}",
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
                            name=model_name,
                            legendgroup=model_name,
                        ),
                        row=row,
                        col=col,
                    )
                    # Set y-axis range for this subplot
                    # fig.update_yaxes(range=y_axis_ranges[metric], row=row, col=col)
                    # Set x-axis to look_back_window values
                    fig.update_xaxes(
                        title_text="Lookback Window",
                        tickmode="array",
                        tickvals=look_back_windows,
                        row=row,
                        col=col,
                    )

    num_rows = (num_preds + 1) * num_datasets
    num_cols = n_metrics

    for i, dataset in enumerate(datasets):  # or however many row groups you have
        # add dataset title
        fig.add_annotation(
            text=f"<b>Dataset {dataset_to_name[dataset]}</b>",
            xref="paper",
            yref="paper",
            x=0.5,
            y=(1 - i / num_datasets) - (1 / (2 * num_rows)),
            showarrow=False,
            font=dict(size=30),
            textangle=0,
            xanchor="center",
            yanchor="middle",
        )

    for row in range(num_rows):
        if row not in [i * offset for i in range(0, num_datasets)]:
            pw = prediction_window[row % offset - 1]
            fig.add_annotation(
                text=f"<b>Prediction Window = {pw}</b>",
                xref="paper",
                yref="paper",
                x=-0.05,
                y=(1 - row / num_rows),
                showarrow=False,
                font=dict(size=10),
                textangle=-90,
                xanchor="left",
                yanchor="middle",
            )

    activity_string = "Activity" if use_dynamic_features else "No Activity"
    fig.update_layout(
        title={
            "text": f"<b>Lookback Window Ablations | {activity_string}</b>",
            "x": 0.5,
            "xanchor": "center",
            "font": dict(
                size=40, family="Arial", color="black"
            ),  # bold by default for many fonts
        },
        height=num_rows * 400,
        width=num_cols * 600,
        template="plotly_white",
        # margin=dict(t=100, b=100, l=100, r=100),
    )

    fig.show()

    if save_html:
        plot_name = f"{dataset}_{use_heart_rate}_{use_dynamic_features}_{'_'.join(models)}_{'_'.join([str(lbw) for lbw in look_back_window])}_{'_'.join([str(pw) for pw in prediction_window])}"
        pio.write_html(fig, file=f"{plot_name}.html", auto_open=True)


def visualize_metric_table(
    fig: go.Figure,
    index: int,
    df: pd.DataFrame,
    df_std: pd.DataFrame,
):
    font_weights = []
    for i in range(len(df)):
        row = df.iloc[i, :]
        bold_idx = (
            np.argmin(row) if i < 2 else np.argmax(row)
        )  # i == 2 is cross correlation
        weights = ["normal"] * len(row)
        weights[bold_idx] = "bold"
        # weights.insert(0, "normal")
        font_weights.append(weights)

    font_weights = np.array(font_weights)

    df.columns = [model_to_name[column] for column in df.columns]
    df_std.columns = [model_to_name[column] for column in df_std.columns]

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric"] + list(df.columns),
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[[metric_to_name[i] for i in df.index]]
                + [
                    [
                        f"<b>{mean:.3f} ± {std:.3f}</b>"
                        if weight == "bold"
                        else f"{mean:.3f} ± {std:.3f}"
                        for mean, std, weight in zip(
                            df[column], df_std[column], font_weights[:, i]
                        )
                    ]
                    for i, column in enumerate(df.columns)
                ],
                fill_color="lavender",
                align="left",
                font=dict(
                    size=11,
                    color="black",
                ),
            ),
        ),
        row=index,
        col=1,
    )


def plot_tables(
    dataset: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    use_dynamic_features: bool,
    use_static_features: bool,
    start_time: str = "2025-5-25",
):
    assert len(dataset) == 1

    print(
        f"Plotting tables for \n Dataset: {dataset[0]} \n Use Heartrate: {use_heart_rate} \n Dynamic Features: {use_dynamic_features}  \n Lookback Windows: {look_back_window} \n Prediction Windows: {prediction_window}"
    )

    n_combinations = len(look_back_window) * len(prediction_window)

    fig = make_subplots(
        rows=n_combinations,
        cols=1,
        specs=[[{"type": "table"}] for _ in range(n_combinations)],
        subplot_titles=[
            f"Lookback Window: {lbw} | Prediction Window: {pw}"
            for lbw, pw in product(look_back_window, prediction_window)
        ],
    )

    for i, (lbw, pw) in enumerate(product(look_back_window, prediction_window)):
        group_name, run_name, tags = create_group_run_name(
            dataset,
            "",
            use_heart_rate,
            lbw,
            pw,
            use_dynamic_features,
            use_static_features,
            fold_nr=-1,  # does not matter, we only want group_name
            fold_datasets=[],  # does not matter
        )

        filters = {
            "$and": [
                {"group": group_name},
                {"state": "finished"},
                {"created_at": {"$gte": start_time}},
            ]
        }

        api = wandb.Api()
        runs = api.runs("c_keusch/thesis", filters=filters)

        print(f"Found {len(runs)} runs.")

        assert len(runs) % 3 == 0, "Attention, length of runs is not divisible by 3!"

        mean_dict, std_dict = get_metrics(runs)

        df = pd.DataFrame.from_dict(
            {k: v[str(lbw)][str(pw)] for k, v in mean_dict.items()}
        )

        df_std = pd.DataFrame.from_dict(
            {k: v[str(lbw)][str(pw)] for k, v in std_dict.items()}
        )

        visualize_metric_table(
            fig,
            i + 1,  # indexing starts at 1 in plotly subplot figures
            df,
            df_std,
        )

    fig.update_layout(
        title_text=f"Model Performance Metrics for {dataset_to_name[dataset]} \n"
        f"Lookback: {look_back_window}, Prediction: {prediction_window}, \n"
        f"HR: {use_heart_rate}, Dynamic Features: {use_dynamic_features}",
        title_x=0.5,
        height=n_combinations * 500,
        width=800 + len(df) * 100,
    )

    fig.show()


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
                column_titles=[metric_to_name[metric] for metric in test_metrics],
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
                    for j, metric in enumerate(test_metrics):
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


def create_params_file_from_optuna(models: list[str], start_time: str):
    import yaml

    assert len(models) == 1
    model = models[0]
    filters = {"$and": [{"created_at": {"$gte": start_time}}]}
    api = wandb.Api()
    runs = api.runs("c_keusch/thesis", filters=filters)
    print(f"Found {len(runs)} runs.")

    filtered_runs = [
        run for run in runs if run.name and run.name.startswith(f"optuna_{model}")
    ]

    final_dict = defaultdict(lambda: defaultdict(dict))
    for run in filtered_runs:
        for file in run.files():
            if file.name == "output.log":
                path = file.download(replace=True).name
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                last_two_lines = [line.strip() for line in lines if line.strip()][-2:]

                name_splitted = run.name.split("_")
                dynamic_info = str("df" in name_splitted)
                lbw = name_splitted[-2]
                pw = name_splitted[-1]

                parameter_line = last_two_lines[0]
                keyword = "Best parameters: "
                start = parameter_line.find(keyword) + len(keyword)

                param_dict = ast.literal_eval(parameter_line[start:])
                param_dict = {
                    key.split(".")[-1]: value for key, value in param_dict.items()
                }

                final_dict[dynamic_info][lbw][pw] = param_dict

                print(param_dict)
                print(type(param_dict))

                os.remove(path)

    def to_regular_dict(d):
        if isinstance(d, defaultdict):
            d = {k: to_regular_dict(v) for k, v in d.items()}
        elif isinstance(d, dict):
            d = {k: to_regular_dict(v) for k, v in d.items()}
        return d

    clean_dict = to_regular_dict(final_dict)
    sorted_dict = dict(
        sorted(
            clean_dict.items(),
            key=lambda item: int(item[0]) if item[0].isdigit() else item[0],
        )
    )

    yaml_str = yaml.dump(sorted_dict, sort_keys=False)
    print(yaml_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WandB Results")

    def list_of_ints(arg):
        return [int(i) for i in arg.split(",")]

    def list_of_strings(arg):
        return arg.split(",")

    parser.add_argument(
        "--type",
        type=str,
        choices=["table", "viz", "activity_ablation", "norm_ablation", "optuna"],
        required=True,
        default="table",
        help="Plot type. Must be either 'table', 'viz' or 'activity_ablation' .",
    )

    parser.add_argument(
        "--dataset",
        type=list_of_strings,
        required=True,
        default=["dalia"],
        help="Dataset must be one of the following: dalia, ieee, wildppg, chapman, ucihar, usc, capture24",
    )

    parser.add_argument(
        "--normalization",
        type=str,
        choices=[
            "none",
            "global",
            "local",
        ],
        required=False,
        default=None,
        help="Normalization must be 'none', 'global' or 'local' ",
    )

    parser.add_argument(
        "--look_back_window",
        type=list_of_ints,
        required=True,
        default=[5],
        help="Lookback window size",
    )

    parser.add_argument(
        "--prediction_window",
        type=list_of_ints,
        required=True,
        default=[3],
        help="Prediction window size",
    )

    parser.add_argument(
        "--use_heart_rate",
        required=False,
        action="store_true",
        help="get runs for heart rate only has an effect for datasets: dalia, ieee & wildppg",
    )
    parser.add_argument(
        "--use_dynamic_features",
        required=False,
        action="store_true",
        help="get runs trained with activity information",
    )
    parser.add_argument(
        "--use_static_features",
        required=False,
        action="store_true",
        default=False,
        help="get runs trained with activity information",
    )

    parser.add_argument(
        "--models",
        required=False,
        default=None,
        type=list_of_strings,
        help="Pass in the models you want to visualize the prediction and look back window. Must be separated by commas , without spaces between the model names! (Correct Example: timesnet,elastst | Wrong Example: gpt4ts, timellm )",
    )

    parser.add_argument(
        "--save_html",
        required=False,
        action="store_true",
        help="save html plot for sharing",
    )

    parser.add_argument(
        "--window_statistic",
        choices=["mean", "var", "power"],
        required=False,
        default=None,
        type=str,
        help="Which window statistic for the heartrate value to use. Can be 'mean', 'var' or 'power'. Only supported by DaLia dataset at the moment.",
    )

    args = parser.parse_args()

    if args.type == "table":
        plot_tables(
            args.dataset,
            args.look_back_window,
            args.prediction_window,
            args.use_heart_rate,
            args.use_dynamic_features,
            args.use_static_features,
        )
    elif args.type == "viz":
        visualize_look_back_window_difference(
            args.dataset,
            args.models,
            args.look_back_window,
            args.prediction_window,
            args.use_heart_rate,
            args.use_dynamic_features,
            args.use_static_features,
            args.normalization,
            save_html=args.save_html,
        )
    elif args.type == "activity_ablation":
        dynamic_feature_ablation(
            datasets=args.dataset,
            models=args.models,
            look_back_window=args.look_back_window,
            prediction_window=args.prediction_window,
            use_heart_rate=args.use_heart_rate,
            use_static_features=args.use_static_features,
            start_time="2025-6-24",
            normalization=args.normalization,
            save_html=args.save_html,
            window_statistic=args.window_statistic,
        )
    elif args.type == "norm_ablation":
        visualize_normalization_difference(
            args.dataset,
            args.models,
            args.look_back_window,
            args.prediction_window,
            args.use_heart_rate,
            args.use_dynamic_features,
            args.use_static_features,
            save_html=args.save_html,
        )

    elif args.type == "optuna":
        create_params_file_from_optuna(models=args.models, start_time="2025-6-24")
