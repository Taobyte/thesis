import pandas as pd
import numpy as np
import wandb
import os
import plotly.graph_objects as go
import ast


from matplotlib import colors
from tqdm import tqdm
from collections import defaultdict
from itertools import product
from typing import Tuple


from src.utils import create_group_run_name
from src.constants import (
    METRICS,
    dataset_to_name,
)


def get_metrics(
    runs: list,
    metrics_to_keep: list[str] = [
        "MSE",
        "MAE",
        "DIRACC",
        "MASE",
        "ND",
        "NRMSE",
        "SMAPE",
    ],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        look_back_window = name_splitted[-2]
        prediction_window = name_splitted[-1]
        summary = run.summary._json_dict
        filtered_summary = {k: summary[k] for k in summary if k in METRICS}
        metrics[model_name][look_back_window][prediction_window].append(
            filtered_summary
        )
        model_run_counts[model_name] += 1

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
                        if metric_name in metrics_to_keep:
                            metric_list[metric_name].append(metric_value)

                mean = {
                    metric_name: float(np.mean(v))
                    for metric_name, v in metric_list.items()
                }
                std = {
                    metric_name: float(np.std(v))
                    for metric_name, v in metric_list.items()
                }
                processed_metrics_mean[model][lbw][pw] = mean
                processed_metrics_std[model][lbw][pw] = std

    return processed_metrics_mean, processed_metrics_std


def get_runs(
    dataset: str,
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    normalization: str = "global",
    start_time: str = "2025-6-12",
    window_statistic: str = None,
    experiment_name: str = "endo_exo",
):
    if normalization == "all":
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
                fold_nr=-1,  # does not matter, we only want group_name
                fold_datasets=[],  # does not matter
                normalization=normalization,
                experiment_name=experiment_name,
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
    use_std: bool = False,
):
    look_back_windows = sorted(list(mean_dict[model].keys()), key=int)
    prediction_windows = [
        sorted(list(mean_dict[model][look_back_windows[0]].keys()))[row_delta // 2]
    ]

    assert len(prediction_windows) == 1
    assert set(test_metrics) == set(
        mean_dict[model][look_back_windows[0]][prediction_windows[0]]
    ), (
        f"Model {model} has not all test metrics! {mean_dict[model][look_back_windows[0]][prediction_windows[0]]} "
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
            if use_std:
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

                try:
                    param_dict = ast.literal_eval(parameter_line[start:])
                except (ValueError, SyntaxError) as e:
                    print(
                        f"[WARN] Skipping run {run.name}: Failed to parse parameters → {e}"
                    )
                    os.remove(path)
                    continue  # Skip this run and go to the next one

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
