import pandas as pd
import numpy as np
import wandb
import plotly.graph_objects as go


from plotly.subplots import make_subplots
from tqdm import tqdm
from itertools import product
from typing import List

from src.wandb_results.utils import get_metrics, get_runs
from utils import create_group_run_name
from src.constants import (
    model_to_name,
    metric_to_name,
    dataset_to_name,
    MODELS,
    METRICS,
)


def plot_best_lbw(
    datasets: list[str],
    prediction_window: list[int] = [3],
    experiment: str = "best_endo_exo",
    start_time: str = "2025-8-08",
    models: list[str] = [],
):
    assert len(prediction_window) == 1
    if len(models) == 0:
        models = MODELS
    pw = prediction_window[0]
    metrics_to_keep = METRICS
    for metric in ["NRMSE", "ND"]:
        metrics_to_keep.remove(metric)

    metrics_to_keep += ["LBW"]  # also include the best LBW
    n_metrics = len(metrics_to_keep)
    cols: dict[str, list[float]] = dict()

    dataset_col: List[str] = []
    for dataset_name in datasets:
        human_readable_name = dataset_to_name[dataset_name]
        dataset_col.append(rf"\multirow{{{n_metrics}}}{{*}}{{{human_readable_name}}}")
        for _ in range(len(metrics_to_keep) - 1):
            dataset_col.append("")

    cols["Datasets"] = dataset_col

    metric_col: list[str] = []
    metric_col = metrics_to_keep * len(datasets)
    cols["Metrics"] = metric_col

    for model_name in models:
        col: List[float] = []
        for dataset in datasets:
            conditions = [
                # HERE I WANT THAT TAGS CONTAINS dataset name and experiment name!
                {"state": "finished"},
                {"created_at": {"$gte": start_time}},
                {"tags": dataset},  # dataset tag must be present
                {"tags": experiment},  # experiment tag must be present
                {"tags": model_name},
            ]
            filters = {"$and": conditions}

            api = wandb.Api()
            best_endo_exo_runs = api.runs("c_keusch/thesis", filters=filters)
            assert len(best_endo_exo_runs) > 0, "No runs were found!"
            assert len(best_endo_exo_runs) % 3 == 0, (
                "Attention, length of runs is not divisible by 3!"
            )
            for run in best_endo_exo_runs:
                print(f"Lookback window {run.config['look_back_window']}")

            best_endo_exo_mean, best_endo_exo_std = get_metrics(best_endo_exo_runs)
            if model_name not in best_endo_exo_mean:
                print(f"Attention {model_name} is not in the dict!")
                continue

            lbw = list(best_endo_exo_mean[model_name].keys())[0]
            metrics: list[float] = []
            for metric_name in metrics_to_keep[:-1]:
                factor = 100 if metric_name == "SMAPE" else 1
                if metric_name in best_endo_exo_mean[model_name][str(lbw)][str(pw)]:
                    metrics.append(
                        factor
                        * best_endo_exo_mean[model_name][str(lbw)][str(pw)][metric_name]
                    )
                else:
                    metrics.append(np.nan)
                    print(
                        f"Metric {metric_name} does not exist for {dataset} {model_name} {lbw}"
                    )

            metrics.append(lbw)
            assert len(metrics) == len(metrics_to_keep)
            col.extend(metrics)
        cols[model_to_name[model_name]] = col

    df = pd.DataFrame(cols)

    n_best = 3  # take the best three values and make them bold, double underlined and underlined

    # make best value bold
    for i in range(len(df)):
        # skip LBW rows
        if i % n_metrics == (n_metrics - 1):
            continue
        row = df.iloc[i].values
        current_row = row[2:]
        if i % n_metrics == 2:
            # DIRACC is at position 3 in the metric list
            best_indices = np.argsort(current_row)[-n_best:] + 2
            worst_idx = current_row.argmin() + 2
        else:
            best_indices = (np.argsort(current_row)[:n_best])[::-1] + 2
            worst_idx = current_row.argmax() + 2

        best_indices = best_indices.tolist()

        third_best_val = df.iloc[i, best_indices[0]]
        second_best_val = df.iloc[i, best_indices[1]]
        best_val = df.iloc[i, best_indices[2]]
        worst_val = df.iloc[i, worst_idx]

        df.iloc[i, best_indices[0]] = rf"\underline{{{float(third_best_val):.3f}}}"
        df.iloc[i, best_indices[1]] = (
            rf"\underline{{\underline{{{float(second_best_val):.3f}}}}}"
        )
        df.iloc[i, best_indices[2]] = rf"\textbf{{{float(best_val):.3f}}}"
        df.iloc[i, worst_idx] = rf"\textit{{{float(worst_val):.3f}}}"

        for j in range(len(current_row)):
            abs_col = j + 2
            if abs_col in ([worst_idx] + best_indices):
                continue
            v = df.iloc[i, abs_col]
            df.iloc[i, abs_col] = f"{float(v):.3f}"

    latex_str = df.to_latex(
        index=False,
        escape=False,
        header=True,
        # column_format=column_format,
        bold_rows=False,
        # float_format="%.3f",
    )
    print(latex_str)


def compare_endo_exo_latex_tables(
    datasets: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    start_time: str = "2025-8-08",
):
    assert len(prediction_window) == 1
    pw = prediction_window[0]
    metrics_to_keep = ["MSE", "MAE", "DIRACC"]
    cols: dict[str, list[float]] = dict()
    fst_col: list[str] = []
    for model_name in MODELS:
        human_readable_name = model_to_name[model_name]
        fst_col.append(rf"\multirow{{3}}{{*}}{{{human_readable_name}}}")
        for _ in range(len(metrics_to_keep) - 1):
            fst_col.append("")
    cols["fst_col"] = fst_col
    cols["snd_col"] = ["MSE", "MAE", "DIR"] * len(MODELS)

    for dataset in datasets:
        endo_only_runs = get_runs(
            dataset,
            models=MODELS,
            look_back_window=look_back_window,
            prediction_window=prediction_window,
            use_heart_rate=True,
            experiment_name="endo_only",
            start_time=start_time,
            normalization="all",
        )
        endo_only_mean, _ = get_metrics(endo_only_runs, metrics_to_keep=metrics_to_keep)
        endo_exo_runs = get_runs(
            dataset,
            models=MODELS,
            look_back_window=look_back_window,
            prediction_window=prediction_window,
            use_heart_rate=True,
            experiment_name="endo_exo",
            start_time=start_time,
            normalization="all",
        )
        endo_exo_mean, _ = get_metrics(endo_exo_runs, metrics_to_keep=metrics_to_keep)
        for lbw in look_back_window:
            col: list[float] = []
            for model_name in MODELS:
                if model_name not in endo_exo_mean or model_name not in endo_only_mean:
                    print(f"Attention {model_name} is not in the dict!")
                    continue
                metrics: list[float] = []
                for metric_name in metrics_to_keep:
                    if (
                        metric_name in endo_only_mean[model_name][str(lbw)][str(pw)]
                        and metric_name in endo_exo_mean[model_name][str(lbw)][str(pw)]
                    ):
                        endo_only_value = endo_only_mean[model_name][str(lbw)][str(pw)][
                            metric_name
                        ]
                        endo_exo_value = endo_exo_mean[model_name][str(lbw)][str(pw)][
                            metric_name
                        ]
                        if metric_name in ["MSE", "MAE"]:
                            relative_improvement = (
                                100
                                * (endo_only_value - endo_exo_value)
                                / endo_only_value
                            )
                        else:
                            relative_improvement = (
                                100
                                * (endo_exo_value - endo_only_value)
                                / endo_only_value
                            )
                        metrics.append(relative_improvement)
                    else:
                        metrics.append(np.nan)
                        print(
                            f"Metric {metric_name} does not exist for {dataset} {model_name} {lbw}"
                        )
                assert len(metrics) == len(metrics_to_keep)
                col.extend(metrics)
            cols[f"{dataset}_{lbw}"] = col

    df = pd.DataFrame(cols)

    # make best value bold
    for col in df.columns[2:]:
        bold_indices: list[int] = []
        worst_indices: list[int] = []
        for i in range(len(metrics_to_keep)):
            best_val = df[col].iloc[i::3].max()
            worst_val = df[col].iloc[i::3].min()
            series = df[col]
            idx: list[int] = series[series == best_val].index.to_list()
            bold_indices.extend(idx)
            idx: list[int] = series[series == worst_val].index.to_list()
            worst_indices.extend(idx)
        processed_column = []
        for i, v in df[col].items():
            if i in bold_indices:
                processed_column.append(rf"\textbf{{{v:.3f}}}")
            elif i in worst_indices:
                processed_column.append(rf"\textit{{{v:.3f}}}")
            else:
                processed_column.append(f"{v:.3f}")

        df[col] = processed_column

    column_format = "|c|c|" + (
        (("c " * len(look_back_window))[:-1] + "|") * len(datasets)
    )
    latex_str = df.to_latex(
        index=False,
        escape=False,
        header=False,
        column_format=column_format,
        bold_rows=False,
        float_format="%.3f",
    )
    print(latex_str)


def latex_metric_table(
    datasets: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    experiment_name: str = "endo_exo",
    start_time: str = "2025-8-08",
):
    assert len(prediction_window) == 1
    assert experiment_name in ["endo_only", "endo_exo"]
    if experiment_name == "endo_exo":
        # add hierarchical forecasting approach to MODELS
        MODELS.insert(1, "hlinear")
        MODELS.insert(4, "hxgboost")
    pw = prediction_window[0]
    metrics_to_keep = ["MSE", "MAE", "DIRACC"]
    cols: dict[str, list[float]] = dict()
    fst_col: list[str] = []
    for model_name in MODELS:
        human_readable_name = model_to_name[model_name]
        fst_col.append(rf"\multirow{{3}}{{*}}{{{human_readable_name}}}")
        for _ in range(len(metrics_to_keep) - 1):
            fst_col.append("")
    cols["fst_col"] = fst_col
    cols["snd_col"] = ["MSE", "MAE", "DIR"] * len(MODELS)

    for dataset in datasets:
        runs = get_runs(
            dataset,
            models=MODELS,
            look_back_window=look_back_window,
            prediction_window=prediction_window,
            use_heart_rate=True,
            experiment_name=experiment_name,
            start_time=start_time,
            normalization="all",
        )

        mean, std = get_metrics(runs, metrics_to_keep=metrics_to_keep)
        for lbw in look_back_window:
            col: list[float] = []
            for model_name in MODELS:
                if model_name not in mean:
                    print(f"Attention {model_name} is not in the dict!")
                    continue
                metrics: list[float] = []
                for metric_name in metrics_to_keep:
                    if metric_name in mean[model_name][str(lbw)][str(pw)]:
                        metrics.append(mean[model_name][str(lbw)][str(pw)][metric_name])
                    else:
                        metrics.append(np.nan)
                        print(
                            f"Metric {metric_name} does not exist for {dataset} {model_name} {lbw}"
                        )
                assert len(metrics) == len(metrics_to_keep)
                col.extend(metrics)
            cols[f"{dataset}_{lbw}"] = col

    df = pd.DataFrame(cols)

    # make best value bold
    for col in df.columns[2:]:
        bold_indices: list[int] = []
        for i in range(len(metrics_to_keep)):
            best_val = df[col].iloc[i::3].min() if i < 2 else df[col].iloc[i::3].max()
            series = df[col]
            idx: list[int] = series[series == best_val].index.to_list()
            bold_indices.extend(idx)
        df[col] = [
            rf"\textbf{{{v:.3f}}}" if i in bold_indices else f"{v:.3f}"
            for i, v in df[col].items()
        ]

    column_format = "|c|c|" + (
        (("c " * len(look_back_window))[:-1] + "|") * len(datasets)
    )
    latex_str = df.to_latex(
        index=False,
        escape=False,
        header=False,
        column_format=column_format,
        bold_rows=False,
        float_format="%.3f",
    )
    print(latex_str)


def visualize_metric_table(
    fig: go.Figure,
    index: int,
    df: pd.DataFrame,
    df_std: pd.DataFrame,
):
    df = df[sorted(df.columns)]
    df_std = df_std[sorted(df_std.columns)]

    font_weights = []
    for i in range(len(df)):
        row = df.iloc[i, :]
        bold_idx = (
            np.argmin(row) if i > 0 else np.argmax(row)
        )  # i == 0 is directional accuracy
        weights = ["normal"] * len(row)
        weights[bold_idx] = "bold"
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
    experiment_name: str = "endo_exo",
    start_time: str = "2025-8-08",
):
    assert len(dataset) == 1
    dataset = dataset[0]

    print(
        f"Plotting tables for \n Dataset: {dataset} \n Use Heartrate: {use_heart_rate} \n Experiment Name: {experiment_name}  \n Lookback Windows: {look_back_window} \n Prediction Windows: {prediction_window}"
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

    for i, (lbw, pw) in tqdm(enumerate(product(look_back_window, prediction_window))):
        group_names = []
        for normalization in ["none", "difference", "global"]:
            group_name, run_name, tags = create_group_run_name(
                dataset,
                "",
                use_heart_rate,
                lbw,
                pw,
                fold_nr=-1,  # does not matter, we only want group_name
                fold_datasets=[],  # does not matter
                experiment_name=experiment_name,
                normalization=normalization,
            )
            group_names.append(group_name)

        filters = {
            "$and": [
                {"group": {"$in": group_names}},
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
        f"HR: {use_heart_rate} Experiment {experiment_name}",
        title_x=0.5,
        height=n_combinations * 500,
        width=800 + len(df) * 100,
    )

    fig.show()
