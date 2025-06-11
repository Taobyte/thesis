import pandas as pd
import numpy as np
import wandb
import argparse
import plotly.graph_objects as go

from tqdm import tqdm
from matplotlib import colors
from plotly.subplots import make_subplots
from collections import defaultdict
from itertools import product
from typing import Tuple

from utils import create_group_run_name

test_metrics = ["test_MSE", "test_MAE", "test_cross_correlation", "test_dir_acc_single"]

model_colors = [
    "blue",
    "green",
    "orange",
    "red",
    "purple",
    "pink",
    "brown",
    "olive",
    "cyan",
    "magenta",
    "lime",
    "teal",
    "navy",
    "gold",
    "gray",
]

metric_to_name = {
    "test_MSE": "Mean Squared Error",
    "test_MAE": "Mean Absolute Deviation",
    "test_cross_correlation": "Cross Correlation",
    "test_dir_acc_single": "Directional Accuracy",
}

model_to_name = {
    "timesnet": "TimesNet",
    "gpt4ts": "GPT4TS",
    "adamshyper": "AdaMSHyper",
    "timellm": "TimeLLM",
    "pattn": "PAttn",
    "simpletm": "SimpleTM",
    "elastst": "ElasTST",
    "gp": "Gaussian Process",
    "bnn": "Bayesian Neural Network",
    "kalmanfilter": "Kalman Filter",
    "linear": "Linear Regression",
    "hmm": "Hidden Markov Model",
    "xgboost": "XGBoost",
}

datset_to_name = {"dalia": "DaLiA", "wildppg": "WildPPG", "ieee": "IEEE"}


def get_metrics(runs: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the run metrics for the wandb training runs in the runs list.
    Returns two dataframes, the first storing the mean and the second the standard deviation for
    each metric MSE, MAE, Cross Correlation and Directional Accuracy.
    """

    metrics = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run in runs:
        name_splitted = run.name.split("_")
        model_name = name_splitted[3]
        look_back_window = name_splitted[-2]
        prediction_window = name_splitted[-1]
        summary = run.summary._json_dict
        filtered_summary = {k: summary[k] for k in summary if k in test_metrics}
        metrics[model_name][look_back_window][prediction_window].append(
            filtered_summary
        )

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
    start_time: str = "2025-6-09",
):
    group_names = []
    for lbw, pw, model in product(look_back_window, prediction_window, models):
        group_name, run_name, _ = create_group_run_name(
            dataset,
            model,
            use_heart_rate,
            lbw,
            pw,
            use_dynamic_features,
            use_static_features,
            fold_nr=-1,  # does not matter, we only want group_name
            fold_datasets=[],  # does not matter
        )

        if run_name.split("_")[1] in models:
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
    return runs


def add_model_mean_std_to_fig(
    model: str,
    model_name: str,
    model_color: str,
    mean_dict: dict,
    std_dict: dict,
    fig: go.Figure,
    dataset: str,
    row: int = None,
    ablation: bool = False,
):
    look_back_windows = sorted(list(mean_dict[model].keys()))
    prediction_windows = sorted(list(mean_dict[model][look_back_windows[0]].keys()))

    assert len(prediction_windows) == 1
    assert set(test_metrics) == set(
        mean_dict[model][look_back_windows[0]][prediction_windows[0]]
    )

    # look_back_windows = [int(lbw) for lbw in look_back_windows]
    # prediction_windows = [int(pw) for pw in prediction_windows]
    x = [int(lbw) for lbw in look_back_windows]
    mse_upper = {"dalia": 5, "wildppg": 200, "ieee": 50}
    mae_upper = {"dalia": 3, "wildppg": 20, "ieee": 5}
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

            if row is None:
                row, col = divmod(i, 2)
                row += 1
                col += 1
            else:
                col = i + 1  # plotly indexing starts at 1 not 0

            color = model_color
            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=means,
                    mode="lines+markers",
                    name=model_name
                    if not ablation
                    else model_name[: model_name.find("Activity") + 8],
                    line=dict(color=color),
                    showlegend=(i == 0) and (row == 1),
                    legendgroup=model_name
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
            fig.update_yaxes(range=y_axis_ranges[metric], row=row, col=col)
            # Set x-axis to look_back_window values
            fig.update_xaxes(
                title_text="Lookback Window",
                tickmode="array",
                tickvals=look_back_windows,
                row=row,
                col=col,
            )


def dynamic_feature_ablation(
    dataset: str,
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    use_static_features: bool,
    start_time: str = "2025-6-09",
):
    n_models = len(models)

    fig = make_subplots(
        rows=n_models,
        cols=4,
        column_titles=[metric_to_name[metric] for metric in test_metrics],
        row_titles=[model_to_name[model] for model in models],
        shared_xaxes=False,
    )

    for i, model in tqdm(enumerate(models), total=len(models)):
        dynamic_runs = get_runs(
            dataset,
            models,
            look_back_window,
            prediction_window,
            use_heart_rate,
            True,
            use_static_features,
            start_time,
        )
        no_dynamic_runs = get_runs(
            dataset,
            models,
            look_back_window,
            prediction_window,
            use_heart_rate,
            False,
            use_static_features,
            start_time,
        )
        assert len(dynamic_runs) > 0 and len(no_dynamic_runs) > 0

        dynamic_mean_dict, dynamic_std_dict = get_metrics(dynamic_runs)
        no_dynamic_mean_dict, no_dynamic_std_dict = get_metrics(no_dynamic_runs)

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
        )

    fig.update_layout(
        title_text=f"Activity Ablation | Dataset {datset_to_name[dataset]} | Prediction Windows: {prediction_window}",
        height=n_models * 400,
        width=1200,
        template="plotly_white",
    )

    fig.update_xaxes(title_text="Lookback Window")
    fig.update_yaxes(title_text="Metric Value")

    fig.show()


def visualize_look_back_window_difference(
    dataset: str,
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    use_dynamic_features: bool,
    use_static_features: bool,
    start_time: str = "2025-6-05",
):
    runs = get_runs(
        dataset,
        models,
        look_back_window,
        prediction_window,
        use_heart_rate,
        use_dynamic_features,
        use_static_features,
        start_time,
    )

    mean_dict, std_dict = get_metrics(runs)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[metric_to_name[m] for m in test_metrics],
        shared_xaxes=True,
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )
    for m, model in enumerate(models):
        add_model_mean_std_to_fig(
            model,
            model_to_name[model],
            model_colors[m],
            mean_dict,
            std_dict,
            fig,
            dataset,
        )

    activity_string = "Activity" if use_dynamic_features else "No Activity"

    fig.update_layout(
        title_text=f"Dataset {datset_to_name[dataset]} | Prediction Window {prediction_window[0]} | {activity_string}",
        # height=600,
        # width=800,
        template="plotly_white",
    )

    fig.update_xaxes(title_text="Lookback Window")
    fig.update_yaxes(title_text="Metric Value")

    fig.show()


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
    dataset: str,
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    use_dynamic_features: bool,
    use_static_features: bool,
    start_time: str = "2025-5-25",
):
    print(
        f"Plotting tables for \n Dataset: {dataset} \n Use Heartrate: {use_heart_rate} \n Dynamic Features: {use_dynamic_features}  \n Lookback Windows: {look_back_window} \n Prediction Windows: {prediction_window}"
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
        title_text=f"Model Performance Metrics for {datset_to_name[dataset]} \n"
        f"Lookback: {look_back_window}, Prediction: {prediction_window}, \n"
        f"HR: {use_heart_rate}, Dynamic Features: {use_dynamic_features}",
        title_x=0.5,
        height=n_combinations * 500,
        width=800 + len(df) * 100,
    )

    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WandB Results")

    parser.add_argument(
        "--type",
        type=str,
        choices=["table", "viz", "activity_ablation"],
        required=True,
        default="table",
        help="Plot type. Must be either 'table', 'viz' or 'activity_ablation' .",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "dalia",
            "ieee",
            "wildppg",
            "chapman",
            "ucihar",
            "usc",
            "capture24",
        ],
        required=True,
        default="dalia",
        help="Dataset must be one of the following: dalia, ieee, wildppg, chapman, ucihar, usc, capture24",
    )

    def list_of_ints(arg):
        return [int(i) for i in arg.split(",")]

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

    def list_of_strings(arg):
        return arg.split(",")

    parser.add_argument(
        "--models",
        required=False,
        default=None,
        type=list_of_strings,
        help="Pass in the models you want to visualize the prediction and look back window. Must be separated by commas , without spaces between the model names! (Correct Example: timesnet,elastst | Wrong Example: gpt4ts, timellm )",
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
        )
    elif args.type == "activity_ablation":
        dynamic_feature_ablation(
            args.dataset,
            args.models,
            args.look_back_window,
            args.prediction_window,
            args.use_heart_rate,
            args.use_static_features,
        )
