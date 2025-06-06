import pandas as pd
import numpy as np
import wandb
import argparse
import plotly.graph_objects as go

from matplotlib import colors
from plotly.subplots import make_subplots
from collections import defaultdict
from itertools import product
from typing import Tuple

from utils import create_group_run_name

test_metrics = ["test_MSE", "test_MAE", "test_cross_correlation", "test_dir_acc_single"]

model_colors = ["blue", "green", "orange", "red", "purple", "pink"]

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
    start_time: str = "2025-6-05",
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
    y_axis_ranges = {
        test_metrics[0]: [0, 20],
        test_metrics[1]: [0, 3],
        test_metrics[2]: [-1, 1],
        test_metrics[3]: [0, 1],
    }

    for pw in prediction_windows:
        for i, metric in enumerate(test_metrics):
            means = [mean_dict[model][lbw][pw][metric] for lbw in look_back_windows]
            stds = [std_dict[model][lbw][pw][metric] for lbw in look_back_windows]

            upper = [m + s for m, s in zip(means, stds)]
            lower = [m - s for m, s in zip(means, stds)]

            row, col = divmod(i, 2)
            row += 1
            col += 1
            color = model_color

            # Mean line
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=means,
                    mode="lines+markers",
                    name=model_name,
                    line=dict(color=color),
                    showlegend=(i == 0),
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

        fig.update_layout(
            title_text=f"{model_to_name[model]} - Prediction Window {pw}",
            height=600,
            width=800,
            template="plotly_white",
        )

        fig.update_xaxes(title_text="Lookback Window")
        fig.update_yaxes(title_text="Metric Value")


def dynamic_feature_ablation(
    dataset: str,
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    use_static_features: bool,
    start_time: str = "2025-6-05",
):
    assert len(models) == 1

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

    model = models[0]

    dynamic_mean_dict, dynamic_std_dict = get_metrics(dynamic_runs)
    no_dynamic_mean_dict, no_dynamic_std_dict = get_metrics(no_dynamic_runs)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[metric_to_name[m] for m in test_metrics],
        shared_xaxes=True,
        horizontal_spacing=0.1,
        vertical_spacing=0.15,
    )

    add_model_mean_std_to_fig(
        model,
        "Activity",
        model_colors[2],
        dynamic_mean_dict,
        dynamic_std_dict,
        fig,
    )

    add_model_mean_std_to_fig(
        model,
        "No Activity",
        model_colors[0],
        no_dynamic_mean_dict,
        no_dynamic_std_dict,
        fig,
    )

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
            model, model_to_name[model], model_colors[m], mean_dict, std_dict, fig
        )

    fig.show()


def visualize_metric_table(
    df: pd.DataFrame,
    df_std: pd.DataFrame,
    dataset: str,
    look_back_window: int,
    prediction_window: int,
    use_heart_rate: bool,
    use_dynamic_features: bool,
    use_static_features: bool,
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

    fig = make_subplots(
        rows=1,
        cols=1,  # You can adjust this if you add loss curves later
        specs=[[{"type": "table"}]],  # Specify the type of subplot for the table
    )
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
        row=1,
        col=1,
    )
    # Final layout
    fig.update_layout(
        title_text=f"Model Performance Metrics for {datset_to_name[dataset]} \n"
        f"(Lookback: {look_back_window}, Prediction: {prediction_window}, \n"
        f"HR: {use_heart_rate}, Dynamic Features: {use_dynamic_features}, Static Features: {use_static_features})",
        title_x=0.5,  # Center the title
        height=800,  # Adjust height dynamically based on number of rows
        width=800 + len(df) * 100,  # Adjust width as needed
    )

    fig.show()


def process_results(
    dataset: str,
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    use_dynamic_features: bool,
    use_static_features: bool,
    start_time: str = "2025-5-25",
    plot_type: str = "table",
):
    print(
        f"Filtering for \n dataset: {dataset} \n look_back_window: {look_back_window} \n prediction_window: {prediction_window} \n HR: {use_heart_rate} \n Dynamic Features: {use_dynamic_features} \n Static Features: {use_static_features}"
    )

    assert len(look_back_window) == 1 and len(prediction_window) == 1, (
        f"Look back window and prediction window lists must be of size 1, but are {len(look_back_window)} and {len(prediction_window)}"
    )

    group_name, run_name, tags = create_group_run_name(
        dataset,
        "",
        use_heart_rate,
        look_back_window[0],
        prediction_window[0],
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

    df, df_std = get_metrics(runs)

    if plot_type == "table":
        visualize_metric_table(
            df,
            df_std,
            dataset,
            look_back_window,
            prediction_window,
            use_heart_rate,
            use_dynamic_features,
            use_static_features,
        )
    elif plot_type == "lbw_viz":
        visualize_look_back_window_difference()


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
        process_results(
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
