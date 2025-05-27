import pandas as pd
import numpy as np
import wandb
import argparse
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from collections import defaultdict

from utils import create_group_run_name


metric_to_name = {
    "test_MSE": "Mean Squared Error",
    "test_MAE": "Mean Absolute Deviation",
    "test_cross_correlation": "Cross Correlation",
    "test_dir_acc": "Directional Accuracy",
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


def process_results(
    dataset: str,
    look_back_window: int,
    prediction_window: int,
    use_heart_rate: bool,
    use_dynamic_features: bool,
    use_static_features: bool,
    start_time: str = "2025-5-25",
):
    print(
        f"Filtering for \n dataset: {dataset} \n look_back_window: {look_back_window} \n prediction_window: {prediction_window} \n HR: {use_heart_rate} \n Dynamic Features: {use_dynamic_features} \n Static Features: {use_static_features}"
    )

    group_name, run_name, tags = create_group_run_name(
        dataset,
        "",
        use_heart_rate,
        look_back_window,
        prediction_window,
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

    metrics = defaultdict(list)

    for run in runs:
        model_name = run.name.split("_")[3]
        summary = run.summary._json_dict
        filtered_summary = {
            k: summary[k]
            for k in summary
            if k in ["test_MSE", "test_MAE", "test_cross_correlation", "test_dir_acc"]
        }
        metrics[model_name].append(filtered_summary)

    # fig = create_loss_plots(dataframes, fig)
    # plot metric table
    processed_metrics_mean = {}
    processed_metrics_std = {}
    for k, v in metrics.items():
        metric_list = defaultdict(list)
        for metric_dict in v:
            for metric_name, metric_value in metric_dict.items():
                metric_list[metric_name].append(metric_value)

        mean = {metric_name: np.mean(v) for metric_name, v in metric_list.items()}
        std = {metric_name: np.std(v) for metric_name, v in metric_list.items()}
        processed_metrics_mean[k] = mean
        processed_metrics_std[k] = std
    df = pd.DataFrame.from_dict(processed_metrics_mean)
    df_std = pd.DataFrame.from_dict(processed_metrics_std)
    order = ["linear", "kalmanfilter", "gp", "xgboost", "simpletm", "pattn"]
    df = df[order]
    df_std = df_std[order]

    font_weights = []
    for i in range(len(df)):
        row = df.iloc[i, :]
        bold_idx = (
            np.argmin(row) if i < 2 else np.argmax(row)
        )  # i == 2 is cross correlation
        weights = ["normal"] * len(row)
        weights[bold_idx] = "bold"
        weights.insert(0, "normal")
        font_weights.append(weights)

    font_weights = [weight for row in font_weights for weight in row]
    import pdb

    pdb.set_trace()

    df = df.round(decimals=3)
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
                        f"{mean:.3f} ± {std:.3f}"
                        for mean, std in zip(df[column], df_std[column])
                    ]
                    for column in df.columns
                ],
                fill_color="lavender",
                align="left",
                font=dict(
                    size=11,
                    color="black",
                    weight=font_weights,
                ),
            ),
        ),
        row=1,
        col=1,
    )
    # Final layout
    fig.update_layout(
        title_text=f"Model Performance Metrics for {dataset.capitalize()} \n"
        f"(Lookback: {look_back_window}, Prediction: {prediction_window}, \n"
        f"HR: {use_heart_rate}, Dynamic Features: {use_dynamic_features}, Static Features: {use_static_features})",
        title_x=0.5,  # Center the title
        height=800,  # Adjust height dynamically based on number of rows
        width=800 + len(df) * 100,  # Adjust width as needed
    )

    fig.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WandB Results")

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

    parser.add_argument(
        "--look_back_window",
        type=int,
        required=True,
        default=5,
        help="Lookback window size",
    )

    parser.add_argument(
        "--prediction_window",
        type=int,
        required=True,
        default=3,
        help="Prediction window size",
    )

    parser.add_argument(
        "--use_heart_rate",
        required=False,
        action="store_true",
        help="get runs for heart rate only has an effect for datasets: dalia, ieee & ppg",
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

    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs("c_keusch/thesis")

    process_results(
        args.dataset,
        args.look_back_window,
        args.prediction_window,
        args.use_heart_rate,
        args.use_dynamic_features,
        args.use_static_features,
    )
