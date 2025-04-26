import pandas as pd
import numpy as np
import wandb
import argparse
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from collections import defaultdict

from utils import create_group_run_name

import pdb

model_to_loss_name = {"gpt4ts": "sMAPE Loss", "TODO": None}

metric_to_name = {
    "test_MSE": "Mean Squared Error",
    "test_MAE": "Mean Absolute Deviation",
    "test_cross_correlation": "Cross Correlation",
}


def create_loss_plots(dataframes, fig):
    # Create subplots with 2 rows (1 for plots, 1 for table) and 6 columns
    def process_column(df: pd.DataFrame):
        df = df[~df.isna()]
        n = len(df)
        x = np.arange(1, n + 1)
        y = df.values
        return x, y

    for col_idx, (model_name, runs) in enumerate(dataframes.items(), start=1):
        df = runs[0]  # Assuming each model has one run

        # Train Loss (left column)
        if "train_loss_epoch" in df.columns:
            x, y = process_column(df["train_loss_epoch"])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"{model_name} (train)",
                    line=dict(color="blue"),
                    showlegend=True,
                ),
                row=1,
                col=col_idx,
            )

        # Val Loss (right column)
        if "val_loss_epoch" in df.columns:
            x, y = process_column(df["val_loss_epoch"])
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=f"{model_name} (val)",
                    line=dict(color="red"),
                    showlegend=True,
                ),
                row=1,
                col=col_idx,
            )

    # Update y-axis titles
    for i in range(1, 7):
        fig.update_yaxes(title_text="Loss", row=1, col=i)

    return fig


def process_results(
    dataset: str,
    look_back_window: int,
    prediction_window: int,
    use_heart_rate: bool,
    use_activity_info: bool,
    start_time: str = "2025-4-25",
):
    print(
        f"Filtering for \n dataset: {dataset} \n look_back_window: {look_back_window} \n prediction_window: {prediction_window} \n HR: {use_heart_rate} \n activity_info: {use_activity_info}"
    )

    group_name, run_name, tags = create_group_run_name(
        dataset,
        "",
        use_heart_rate,
        use_activity_info,
        look_back_window,
        prediction_window,
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

    metrics = defaultdict(list)
    dataframes = defaultdict(list)

    for run in runs:
        model_name = run.name.split("_")[1]
        summary = run.summary._json_dict
        filtered_summary = {
            k: summary[k]
            for k in summary
            if k in ["test_MSE", "test_MAE", "test_cross_correlation"]
        }
        metrics[model_name].append(filtered_summary)
        history = run.history()
        dataframes[model_name].append(history)

    fig = make_subplots(
        rows=2,
        cols=6,
        # First row: 6 individual plots
        specs=[
            [
                {"type": "xy"},
                {"type": "xy"},
                {"type": "xy"},
                {"type": "xy"},
                {"type": "xy"},
                {"type": "xy"},
            ],
            # Second row: single table spanning all columns
            [{"type": "table", "colspan": 6}, None, None, None, None, None],
        ],
        subplot_titles=[model_name for model_name in dataframes.keys()]
        + ["Metrics Table"],
        row_heights=[0.7, 0.3],  # 70% for plots, 30% for table
        vertical_spacing=0.1,
    )

    # Usage
    fig = create_loss_plots(dataframes, fig)

    # plot metric table
    processed_metrics = {}
    for k, v in metrics.items():
        processed_metrics[k] = v[0]
    df = pd.DataFrame.from_dict(processed_metrics)

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

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric"] + list(df.columns),
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[[metric_to_name[i] for i in df.index]]
                + [df[column].tolist() for column in df.columns],
                fill_color="lavender",
                align="left",
                font=dict(
                    size=11,
                    color="black",
                    weight=font_weights,
                ),
            ),
        ),
        row=2,
        col=1,
    )
    # Final layout
    fig.update_layout(
        title="Loss Curves and Final Metrics",
        height=800,  # Adjust height as needed
        xaxis_title="Step",
        yaxis_title="Loss",
        legend_title="Model",
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
        required=True,
        action="store_true",
        help="get runs for heart rate only has an effect for datasets: dalia, ieee & ppg",
    )
    parser.add_argument(
        "--use_activity_info",
        required=False,
        action="store_true",
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
        args.use_activity_info,
    )
