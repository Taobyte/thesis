import pandas as pd
import wandb
import argparse
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from datetime import datetime


def process_results(
    dataset: str,
    look_back_window: int,
    prediction_window: int,
    use_heart_rate: bool,
    use_activity_info: bool,
):
    print(
        f"Filtering for \n dataset: {dataset} \n look_back_window: {look_back_window} \n prediction_window: {prediction_window} \n HR: {use_heart_rate} \n activity_info: {use_activity_info}"
    )

    filter = [
        {"config.dataset.name": dataset},
        {"config.look_back_window": look_back_window},
        {"config.prediction_window": prediction_window},
        {"config.use_activity_info": use_activity_info},
        {"config.dataset.datamodule.use_heart_rate": use_heart_rate},
    ]

    api = wandb.Api()
    runs = api.runs("c_keusch/thesis", filters={"$and": filter})

    start_time = datetime(2025, 4, 23)

    import pdb

    time_filtered_runs = [
        run
        for run in runs
        if start_time <= datetime.strptime(run.created_at.split("T")[0], "%Y-%m-%d")
        and run.state == "finished"
    ]

    print(f"Found {len(time_filtered_runs)} number of runs.")

    metrics = {}
    loss = {}

    for run in time_filtered_runs:
        model_name = run.name.split("_")[1]
        if model_name not in metrics:
            metrics[model_name] = []
            loss[model_name] = []
        summary = run.summary._json_dict
        filtered_summary = {
            k: summary[k]
            for k in summary
            if k in ["test_MSE", "test_MAE", "test_cross_correlation"]
        }
        metrics[model_name].append(filtered_summary)
        history = run.history(keys=["train_loss_epoch", "val_loss_epoch"])
        loss[model_name].append(history)

    # TODO: plot training loss
    # TODO: plot val loss
    # TODO: plot table with metrics

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.15,
        specs=[[{"type": "xy"}], [{"type": "table"}]],
    )

    for model_name, runs in loss.items():
        df = runs[0]
        if "train_loss_epoch" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["_step"],
                    y=df["train_loss_epoch"],
                    mode="lines",
                    name=f"{model_name} (train)",
                    line=dict(dash="solid"),
                )
            )
        if "val_loss_epoch" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["_step"],
                    y=df["val_loss_epoch"],
                    mode="lines",
                    name=f"{model_name} (val)",
                    line=dict(dash="dash"),
                )
            )

    processed_metrics = {}
    for k, v in metrics.items():
        processed_metrics[k] = v[0]
    df = pd.DataFrame.from_dict(processed_metrics)
    fig.add_trace(
        go.Table(
            header=dict(
                values=list(df.columns),
                fill_color="paleturquoise",
                align="left",
            ),
            cells=dict(
                values=[df[column].to_list() for column in df.columns],
                fill_color="lavender",
                align="left",
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
