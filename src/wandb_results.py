import pandas as pd
import wandb
import argparse
import plotly.graph_objects as go


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

    print(f"Found {len(runs)} number of runs.")
    print("Model names:")
    for run in runs:
        # print(run.config["model"])
        if run.config["model"] is None:
            print(run)
    metrics = {}
    loss = {}

    for run in runs:
        model_name = run.name.split("_")[1]
        if model_name not in metrics:
            metrics[model_name] = []
            loss[model_name] = []
        summary = run.summary._json_dict
        metrics[model_name].append(summary)
        history = run.history(keys=["train_loss_epoch"])
        loss[model_name].append(history)

    # TODO: plot training loss
    # TODO: plot val loss
    # TODO: plot table with metrics

    import pdb

    pdb.set_trace()

    loss_fig = go.Figure()
    for model_name, runs in loss.items():
        df = runs[0]
        if "train_loss_epoch" in df.columns:
            loss_fig.add_trace(
                go.Scatter(
                    x=df["_step"],
                    y=df["train_loss_epoch"],
                    mode="lines",
                    name=f"{model_name} (train)",
                    line=dict(dash="solid"),
                )
            )
        if "val/loss" in df.columns:
            loss_fig.add_trace(
                go.Scatter(
                    x=df["_step"],
                    y=df["val/loss"],
                    mode="lines",
                    name=f"{model_name} (val)",
                    line=dict(dash="dash"),
                )
            )

    loss_fig.update_layout(
        title="Training and Validation Loss Over Time",
        xaxis_title="Step",
        yaxis_title="Loss",
        legend_title="Model",
    )
    loss_fig.show()


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
