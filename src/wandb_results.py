import wandb
import argparse


def process_results(
    dataset: str,
    look_back_window: int,
    prediction_window: int,
    use_heart_rate: bool,
    use_activity_info: bool,
):
    filter = [
        dataset,
        {"config.look_back_window:": look_back_window},
        {"config.prediction_window": prediction_window},
        {"config.dataset.datamodule.use_heart_rate": use_heart_rate},
        {"config.use_activity_info": use_activity_info},
    ]

    api = wandb.Api()
    runs = api.runs("c_keusch/thesis", filters={"tags": {"$and": filter}})

    metrics = {}
    loss = {}

    for run in runs:
        model_name = run.name.split("_")[1]
        if model_name not in metrics:
            metrics["model_name"] = []
            loss["model_name"] = []
        summary = run.summary._json_dict
        # TODO append metrics to metrics dict
        history = run.scan_history()
        loss["model_name"].append(history)

    # TODO: plot training loss
    # TODO: plot val loss
    # TODO: plot table with metrics


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

    process_results(
        args.dataset,
        args.look_back_window,
        args.prediction_window,
        args.use_heart_rate,
        args.use_activity_info,
    )
