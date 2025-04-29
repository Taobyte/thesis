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
}

model_to_name = {
    "timesnet": "TimesNet",
    "gpt4ts": "GPT4TS",
    "adamshyper": "AdaMSHyper",
    "timellm": "TimeLLM",
    "pattn": "PAttn",
    "simpletm": "SimpleTM",
    "elastst": "ElasTST",
}


def create_loss_plots(dataframes, fig):
    def create_mean_std(runs: list[pd.DataFrame]) -> pd.DataFrame:
        values = {"train_loss": [], "val_loss": []}
        for run in runs:
            values["train_loss"].append(run["train_loss_epoch"].values)
            values["val_loss"].append(run["val_loss_epoch"].values)
            assert len(run["train_loss_epoch"].values) == len(
                run["val_loss_epoch"].values
            )

        values = {k: np.vstack(v) for k, v in values.items()}
        df = pd.DataFrame()
        df["train_loss_mean"] = np.mean(values["train_loss"], axis=0)
        df["train_loss_std"] = np.std(values["train_loss"], axis=0)

        df["val_loss_mean"] = np.mean(values["val_loss"], axis=0)
        df["val_loss_std"] = np.std(values["val_loss"], axis=0)

        return df

    for col_idx, (model_name, runs) in enumerate(dataframes.items(), start=1):
        df = create_mean_std(runs)
        epochs = np.arange(1, len(df) + 1)
        showlegend = col_idx == 1
        model_name = model_to_name[model_name]
        print(model_name)
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=df["train_loss_mean"],
                mode="lines",
                name=f"{model_name} (train)",
                line=dict(color="blue"),
                showlegend=showlegend,
            ),
            row=1,
            col=col_idx,
        )
        # Plot train std band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([epochs, epochs[::-1]]),
                y=np.concatenate(
                    [
                        df["train_loss_mean"] + df["train_loss_std"],
                        (df["train_loss_mean"] - df["train_loss_std"])[::-1],
                    ]
                ),
                fill="toself",
                fillcolor="rgba(0, 0, 255, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=showlegend,
            ),
            row=1,
            col=col_idx,
        )

        # Plot val mean
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=df["val_loss_mean"],
                mode="lines",
                name=f"{model_name} (val)",
                line=dict(color="red"),
                showlegend=showlegend,
            ),
            row=1,
            col=col_idx,
        )
        # Plot val std band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([epochs, epochs[::-1]]),
                y=np.concatenate(
                    [
                        df["val_loss_mean"] + df["val_loss_std"],
                        (df["val_loss_mean"] - df["val_loss_std"])[::-1],
                    ]
                ),
                fill="toself",
                fillcolor="rgba(255, 0, 0, 0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=showlegend,
            ),
            row=1,
            col=col_idx,
        )

    """
    # Update y-axis titles
    for i in range(1, 7):
        fig.update_yaxes(title_text="Loss", row=1, col=i)
    """
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
        history = run.scan_history()
        train_loss = []
        val_loss = []
        for row in history:
            if "train_loss_epoch" in row and row["train_loss_epoch"]:
                train_loss.append(row["train_loss_epoch"])
            if "val_loss_epoch" in row and row["val_loss_epoch"]:
                val_loss.append(row["val_loss_epoch"])

        loss_df = pd.DataFrame.from_dict(
            {"train_loss_epoch": train_loss, "val_loss_epoch": val_loss}
        )
        dataframes[model_name].append(loss_df)

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
        subplot_titles=[model_to_name[model_name] for model_name in dataframes.keys()]
        + ["Metrics Table"],
        row_heights=[0.7, 0.3],  # 70% for plots, 30% for table
        vertical_spacing=0.1,
    )

    # Usage
    fig = create_loss_plots(dataframes, fig)

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
    print(df_std)

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

    df = df.round(decimals=3)
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
        row=2,
        col=1,
    )
    # Final layout
    fig.update_layout(
        title="Loss Curves and Final Metrics",
        # height=800,  # Adjust height as needed
        # width=1200,
        # xaxis_title="Step",
        # yaxis_title="Loss",
        # legend_title="Model",
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
