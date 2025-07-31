import pandas as pd
import numpy as np
import wandb
import plotly.graph_objects as go


from plotly.subplots import make_subplots
from tqdm import tqdm
from itertools import product

from src.wandb_results.utils import get_metrics
from utils import create_group_run_name
from src.constants import (
    model_to_name,
    metric_to_name,
    dataset_to_name,
)


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
    start_time: str = "2025-5-25",
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
