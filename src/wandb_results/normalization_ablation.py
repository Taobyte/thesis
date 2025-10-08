import pandas as pd
import numpy as np
import wandb

from typing import List

from src.wandb_results.utils import get_metrics
from src.constants import (
    model_to_abbr,
    dataset_to_name,
    MODELS,
)


def plot_normalization_table(
    datasets: list[str],
    prediction_window: list[int] = [3],
    start_time: str = "2025-10-08",
    models: list[str] = MODELS,
    use_std: bool = False,
    metric: str = "MASE",
):
    assert len(prediction_window) == 1
    pw = prediction_window[0]

    cols: dict[str, list[float]] = dict()

    model_col: List[str] = ["Norm"]
    for model_name in models:
        model_name = model_to_abbr[model_name]
        model_col.append(model_name)

    cols["Models"] = model_col

    experiments = [
        "lnone",
        "local_z",
        "difference",
    ]
    # abbr_exp_names = ["NN", "GZ", "MM", "LZ", "DF"]
    abbr_exp_names = ["GZ", "LZ", "DF"]

    for dataset in datasets:
        print(f"DATASET: {dataset}")
        for local_norm, name in zip(experiments, abbr_exp_names):
            col: List[str] = [name]
            for model_name in models:
                conditions = [
                    {"state": "finished"},
                    {"created_at": {"$gte": start_time}},
                    {"tags": dataset},  # dataset tag must be present
                    {
                        "config.normalization": "global",
                        "config.local_norm": local_norm,
                    },
                    {"tags": model_name},
                ]
                filters = {"$and": conditions}

                api = wandb.Api()
                runs = api.runs("c_keusch/thesis", filters=filters)
                print(f"{model_name} length of runs {len(runs)}")

                if len(runs) > 0:
                    _, mean_dict, std_dict = get_metrics(runs)

                    if list(mean_dict[model_name].keys()):
                        lbw = list(mean_dict[model_name].keys())[0]
                    else:
                        lbw = None

                    if lbw and metric in mean_dict[model_name][lbw][pw]:
                        mean = mean_dict[model_name][lbw][pw][metric]
                        std = std_dict[model_name][lbw][pw][metric]
                    else:
                        print(
                            f"Metric {metric} not in mean_dict[{model_name}][{lbw}][{pw}]"
                        )
                        print(
                            f"Setting mean = NaN for {model_name} {local_norm} {dataset}"
                        )
                        mean = np.nan

                entry = f"{mean:.2f}"
                if use_std:
                    entry += f" $\pm$ {std:.2f}"
                col.append(entry)
            cols[f"{dataset} {name}"] = col

    df = pd.DataFrame(cols)
    df = df.T
    df = df.reset_index(drop=True)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)

    n = len(experiments)
    dataset_col = []
    for dataset in datasets:
        dataset_col.extend(
            [
                rf"\multirow{{{n}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{dataset_to_name[dataset]}}}}}"
            ]
            + [""] * (n - 1)
        )
    df.insert(0, "Dataset", dataset_col)

    latex_str = df.to_latex(
        index=False,
        escape=False,
        header=True,
        column_format="|".join(["c"] * len(df.columns)),
        bold_rows=False,
    )
    print(latex_str)
