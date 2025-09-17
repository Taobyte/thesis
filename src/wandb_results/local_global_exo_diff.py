import json
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots
from tqdm import tqdm
from typing import Tuple, Union
from collections import defaultdict
from pathlib import Path

from src.wandb_results.utils import get_runs
from src.constants import (
    MODELS,
    BASELINES,
    DL,
    dataset_to_name,
    model_to_name,
    model_colors,
    dataset_colors,
)


def load(
    dataset: str,
    models: list[str],
    metric: str = "MASE",
    look_back_window: list[int] = [10],
    prediction_window: list[int] = [1, 3, 5, 10],
    start_time: str = "2025-09-16",
    artifacts_path: str = "C:/Users/cleme/ETH/Master/Thesis/ns-forecast/artifacts",
):
    if dataset.startswith("l"):
        table_name = "global_mean_metrics"
    else:
        table_name = "local_results"

    exo_runs = get_runs(
        dataset,
        look_back_window,
        prediction_window,
        models,
        start_time,
        experiment_name="endo_exo",
    )
    endo_runs = get_runs(
        dataset,
        look_back_window,
        prediction_window,
        models,
        start_time,
        experiment_name="endo_only",
    )

    def process_runs(runs):
        metrics = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )

        for run in tqdm(runs):
            config = run.config
            model_name = config["model"]["name"]
            look_back_window = config["look_back_window"]
            prediction_window = config["prediction_window"]
            seed = config["seed"]
            raw_artifact = next(
                (a for a in run.logged_artifacts() if table_name in a.name), None
            )
            if raw_artifact is None:
                print(f"No raw_metrics table for run {run.name}")
                continue
            else:
                art_dir = Path(artifacts_path) / str(raw_artifact.name).replace(
                    ":", "-"
                )

                if not os.path.exists(art_dir):
                    raw_artifact.download()

                json_path = art_dir / f"{table_name}.table.json"
                with open(json_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                data = obj["data"]
                cols = obj["columns"]
                df = pd.DataFrame(data, columns=cols)

                metrics[model_name][look_back_window][prediction_window][seed].append(
                    float(df[metric].values)
                )

        processed_metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for model, lbw_v in metrics.items():
            for lbw, pw_v in lbw_v.items():
                for pw, seed_v in pw_v.items():
                    values = []

                    for seed, v in seed_v.items():
                        values.extend(v)
                    processed_metric[model][lbw][pw] = float(np.mean(values))

        return processed_metric

    return process_runs(exo_runs), process_runs(endo_runs)


def local_global_diff(
    datasets: list[str],
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    start_time: str = "2025-6-05",
    use_std: bool = False,
    metric: str = "MASE",
):
    assert len(look_back_window) == 1
    lbw = look_back_window[0]

    for dataset in datasets:
        local_exo_res, local_endo_res = load(
            "l" + dataset,
            models,
            metric,
            look_back_window,
            prediction_window,
        )

        global_exo_res, global_endo_res = load(
            dataset,
            models,
            metric,
            look_back_window,
            prediction_window,
        )
        cols = defaultdict(list)
        for model in models:
            for pw in prediction_window:
                l_ex = local_exo_res[model][lbw][pw]
                l_end = local_endo_res[model][lbw][pw]
                l_impr = 100 * (l_ex - l_end) / l_end
                g_ex = global_exo_res[model][lbw][pw]
                g_end = global_endo_res[model][lbw][pw]
                g_impr = 100 * (g_ex - g_end) / g_end
                cols[model].extend([l_end, l_ex, l_impr, g_end, g_ex, g_impr])

        df = pd.DataFrame.from_dict(cols)

        latex_str = df.to_latex(
            index=False,
            escape=False,
            header=True,
            column_format="|".join(["c"] * len(df.columns)),
            bold_rows=False,
            float_format="%.3f",
        )
        print(latex_str)
