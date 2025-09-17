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
    model_to_abbr,
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
        cols["Dataset"].extend(
            [
                rf"\multirow{{{2 * len(prediction_window)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{dataset_to_name[dataset]}}}}}"
            ]
            + [""] * (len(prediction_window) * 2 - 1)
        )
        cols["Metrics"].extend(["Abs Gain", "Exo Gain"] * len(prediction_window))
        for pw in prediction_window:
            cols["PW"].extend([rf"\multirow{{2}}{{*}}{{{pw}}}", ""])
        deltas = defaultdict(list)
        for model in models:
            for pw in prediction_window:
                l_ex = local_exo_res[model][lbw][pw]
                l_end = local_endo_res[model][lbw][pw]
                best_local = min(l_ex, l_end)
                l_impr = 100 * (l_ex - l_end) / l_end
                g_ex = global_exo_res[model][lbw][pw]
                g_end = global_endo_res[model][lbw][pw]
                g_impr = 100 * (g_ex - g_end) / g_end
                best_global = min(g_ex, g_end)
                exo_gain_delta = g_impr - l_impr
                abs_gain_delta = best_local - best_global
                deltas[model].append(exo_gain_delta)
                cols[model].extend([abs_gain_delta, exo_gain_delta])
        import pdb

        pdb.set_trace()
        df = pd.DataFrame.from_dict(cols)
        df.columns = [model_to_abbr[m] for m in models]
        order = ["Dataset", "PW", "Metric"] + [model_to_abbr[m] for m in models]
        df = df[order]
        latex_str = df.to_latex(
            index=False,
            escape=False,
            header=True,
            column_format="|".join(["c"] * len(df.columns)),
            bold_rows=False,
            float_format="%.3f",
        )
        print(latex_str)
        import pdb

        pdb.set_trace()
        delta_df = pd.DataFrame.from_dict(deltas)
        delta_df.index = prediction_window
        delta_df.columns = [model_to_abbr[m] for m in models]

        z = delta_df.to_numpy(dtype=float)
        cbar_title = "Local − Global exo gain (%)"

        x = list(delta_df.columns)
        y = [str(i) for i in delta_df.index]

        vmax = np.nanpercentile(np.abs(z), 90)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        vmin = -vmax

        # value labels (optional)
        text = np.where(np.isnan(z), "", np.round(z, 2).astype(str))

        fig = go.Figure(
            go.Heatmap(
                z=z,
                x=x,
                y=y,
                colorscale="RdBu",
                zmin=vmin,
                zmax=vmax,
                zmid=0,
                colorbar=dict(title=cbar_title),
                hovertemplate="Model=%{x}<br>Horizon=%{y}<br>ΔΔ=%{z:.2f}%<extra></extra>",
                text=text,
                texttemplate="%{text}",
                textfont={"size": 10},
                showscale=True,
            )
        )
        title = "Exogenous gain: Local − Global (%)"
        fig.update_layout(
            title=title,
            xaxis_title="Model",
            yaxis_title="Horizon (steps)",
            yaxis=dict(
                autorange="reversed"
            ),  # put H=1 at top if your index is ascending
            margin=dict(l=80, r=40, t=60, b=60),
            width=max(720, 60 * len(x)),
            height=max(340, 40 * len(y) + 140),
        )
        fig.show()
