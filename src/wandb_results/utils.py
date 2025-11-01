import os
import json
import yaml
import pandas as pd
import numpy as np
import wandb
import ast

from wandb.apis.public.runs import Run
from tqdm import tqdm
from collections import defaultdict
from typing import Tuple
from pathlib import Path


os.environ.setdefault(
    "WANDB_CACHE_DIR", "C:/Users/cleme/ETH/Master/Thesis/ns-forecast/artifacts"
)


def model_to_lbw(
    dataset: str,
    model: str,
    params_path: str = "C:/Users/cleme/ETH/Master/Thesis/ns-forecast/config/params",
) -> int:
    yaml_file = Path(params_path) / model / dataset / "lbw.yaml"
    if not yaml_file.is_file():
        raise FileNotFoundError(f"Missing file: {yaml_file}")
    with yaml_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return int(data["look_back_window"])


def get_metrics(
    runs: list,
    metrics_to_keep: list[str] = [
        "MSE",
        "MAE",
        "DIRACC",
        "MASE",
        "ND",
        "NRMSE",
        "SMAPE",
        "TMAE",
        "TMSE",
    ],
    artifacts_path: str = "C:/Users/cleme/ETH/Master/Thesis/ns-forecast/artifacts",
) -> Tuple[dict, dict, dict]:
    """
    Preprocess the run metrics for the wandb training runs in the runs list.
    Returns two dataframes, the first storing the mean and the second the standard deviation for
    each metric in the METRICS list defined in src/constants.py

    returns: - raw metric dictionary with keys 'model' -> 'look_back_window' -> 'prediction_window' -> 'fold_nr' -> 'seed'
             - mean dict, taking mean over folds and seeds  keys = 'model' -> 'look_back_window' -> 'prediction_window'
             - std dict, taking std over folds and seeds keys = 'model' -> 'look_back_window' -> 'prediction_window'
    """

    metrics = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
    )
    model_run_counts = defaultdict(int)
    for run in tqdm(runs):
        config = run.config
        model_name = config["model"]["name"]
        dataset_name = config["dataset"]["name"]
        is_global = (
            dataset_name in ["dalia", "wildppg", "ieee"]
        ) or dataset_name == "lmitbih"
        look_back_window = config["look_back_window"]
        prediction_window = config["prediction_window"]
        seed = config["seed"]
        if is_global:
            fold = config["folds"]["fold_nr"]
            summary = run.summary._json_dict
            filtered_summary = {k: summary[k] for k in summary if k in metrics_to_keep}
            metrics[model_name][look_back_window][prediction_window][fold][seed] = (
                filtered_summary
            )
        else:
            raw_artifact = next(
                (a for a in run.logged_artifacts() if "raw_metrics" in a.name), None
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

                json_path = art_dir / "raw_metrics.table.json"
                with open(json_path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                data = obj["data"]
                cols = obj["columns"]
                df = pd.DataFrame(data, columns=cols)
                for fold, row in df.iterrows():
                    d = row.to_dict()
                    metrics[model_name][look_back_window][prediction_window][fold][
                        seed
                    ] = d

        model_run_counts[model_name] += 1

    for k, v in model_run_counts.items():
        print(f"Model {k}: {v}")

    processed_metrics_mean = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    processed_metrics_std = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for model, v in metrics.items():
        for lbw, w in v.items():
            for pw, fold_dict in w.items():
                metric_list = defaultdict(list)
                for fold_nr, seed_dict in fold_dict.items():
                    for seed, metric_dict in seed_dict.items():
                        for metric_name, metric_value in metric_dict.items():
                            if metric_name in metrics_to_keep:
                                if isinstance(metric_value, str):
                                    print(
                                        f"VALUE IS STRING {metric_value} for model {model} lbw {lbw} pw {pw} seed {seed} fold {fold_nr}"
                                    )
                                elif np.isinf(metric_value):
                                    print(
                                        f"VALUE IS INF {metric_value} for model {model} lbw {lbw} pw {pw} seed {seed} fold {fold_nr}"
                                    )
                                elif np.isnan(metric_value):
                                    print(
                                        f"VALUE IS NAN {metric_value} for model {model} lbw {lbw} pw {pw} seed {seed} fold {fold_nr}"
                                    )
                                else:
                                    metric_list[metric_name].append(metric_value)

                mean = {
                    metric_name: float(np.mean(v))
                    for metric_name, v in metric_list.items()
                }
                std = {
                    metric_name: float(np.std(v))
                    for metric_name, v in metric_list.items()
                }
                processed_metrics_mean[model][lbw][pw] = mean
                processed_metrics_std[model][lbw][pw] = std
    return metrics, processed_metrics_mean, processed_metrics_std


def get_runs(
    dataset: str,
    look_back_window: list[int],
    prediction_window: list[int],
    models: list[str],
    start_time: str = "2025-6-12",
    feature: str = "mean",
    local_norm_endo_only: bool = False,
    predictions: bool = False,
) -> list[Run]:
    conditions = [
        {"config.use_prediction_callback": predictions},
        {"config.local_norm_endo_only": local_norm_endo_only},
        {"config.feature.name": {"$in": [feature]}},
        {"config.dataset.name": {"$in": [dataset]}},
        {"config.look_back_window": {"$in": look_back_window}},
        {"config.prediction_window": {"$in": prediction_window}},
        {"config.model.name": {"$in": models}},
        {"state": "finished"},
        {"created_at": {"$gte": start_time}},
    ]

    filters = {"$and": conditions}

    api = wandb.Api()
    runs = api.runs("c_keusch/thesis", filters=filters)
    runs: list[Run] = list(runs)
    print(f"Found {len(runs)} runs.")

    # assert len(runs) > 0, "No runs were found!"
    # assert len(runs) % 3 == 0, "Attention, length of runs is not divisible by 3!"
    return runs


def unflatten(d, sep="."):
    out = {}
    for k, v in d.items():
        parts = k.split(sep)
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        processed_value = v
        if isinstance(v, float):
            processed_value = round(v, ndigits=4)
        elif isinstance(v, str) and v.startswith("[") and v.endswith("]"):
            processed_value = ast.literal_eval(v)
        cur[parts[-1]] = processed_value
    return out


def create_params_file_from_optuna(models: list[str], start_time: str):
    lbw_to_name = {"3": "f", "5": "a", "10": "b", "20": "c", "30": "d", "60": "e"}
    params_dir = Path("C:/Users/cleme/ETH/Master/Thesis/ns-forecast/config/params")

    filters = {
        "$and": [
            {"created_at": {"$gte": start_time}},
        ]
    }
    api = wandb.Api()
    runs = api.runs("c_keusch/thesis", filters=filters)
    print(f"Found {len(runs)} runs.")
    for model in models:
        print(50 * "-")
        print(f"{model}")
        print(50 * "-")
        filtered_runs = [
            run for run in runs if run.name and run.name.startswith(f"optuna_{model}_")
        ]
        print(f"Found {len(filtered_runs)} runs.")

        for run in filtered_runs:
            run_name = run.name
            state = run.state
            crashed = state in {"crashed", "failed", "killed"}
            running = state == "running"
            splitted = run_name.split("_")
            dataset = splitted[3]
            if "lmitbih" in run_name:
                lbw_name = "lbw"
            else:
                lbw_name = lbw_to_name[splitted[-4]]
            summary = run.summary._json_dict
            processed_summary = unflatten(summary)
            if "model" in processed_summary and not crashed and not running:
                filtered = {
                    k: v
                    for k, v in processed_summary.items()
                    if k in ["model", "pl_model", "look_back_window"]
                }
                p = params_dir / model / dataset
                p.mkdir(parents=True, exist_ok=True)
                out_path = (p / lbw_name).with_suffix(".yaml")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write("# @package _global_\n")
                    f.write("\n")
                    yaml.safe_dump(filtered, f, sort_keys=False)

                print(f"Successfully written {model} {dataset} {lbw_name}")

            else:
                if crashed:
                    print(f"Run for {model} {dataset} {lbw_name} did not finish.")
                    print(processed_summary)
                elif running:
                    print(
                        f"This job for {model} {dataset} {lbw_name} is still running."
                    )
