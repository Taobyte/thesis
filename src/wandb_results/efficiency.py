import pandas as pd
import numpy as np
import wandb

from tqdm import tqdm
from collections import defaultdict


def plot_efficiency_table(
    dataset: list[str] = ["dalia"],
    look_back_window: int = 30,
    prediction_window: int = 3,
    models: list[str] = [],
    start_time: str = "2025-09-01",
):
    assert len(dataset) == 1
    experiment_name = "efficiency"

    eff_metrics = [
        "params",
        "peak_vram_gib_train",
        "peak_vram_gib_infer",
        "train_seconds_per_epoch_med",
        "epochs_to_best",
        "time_to_best_min",
        "total_fit_minutes",
        "inference_ms_per_window",
    ]

    conditions = [
        {"config.experiment.experiment_name": {"$in": [experiment_name]}},
        {"config.dataset.name": {"$in": dataset}},
        {"config.look_back_window": {"$in": look_back_window}},
        {"config.prediction_window": {"$in": prediction_window}},
        {"config.model.name": {"$in": models}},
        {"state": "finished"},
        {"created_at": {"$gte": start_time}},
    ]

    filters = {"$and": conditions}

    api = wandb.Api()
    runs = api.runs("c_keusch/thesis", filters=filters)

    metrics = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        )
    )

    model_run_counts = defaultdict(int)
    for run in tqdm(runs):
        config = run.config
        model_name = config["model"]["name"]
        look_back_window = config["look_back_window"]
        prediction_window = config["prediction_window"]
        seed = config["seed"]
        fold = config["folds"]["fold_nr"]
        summary = run.summary._json_dict
        filtered_summary = {k: summary[k] for k in summary if k in eff_metrics}
        metrics[model_name][look_back_window][prediction_window][fold][seed] = (
            filtered_summary
        )

        for k, v in model_run_counts.items():
            print(f"Model {k}: {v}")

    processed_metrics_mean = {}
    processed_metrics_std = {}
    for model, v in metrics.items():
        for lbw, w in v.items():
            for pw, fold_dict in w.items():
                metric_list = defaultdict(list)
                for fold_nr, seed_dict in fold_dict.items():
                    for seed, metric_dict in seed_dict.items():
                        for metric_name, metric_value in metric_dict.items():
                            if metric_name in eff_metrics:
                                if isinstance(metric_value, str):
                                    print(
                                        f"VALUE IS STRING {metric_value} for model {model} lbw {lbw} pw {pw} seed {seed}"
                                    )
                                elif np.isinf(metric_value):
                                    print(
                                        f"VALUE IS INF {metric_value} for model {model} lbw {lbw} pw {pw} seed {seed}"
                                    )
                                elif np.isnan(metric_value):
                                    print(
                                        f"VALUE IS NAN {metric_value} for model {model} lbw {lbw} pw {pw} seed {seed}"
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
                processed_metrics_mean[model] = mean
                processed_metrics_std[model] = std

    df = pd.DataFrame.from_dict(processed_metrics_mean, orient="columns")

    df.index.name = "metric"
    df.columns.name = "model"

    import pdb

    pdb.set_trace()
