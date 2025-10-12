import pandas as pd
import numpy as np

from collections import defaultdict

from src.wandb_results.utils import get_metrics, get_runs


def wppg_feature_ablation():
    model_settings = {
        "linear": 20,
        "xgboost": 20,
        "gp": 30,
        "mlp": 30,
        "gpt4ts": 30,
        "nbeatsx": 30,
        "timesnet": 30,
        "simpletm": 30,
    }
    metric = "TMSE"
    dataset = "wildppg"
    features = ["none", "mean", "rms_last2s_rms_jerk_centroid", "catch22"]
    pw = 3
    metrics = defaultdict(lambda: defaultdict(float))

    for model, lbw in model_settings.items():
        for feature in features:
            runs = get_runs(
                dataset, [lbw], [pw], [model], feature=feature, start_time="2025-10-08"
            )
            _, feature_metric, _ = get_metrics(runs, metrics_to_keep=[metric])
            metrics[model][feature] = (
                feature_metric[model][lbw][pw][metric]
                if metric in feature_metric[model][lbw][pw]
                else np.nan
            )

    df = pd.DataFrame.from_dict(metrics)
    print(df)
