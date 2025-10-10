import pandas as pd

from collections import defaultdict

from src.wandb_results.utils import get_metrics, get_runs


def wppg_feature_ablation():
    model_settings = {
        "xgboost": 20,
        "mlp": 30,
        "gpt4ts": 30,
        "nbeatsx": 30,
    }
    metric = "MAE"
    dataset = "wildppg"
    features = ["none", "mean", "rms_last2s_rms_jerk_centroid", "catch22"]
    pw = 3
    metrics = defaultdict(lambda: defaultdict(float))
    import pdb

    for model, lbw in model_settings.items():
        for feature in features:
            runs = get_runs(
                dataset, [lbw], [pw], [model], feature=feature, start_time="2025-10-08"
            )
            _, feature_metric, _ = get_metrics(runs, metrics_to_keep=[metric])
            metrics[model][feature] = feature_metric[model][lbw][pw][metric]

    pdb.set_trace()
