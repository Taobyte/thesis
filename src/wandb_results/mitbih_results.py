import pandas as pd

from collections import defaultdict

from src.wandb_results.utils import get_runs, get_metrics
from src.constants import MODELS, model_to_name


def plot_mitbih_metric_table(
    models: list[str] = MODELS, start_time: str = "2025-09-01"
):
    dataset = "lmitbih"
    metrics = ["MAE", "RMSE", "MAPE"]
    experiment_name = "lbw_mitbih"
    look_back_window = [10, 20, 30, 40, 50, 60, 70, 80]
    prediction_window = [30]
    pw = prediction_window[0]

    runs = get_runs(
        dataset,
        look_back_window,
        prediction_window,
        models,
        start_time=start_time,
        experiment_name=experiment_name,
    )

    _, mean_metrics, std_metrics = get_metrics(runs, metrics_to_keep=metrics)

    cols = defaultdict(list)
    for model in models:
        lbw = list(mean_metrics[model].keys())[0]
        for metric in metrics:
            cols[model_to_name[model]].append(mean_metrics[model][lbw][pw][metric])

    df = pd.DataFrame.from_dict(cols)
    df.columns.name = "Model"
    df.index = metrics
    df.index.name = "Metric"

    latex_str = df.to_latex(
        index=False,
        escape=False,
        header=True,
        # column_format=column_format,
        bold_rows=False,
        float_format="%.3f",
    )
    print(latex_str)
