from src.wandb_results.utils import get_runs, get_metrics
from src.constants import MODELS


def plot_mitbih_metric_table(start_time: str = "2025-09-01"):
    dataset = "lmitbih"
    experiment_name = "lbw_mitbih"
    look_back_window = [10, 20, 30, 40, 50, 60, 70, 80]
    prediction_window = [30]

    runs = get_runs(
        dataset,
        look_back_window,
        prediction_window,
        MODELS,
        start_time=start_time,
        experiment_name=experiment_name,
    )
