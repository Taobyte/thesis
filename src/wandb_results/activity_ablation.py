import plotly.io as pio
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from src.wandb_results.utils import get_metrics, get_runs
import matplotlib.pyplot as plt


def activity_latex_table(
    datasets: list[str],
    models: list[str],
    look_back_window: list[int] = [30],
    prediction_window: list[int] = [3],
    metric: str = "MSE",
    start_time: str = "2025-8-28",
):
    pass


def visualize_exo_difference(
    datasets: list[str],
    models: list[str],
    look_back_window: list[int] = [30],
    prediction_window: list[int] = [3],
    metric: str = "MSE",
    start_time: str = "2025-8-28",
):
    lbw = look_back_window[0]
    pw = prediction_window[0]

    results = {}  # return the diffs for convenience

    for dataset in datasets:
        print(f"Processing {dataset}...")
        # Fetch metrics for both experiment settings
        runs_exo = get_runs(
            dataset,
            look_back_window,
            prediction_window,
            models,
            experiment_name="endo_exo",
            start_time=start_time,
        )
        exo_mean_metrics, exo_std_metrics = get_metrics(runs_exo)

        runs_endo = get_runs(
            dataset,
            look_back_window,
            prediction_window,
            models,
            experiment_name="endo_only",
            start_time=start_time,
        )
        endo_mean_metrics, endo_std_metrics = get_metrics(runs_endo)

        diffs = []
        labels = []
        missing = []

        for m in models:
            try:
                exo = exo_mean_metrics[m][lbw][pw][metric]
                endo = endo_mean_metrics[m][lbw][pw][metric]
                diffs.append(endo - exo)  # positive => exo better
                labels.append(m)
            except KeyError as e:
                missing.append((m, str(e)))

        # Optional heads-up for missing entries
        if missing:
            print(
                f"[{dataset}] skipped {len(missing)} model(s) due to missing keys:",
                missing,
            )

        # Store for programmatic access
        results[dataset] = dict(zip(labels, diffs))

        # ---- Plot (one figure per dataset) ----
        fig = plt.figure(figsize=(max(6, 1.5 * max(1, len(labels))), 4.5))
        ax = plt.gca()
        x = range(len(labels))

        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.scatter(x, diffs)

        # Optional: annotate values
        for xi, yi in zip(x, diffs):
            ax.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("Model")
        ax.set_ylabel(
            r"$\Delta$NRMSE = \mathrm{NRMSE}(\text{endo\_only}) - \mathrm{NRMSE}(\text{endo{+}exo})$"
        )
        ax.set_title(f"{dataset} · L={lbw}, H={pw}")

        plt.tight_layout()
        plt.show()

    return results
