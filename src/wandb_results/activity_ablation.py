import os
import time
import plotly.io as pio
import plotly.graph_objects as go

from tqdm import tqdm
from plotly.subplots import make_subplots
from src.wandb_results.utils import get_metrics, get_runs, add_model_mean_std_to_fig
from src.constants import (
    metric_to_name,
    test_metrics,
    model_to_name,
    model_colors,
    dataset_to_name,
)


def dynamic_feature_ablation(
    datasets: list[str],
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    start_time: str = "2025-6-24",
    normalization: str = "global",
    save_html: bool = False,
    window_statistic: str = None,
    use_std: bool = False,
    table: bool = True,
):
    assert len(models) > 0
    assert len(datasets) > 0
    assert len(look_back_window) > 0
    assert len(prediction_window) > 0

    n_models = len(models)
    current_time = int(time.time())

    for dataset in datasets:
        dynamic_runs = get_runs(
            dataset,
            models,
            look_back_window,
            prediction_window,
            use_heart_rate,
            normalization,
            start_time,
            window_statistic=window_statistic,
            experiment_name="endo_exo",
        )

        no_dynamic_runs = get_runs(
            dataset,
            models,
            look_back_window,
            prediction_window,
            use_heart_rate,
            normalization,
            start_time,
            window_statistic=window_statistic,
            experiment_name="endo_only",
        )
        dynamic_mean_dict, dynamic_std_dict = get_metrics(dynamic_runs)
        no_dynamic_mean_dict, no_dynamic_std_dict = get_metrics(no_dynamic_runs)

        if not table:
            print(
                f"Processing Activity Ablation Visualization for {dataset_to_name[dataset]}"
            )
            for p, pw in enumerate(prediction_window):
                fig = make_subplots(
                    rows=n_models,
                    cols=len(test_metrics),
                    column_titles=[metric_to_name[metric] for metric in test_metrics],
                    row_titles=[model_to_name[model] for model in models],
                    shared_xaxes=False,
                )
                for i, model in tqdm(enumerate(models), total=len(models)):
                    add_model_mean_std_to_fig(
                        model,
                        f"Activity {model_to_name[model]}",
                        model_colors[2],
                        dynamic_mean_dict,
                        dynamic_std_dict,
                        fig,
                        dataset,
                        i + 1,
                        True,
                        row_delta=2 * p,
                        use_std=use_std,
                    )

                    add_model_mean_std_to_fig(
                        model,
                        f"No Activity {model_to_name[model]}",
                        model_colors[0],
                        no_dynamic_mean_dict,
                        no_dynamic_std_dict,
                        fig,
                        dataset,
                        i + 1,
                        True,
                        row_delta=2 * p,
                        use_std=use_std,
                    )

            fig.update_layout(
                title_text=f"Activity Ablation | Dataset {dataset_to_name[dataset]} | Prediction Windows {pw} | Window Statistic {window_statistic}",
                height=n_models * 400,
                width=len(test_metrics) * 300,
                template="plotly_white",
            )

            fig.update_xaxes(title_text="Lookback Window")
            fig.update_yaxes(title_text="Metric Value")

            if save_html:
                plot_name = f"{current_time}_{dataset}_{window_statistic}_lbw_{'_'.join([str(lbw) for lbw in look_back_window])}_pw_{pw}_{'_'.join(models)}"
                base_dir = f"./plots/ablations/{window_statistic}"
                os.makedirs(base_dir, exist_ok=True)
                pio.write_html(
                    fig,
                    file=f"{base_dir}/{plot_name}.html",
                    auto_open=True,
                )
                fig.write_image(
                    f"./plots/ablations/look_back/{plot_name}.pdf",
                    width=1920,  # width in pixels
                    height=1080,
                    scale=2,
                )
                print(f"Saved successfully: {plot_name}")
            else:
                fig.show()

        else:
            print(f"Processing Activity Ablation Table for {dataset_to_name[dataset]}")
            pw = prediction_window[0]

            for metric in test_metrics:
                cols = []
                first_col = []
                for model in models:
                    name = model_to_name[model]
                    first_col.append(f"{name} (Act)")
                    first_col.append(f"{name} (No Act)")
                    first_col.append(f"{name} (Imprv)")
                cols.append(first_col)

                for lbw in look_back_window:
                    cells = []
                    for model in models:
                        dyn_perf = dynamic_mean_dict[model][str(lbw)][str(pw)][metric]
                        no_dyn_perf = no_dynamic_mean_dict[model][str(lbw)][str(pw)][
                            metric
                        ]
                        cells.append(
                            round(
                                float(dyn_perf),
                                4,
                            )
                        )
                        cells.append(
                            round(
                                float(no_dyn_perf),
                                4,
                            )
                        )
                        if metric in ["test_cross_correlation", "test_dir_acc_single"]:
                            improvement = round(
                                float(((dyn_perf - no_dyn_perf) / no_dyn_perf) * 100), 4
                            )
                        else:
                            improvement = round(
                                float(((no_dyn_perf - dyn_perf) / no_dyn_perf) * 100), 4
                            )
                        cells.append(f"{improvement}%")
                    cols.append(cells)
                table_fig = go.Figure(
                    data=[
                        go.Table(
                            header=dict(
                                values=["Model (Metric)"]
                                + [str(lbw) for lbw in look_back_window]
                            ),
                            cells=dict(
                                values=cols,
                                align="center",
                            ),
                        )
                    ]
                )
                table_fig.update_layout(
                    title=dict(
                        text=f"Model Performance Table {metric_to_name[metric]}",
                        x=0.5,
                        xanchor="center",
                        font=dict(size=20, family="Arial", color="black"),
                    )
                )

                table_fig.show()
