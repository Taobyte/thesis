import os
import time
import plotly.io as pio

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
    use_static_features: bool,
    start_time: str = "2025-6-24",
    normalization: str = "global",
    save_html: bool = False,
    window_statistic: str = None,
):
    n_models = len(models)
    current_time = int(time.time())

    for dataset in datasets:
        print(f"Processing Activity Ablation for {dataset_to_name[dataset]}")
        dynamic_runs = get_runs(
            dataset,
            models,
            look_back_window,
            prediction_window,
            use_heart_rate,
            True,
            use_static_features,
            normalization,
            start_time,
            window_statistic=window_statistic,
        )

        no_dynamic_runs = get_runs(
            dataset,
            models,
            look_back_window,
            prediction_window,
            use_heart_rate,
            False,
            use_static_features,
            normalization,
            start_time,
            window_statistic=window_statistic,
        )
        dynamic_mean_dict, dynamic_std_dict = get_metrics(dynamic_runs)
        no_dynamic_mean_dict, no_dynamic_std_dict = get_metrics(no_dynamic_runs)
        for p, pw in enumerate(prediction_window):
            fig = make_subplots(
                rows=n_models,
                cols=4,
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
                )

                for lbw in look_back_window:
                    for metric in test_metrics:
                        p_act = dynamic_mean_dict[model][str(lbw)][str(pw)][metric]
                        p_no_act = no_dynamic_mean_dict[model][str(lbw)][str(pw)][
                            metric
                        ]
                        print(
                            f"Model {model} | lbw {lbw} | pw {pw} | Metric {metric} PI: {round((1 - p_act / p_no_act) * 100, 4)}%"
                        )

            fig.update_layout(
                title_text=f"Activity Ablation | Dataset {dataset_to_name[dataset]} | Prediction Windows {pw} | Window Statistic {window_statistic}",
                height=n_models * 400,
                width=1200,
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
                print(f"Saved successfully: {plot_name}")
            else:
                fig.show()
