import plotly.graph_objects as go
import plotly.io as pio

from plotly.subplots import make_subplots
from matplotlib import colors
from tqdm import tqdm

from src.wandb_results.utils import get_metrics, get_runs
from src.constants import (
    test_metrics,
    dataset_to_name,
    metric_to_name,
    model_to_name,
    model_colors,
)


def visualize_look_back_window_difference(
    datasets: list[str],
    models: list[str],
    look_back_window: list[int],
    prediction_window: list[int],
    use_heart_rate: bool,
    use_dynamic_features: bool,
    use_static_features: bool,
    normalization: str,
    start_time: str = "2025-6-05",
    save_html: bool = False,
):
    num_datasets = len(datasets)
    num_preds = len(prediction_window)
    n_metrics = 4
    offset = num_preds + 1

    readable_metric_names = [metric_to_name[m] for m in test_metrics]
    block_titles = n_metrics * [""] + (num_preds * readable_metric_names)
    subplot_titles = num_datasets * block_titles

    mse_upper = {"dalia": 10, "wildppg": 200, "ieee": 100}
    mae_upper = {"dalia": 5, "wildppg": 20, "ieee": 10}

    fig = make_subplots(
        rows=num_datasets * offset,
        cols=n_metrics,
        subplot_titles=subplot_titles,
        # shared_xaxes=True,
        # horizontal_spacing=0.1,
        # vertical_spacing=0.1,
    )
    for b, dataset in tqdm(enumerate(datasets), total=num_datasets):
        runs = get_runs(
            dataset,
            models,
            look_back_window,
            prediction_window,
            use_heart_rate,
            use_dynamic_features,
            use_static_features,
            normalization,
            start_time,
        )

        dataset_name = dataset_to_name[dataset]
        y_axis_ranges = {
            test_metrics[0]: [0, mse_upper[dataset]],
            test_metrics[1]: [0, mae_upper[dataset]],
            test_metrics[2]: [-1, 1],
            test_metrics[3]: [0, 1],
        }

        mean_dict, std_dict = get_metrics(runs)

        for i, pw in enumerate(prediction_window):
            for j, metric in enumerate(test_metrics):
                for m, model in enumerate(models):
                    row = b * offset + i + 2
                    col = j + 1

                    model_name = model_to_name[model]

                    look_back_windows = sorted(list(mean_dict[model].keys()), key=int)
                    x = [int(lbw) for lbw in look_back_windows]

                    means = [
                        mean_dict[model][lbw][str(pw)][metric]
                        for lbw in look_back_windows
                    ]
                    stds = [
                        std_dict[model][lbw][str(pw)][metric]
                        for lbw in look_back_windows
                    ]
                    upper = [m + s for m, s in zip(means, stds)]
                    lower = [m - s for m, s in zip(means, stds)]

                    color = model_colors[m]

                    # Mean line
                    fig.add_trace(
                        go.Scatter(
                            x=x,
                            y=means,
                            mode="lines+markers",
                            name=f"{dataset_name} {model_name}",
                            line=dict(color=color),
                            showlegend=(j == 0)
                            and (row in [2 + k * offset for k in range(num_datasets)]),
                            legendgroup=f"{dataset_name} {model_name}",
                            # legendgrouptitle_text=dataset_name,
                        ),
                        row=row,
                        col=col,
                    )

                    # Std deviation band (fill between)
                    fig.add_trace(
                        go.Scatter(
                            x=x + x[::-1],
                            y=upper + lower[::-1],
                            fill="toself",
                            fillcolor=color.replace("1.0", "0.2")
                            if "rgba" in color
                            else f"rgba({','.join(str(int(c * 255)) for c in colors.to_rgb(color))},0.2)",
                            line=dict(color="rgba(255,255,255,0)"),
                            hoverinfo="skip",
                            showlegend=False,
                            name=model_name,
                            legendgroup=model_name,
                        ),
                        row=row,
                        col=col,
                    )
                    # Set y-axis range for this subplot
                    # fig.update_yaxes(range=y_axis_ranges[metric], row=row, col=col)
                    # Set x-axis to look_back_window values
                    fig.update_xaxes(
                        title_text="Lookback Window",
                        tickmode="array",
                        tickvals=look_back_windows,
                        row=row,
                        col=col,
                    )

    num_rows = (num_preds + 1) * num_datasets
    num_cols = n_metrics

    for i, dataset in enumerate(datasets):  # or however many row groups you have
        # add dataset title
        fig.add_annotation(
            text=f"<b>Dataset {dataset_to_name[dataset]}</b>",
            xref="paper",
            yref="paper",
            x=0.5,
            y=(1 - i / num_datasets) - (1 / (2 * num_rows)),
            showarrow=False,
            font=dict(size=30),
            textangle=0,
            xanchor="center",
            yanchor="middle",
        )

    for row in range(num_rows):
        if row not in [i * offset for i in range(0, num_datasets)]:
            pw = prediction_window[row % offset - 1]
            fig.add_annotation(
                text=f"<b>Prediction Window = {pw}</b>",
                xref="paper",
                yref="paper",
                x=-0.05,
                y=(1 - row / num_rows),
                showarrow=False,
                font=dict(size=10),
                textangle=-90,
                xanchor="left",
                yanchor="middle",
            )

    activity_string = "Activity" if use_dynamic_features else "No Activity"
    fig.update_layout(
        title={
            "text": f"<b>Lookback Window Ablations | {activity_string}</b>",
            "x": 0.5,
            "xanchor": "center",
            "font": dict(
                size=40, family="Arial", color="black"
            ),  # bold by default for many fonts
        },
        height=num_rows * 400,
        width=num_cols * 600,
        template="plotly_white",
        # margin=dict(t=100, b=100, l=100, r=100),
    )

    if save_html:
        plot_name = f"{dataset}_{use_heart_rate}_{use_dynamic_features}_{'_'.join(models)}_{'_'.join([str(lbw) for lbw in look_back_window])}_{'_'.join([str(pw) for pw in prediction_window])}"
        pio.write_html(
            fig, file=f"./plots/ablations/look_back/{plot_name}.html", auto_open=True
        )
        print(f"Successfully saved {plot_name}")
    else:
        fig.show()
