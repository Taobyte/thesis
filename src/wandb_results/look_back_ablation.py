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
    use_std: bool = False,
):
    num_datasets = len(datasets)
    n_metrics = 4

    readable_metric_names = [metric_to_name[m] for m in test_metrics]
    readable_dataset_names = [dataset_to_name[d] for d in datasets]
    subplot_titles = num_datasets * readable_metric_names
    row_titles = readable_dataset_names

    fig = make_subplots(
        rows=num_datasets,
        cols=n_metrics,
        subplot_titles=subplot_titles,
        row_titles=row_titles,
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

        mean_dict, std_dict = get_metrics(runs)

        for i, pw in enumerate(prediction_window):
            for j, metric in enumerate(test_metrics):
                assert set(mean_dict.keys()) == set(models), (
                    f"Models froms runs: {mean_dict.keys()} | Models from cmd {models}"
                )
                for m, model in enumerate(mean_dict.keys()):
                    row = b + 1
                    col = j + 1

                    # mse_upper = {"dalia": 10, "wildppg": 200, "ieee": 100}
                    # mae_upper = {"dalia": 5, "wildppg": 20, "ieee": 10}

                    # y_axis_ranges = {
                    #     test_metrics[0]: [0, mse_upper[dataset]],
                    #     test_metrics[1]: [0, mae_upper[dataset]],
                    #     test_metrics[2]: [-1, 1],
                    #     test_metrics[3]: [0, 1],
                    # }

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
                            showlegend=(j == 0) and row == 1,
                            legendgroup=f"{dataset_name} {model_name}",
                            # legendgrouptitle_text=dataset_name,
                        ),
                        row=row,
                        col=col,
                    )

                    # Std deviation band (fill between)
                    if use_std:
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

    num_rows = num_datasets
    num_cols = n_metrics
    activity_string = "Activity" if use_dynamic_features else "No Activity"
    fig.update_layout(
        #  title={
        #      "text": f"<b>Lookback Window Ablations | {activity_string}</b>",
        #      "x": 0.5,
        #      "xanchor": "center",
        #      "font": dict(
        #          size=40, family="Arial", color="black"
        #      ),  # bold by default for many fonts
        #  },
        # height=num_rows * 200,
        # width=num_cols * 600,
        template="plotly_white",
        # margin=dict(t=100, b=100, l=100, r=100),
    )

    for annotation in fig["layout"]["annotations"]:
        if annotation["text"] in row_titles:
            annotation["x"] = -0.02
            annotation["xanchor"] = "right"  # Align text to the right of x position
            annotation["font"] = dict(size=16, color="black", family="Arial")
            annotation["text"] = f"<b>{annotation['text']}</b>"  # Make text bold

    if save_html:
        plot_name = f"{dataset}_{use_heart_rate}_{use_dynamic_features}_{'_'.join(models)}_{'_'.join([str(lbw) for lbw in look_back_window])}_{'_'.join([str(pw) for pw in prediction_window])}"
        pio.write_html(
            fig, file=f"./plots/ablations/look_back/{plot_name}.html", auto_open=True
        )
        print(f"Successfully saved {plot_name}")
    else:
        fig.show()
