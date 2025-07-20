import argparse
import numpy as np
import plotly.graph_objects as go

from statsmodels.tsa.stattools import grangercausalitytests
from tqdm import tqdm
from plotly.subplots import make_subplots
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import kpss, adfuller, acf

from src.constants import (
    dataset_to_name,
)
from src.utils import (
    compute_square_window,
    compute_input_channel_dims,
    get_optuna_name,
)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("optuna_name", get_optuna_name)
    OmegaConf.register_new_resolver(
        "compute_input_channel_dims", compute_input_channel_dims
    )

    parser = argparse.ArgumentParser(description="WandB Results")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wildppg", "dalia", "ieee", "mhc6mwt", "usc"],
        required=True,
        default="ieee",
        help="Dataset to plot. Must be 'ieee', 'dalia', 'wildppg' or 'mhc6mwt' ",
    )

    parser.add_argument(
        "--type",
        choices=[
            "granger",
            "pearson",
            "infos",
            "mutual",
            "test",
            "scatter",
            "acf",
            "difference",
        ],
        required=True,
        help="checks granger causality for the timeseries in the dataset",
    )

    args = parser.parse_args()

    with initialize(version_base=None, config_path="../../config/"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"dataset={args.dataset}",
                "folds=all",
                "use_dynamic_features=True",
            ],
        )

    datamodule = instantiate(cfg.dataset.datamodule)

    datamodule.setup("fit")
    # datamodule.setup("test")
    dataset = datamodule.train_dataset.data

    if args.type == "infos":

        def get_channel_stats(dataset):
            mins = [np.min(s) for s in dataset]
            maxs = [np.max(s) for s in dataset]
            means = [np.mean(s) for s in dataset]
            median = [np.median(s) for s in dataset]
            stds = [np.std(s) for s in dataset]
            return {
                "mean": means,
                "median": median,
                "std": stds,
                "min": mins,
                "max": maxs,
            }

        # Stats for HR and Activity
        hr_stats = get_channel_stats([s[:, 0] for s in dataset])
        act_stats = get_channel_stats([s[:, 1] for s in dataset])

        # Flatten all HR and ACT values across the dataset
        all_hr_values = np.concatenate([s[:, 0] for s in dataset])
        all_act_values = np.concatenate([s[:, 1] for s in dataset])

        # Compute quantiles
        quantiles_hr = np.percentile(all_hr_values, [0, 25, 50, 75, 100])
        quantiles_act = np.percentile(all_act_values, [0, 25, 50, 75, 100])

        stats_names = list(hr_stats.keys())

        fig = make_subplots(
            rows=2,
            cols=2,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Heart Rate Statistics", "Activity Statistics"),
        )

        for stat_name in stats_names:
            fig.add_trace(
                go.Scatter(
                    x=[stat_name] * len(hr_stats[stat_name]),
                    y=hr_stats[stat_name],
                    mode="markers",
                    name=f"HR {stat_name}",
                    marker=dict(symbol="circle", size=6, color="blue"),
                    showlegend=False,  # Legend redundant here
                ),
                row=1,
                col=1,
            )

        for stat_name in stats_names:
            fig.add_trace(
                go.Scatter(
                    x=[stat_name] * len(act_stats[stat_name]),
                    y=act_stats[stat_name],
                    mode="markers",
                    name=f"ACT {stat_name}",
                    marker=dict(symbol="square", size=6, color="orange"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        fig.add_trace(go.Box(y=all_hr_values, name="Heart Rate Box Plot"), row=1, col=2)
        fig.add_trace(go.Box(y=all_act_values, name="Activity Box Plot"), row=2, col=2)

        # Update layout
        fig.update_layout(
            height=1200,
            width=1200,
            title="Channel Statistics (Heart Rate & Activity)",
            xaxis=dict(type="category", title="Statistic"),
            xaxis2=dict(type="category", title="Statistic"),
            yaxis=dict(title="Heart Rate"),
            yaxis2=dict(title="Activity Level"),
        )

        fig.show()

        lengths = [len(s) for s in dataset]
        print(f"Total length of dataset {args.dataset} is {sum(lengths)}")
        print("Heart Rate Quantiles (0%, 25%, 50%, 75%, 100%):", quantiles_hr)
        print("Activity Quantiles (0%, 25%, 50%, 75%, 100%):", quantiles_act)

    elif args.type == "granger":
        lags = [1, 3, 5, 10, 20, 30]
        print(
            f"Computing Granger Causality for dataset {dataset_to_name[datamodule.name]} with lags {lags}"
        )

        fig = make_subplots(
            rows=len(dataset),
            cols=1,
            column_titles=[f"Series {i + 1}" for i in range(len(dataset))],
            shared_xaxes=False,
            vertical_spacing=0.05,
        )
        mean_p = []
        for i, series in tqdm(enumerate(dataset)):
            p_values = []
            for lag in lags:
                gc_res = grangercausalitytests(series, [lag], verbose=False)
                p_value = gc_res[lag][0]["ssr_ftest"][1]
                p_values.append(round(p_value, 3))
            fig.add_trace(
                go.Scatter(
                    x=lags,
                    y=p_values,
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )
            fig.update_xaxes(
                title_text="Lags",
                tickmode="array",
                tickvals=lags,
                row=i + 1,
                col=1,
            )
            mean_p.append(np.mean(p_values))

        fig.update_layout(
            title={
                "text": f"<b>Granger Causality P-Values for {dataset_to_name[datamodule.name]}</b>",
                "x": 0.5,
                "xanchor": "center",
                "font": dict(size=40, family="Arial", color="black"),
            },
        )

        mean_means = np.mean(mean_p)
        std_means = np.std(mean_p)
        print(f"Mean {mean_means} and std: {std_means}")

        # fig.show()

    elif args.type == "pearson":

        def max_pearson_corr(x, y):
            x = (x - np.mean(x)) / (np.std(x) * len(x))
            y = (y - np.mean(y)) / np.std(y)
            cors = np.abs(np.correlate(x, y, mode="full"))
            max_cor = np.max(cors)
            best_lag = -len(x) + np.argmax(cors)
            return max_cor, best_lag

        pearsons = []
        best_lags = []
        for i, series in tqdm(enumerate(dataset)):
            print(f"Processing series {i + 1}")
            heartrate = series[:, 0]
            activity = series[:, 1]
            time = list(range(len(heartrate)))
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=heartrate,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=activity,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
            max_corr, best_lag = max_pearson_corr(heartrate, activity)
            print(f"Maximum correlation {max_corr} for lag {best_lag}")
            print(f"Finished processing series {i}")
            # fig.show()
            pearsons.append(max_corr)
            best_lags.append(best_lag)
        mean_cor = np.mean(pearsons)
        std_cor = np.std(pearsons)
        median_lag = np.median(best_lags)
        print(f"Max Abs Corr: {mean_cor} and std {std_cor}")
        print(f"Median lag {median_lag}")

    elif args.type == "mutual":
        # TODO: use Mutual Information to get a score for how important the physical activity is
        mis = []
        for i, series in enumerate(dataset):
            print(f"Processing series {i}")
            heartrate = series[:, 0]
            activity = series[:, 1].reshape(-1, 1)
            mi = mutual_info_regression(activity, heartrate)
            mis.append(mi[0])
            print(f"Mutual Information Regression Value {mi[0]}")

        print(f"Mean {np.mean(mis)} | Std {np.std(mis)}")

    elif args.type == "test":
        # Augmented Dicky-Fuller Test
        for i, series in enumerate(dataset):
            print(f"Processing series {i}")
            heartrate = series[:, 0]
            activity = series[:, 1].reshape(-1, 1)

            ad_hr = adfuller(heartrate)
            print(f"adfuller heartrate: {ad_hr[1]}")
            ad_act = adfuller(activity)
            print(f"adfuller activity: {ad_act[1]}")

            kpps_c_hr = kpss(heartrate, regression="c")
            print(f"KPPS heartrate constant: {kpps_c_hr[1]}")
            kpps_ct_hr = kpss(heartrate, regression="ct")
            print(f"KPPS heartrate linear: {kpps_ct_hr[1]}")

            kpps_c_act = kpss(activity, regression="c")
            print(f"KPPS activity constant: {kpps_c_act[1]}")
            kpps_ct_act = kpss(activity, regression="ct")
            print(f"KPPS activity linear: {kpps_ct_act[1]}")

    elif args.type == "scatter":
        lags = 12
        fig = make_subplots(
            rows=len(dataset),
            cols=lags,
            column_titles=[f"Lag {i}" for i in range(lags)],
        )
        for i, series in enumerate(dataset):
            heartrate = series[:, 0]
            activity = series[:, 1]
            for lag in range(lags):
                fig.add_trace(
                    go.Scatter(
                        x=activity[: len(heartrate) - lag],
                        y=heartrate[lag:],
                        mode="markers",
                    ),
                    row=i + 1,
                    col=lag + 1,
                )

        fig.update_layout(
            # title=f"Scatter Plots plotting Activity against Heartrate for {dataset}",
            height=200 * len(dataset),
            width=200 * lags,
            showlegend=False,
        )

        fig.show()

    elif args.type == "acf":
        lags = 50
        fig = make_subplots(rows=len(dataset), cols=1)
        for i, series in enumerate(dataset):
            heartrate = series[:, 0]
            autocorr = acf(heartrate, nlags=lags, fft=True)

            fig.add_trace(
                go.Scatter(
                    x=list(range(lags + 1)),
                    y=autocorr,
                    mode="markers+lines",
                    line=dict(width=2),
                ),
                row=i + 1,
                col=1,
            )

        fig.show()

    elif args.type == "difference":
        fig = make_subplots(rows=len(dataset) * 2, cols=3)
        # dataset = [dataset[0]]
        for i, series in enumerate(dataset):
            heartrate = series[:, 0]
            activity = series[:, 1]
            length = len(heartrate)
            diff = np.diff(heartrate, n=1)
            diffdiff = np.diff(heartrate, n=2)
            fig.add_trace(
                go.Scatter(
                    x=list(range(length)),
                    y=heartrate,
                    mode="lines",
                    # line=dict(width=2),
                ),
                row=2 * i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(length - 1)),
                    y=diff,
                    mode="lines",
                    # line=dict(width=2),
                ),
                row=2 * i + 1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(length - 2)),
                    y=diffdiff,
                    mode="lines",
                    # line=dict(width=2),
                ),
                row=2 * i + 1,
                col=3,
            )

            for j in range(3):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(length - j)),
                        y=activity[j:],
                        mode="lines",
                        # line=dict(width=2),
                    ),
                    row=2 * i + 2,
                    col=j + 1,
                )

        fig.show()
