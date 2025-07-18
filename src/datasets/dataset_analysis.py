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
from statsmodels.tsa.stattools import kpss, adfuller

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
        choices=["wildppg", "dalia", "ieee", "mhc6mwt"],
        required=True,
        default="ieee",
        help="Dataset to plot. Must be 'ieee', 'dalia', 'wildppg' or 'mhc6mwt' ",
    )

    parser.add_argument(
        "--type",
        choices=["granger", "pearson", "infos", "mutual", "test"],
        required=True,
        help="checks granger causality for the timeseries in the dataset",
    )

    args = parser.parse_args()

    with initialize(version_base=None, config_path="../../config/"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"dataset={args.dataset}",
                "experiment=all",
                "use_dynamic_features=True",
            ],
        )

    datamodule = instantiate(cfg.dataset.datamodule)

    datamodule.setup("fit")
    # datamodule.setup("test")
    dataset = datamodule.train_dataset.data
    if args.type == "infos":
        length = sum([len(s) for s in dataset])
        print(f"Total length of dataset {args.dataset} is {length}")
        print([len(s) for s in dataset])
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
        # TODO: use distance correlation package 'dcor' to analyze the dataset further
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
