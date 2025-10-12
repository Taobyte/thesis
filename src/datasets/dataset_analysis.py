import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import antropy as ant
import pycatch22
import math
import ruptures as rpt  # our package
import matplotlib.pyplot as plt  # for display purposes

from lightning import LightningDataModule
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import bds
from statsmodels.tsa.stattools import kpss, adfuller, pacf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
from plotly.subplots import make_subplots
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sklearn.feature_selection import mutual_info_regression
from typing import Optional, Tuple, List, Dict, Any
from numpy.typing import NDArray
from collections import defaultdict
from src.normalization import local_z_norm_numpy, min_max_norm_numpy

from src.constants import dataset_to_name

from src.utils import (
    get_optuna_name,
    compute_square_window,
    compute_input_channel_dims,
    get_min,
    resolve_str,
    number_of_exo_vars,
)


OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("optuna_name", get_optuna_name)
OmegaConf.register_new_resolver(
    "compute_input_channel_dims", compute_input_channel_dims
)
OmegaConf.register_new_resolver("min", get_min)
OmegaConf.register_new_resolver("str", resolve_str)
OmegaConf.register_new_resolver("number_of_exo_vars", number_of_exo_vars)


def main():
    parser = argparse.ArgumentParser(description="WandB Results")

    def list_of_strings(arg: str) -> list[str]:
        return arg.split(",")

    parser.add_argument(
        "--datasets",
        type=list_of_strings,
        required=False,
        default=["dalia", "wildppg8", "ieee"],
        help="Dataset to plot. Must be one or more (separated by,) of 'ieee', 'dalia', 'wildppg' ",
    )

    parser.add_argument(
        "--type",
        choices=[
            "viz",
            "infos",
            "scatter",
            "granger",
            "pearson",
            "mutual",
            "test",
            "acf",
            "window_stats",
            "difference",
            "pacf",
            "bds",
            "forecastibility",
            "pca",
            "catch22",
            "norm_viz",
            "beliefppg",
            "adfuller",
            "chaos",
            "wildppg_exo",
            "cp_detection",
            "downsample",
            "svg",
        ],
        required=True,
    )

    args = parser.parse_args()
    datamodules: List[LightningDataModule] = []
    for dataset in args.datasets:
        with initialize(version_base=None, config_path="../../config/"):
            cfg = compose(
                config_name="config",
                overrides=[
                    f"dataset={dataset}",
                    "folds=all",
                    "feature=mean",
                ],
            )

        datamodule = instantiate(cfg.dataset.datamodule)
        datamodule.setup("fit")
        datamodules.append(datamodule)
    if args.type == "adfuller":
        adfuller_test(datamodules)
    elif args.type == "wildppg_exo":
        viz_exo_wildppg()
    elif args.type == "beliefppg":
        visualize_histogram(datamodules)
    elif args.type == "norm_viz":
        visualize_norm(datamodules)
    if args.type == "catch22":
        compute_catch22_correlation(datamodules)
    elif args.type == "pca":
        pca_plot(datamodules)
    elif args.type == "viz":
        visualize_timeseries(datamodules)
    elif args.type == "infos":
        print_infos(datamodules)
    elif args.type == "forecastibility":
        forecastibility(datamodules)
    elif args.type == "granger":
        granger_test(datamodules)
    elif args.type == "pearson":
        max_pearson(datamodules)
    elif args.type == "pacf":
        partial_correlation(datamodules)
    elif args.type == "bds":
        bds_test(datamodules)
    elif args.type == "chaos":
        chaos_script(datamodules)
    elif args.type == "scatter":
        scatter_plots(datamodules)
    elif args.type == "cp_detection":
        change_point_detection(datamodules)
    elif args.type == "downsample":
        downsample_series(datamodules)
    elif args.type == "svg":
        ieee_svg(datamodules)


def ieee_svg(datamodules):
    assert len(datamodules) == 1
    assert datamodules[0].name == "ieee"
    dm = datamodules[0]
    data = dm.train_dataset.data
    for i, series in enumerate(data):
        hr = series[:, 0]
        fig = px.line(hr)
        fig.update_xaxes(visible=False, showgrid=False, zeroline=False)
        fig.update_yaxes(visible=False, showgrid=False, zeroline=False)
        fig.update_layout(showlegend=False, template="none")
        fig.write_image(f"./data/ieee_{i}.svg")
        fig.show()


def downsample_series(datamodules):
    N_SERIES = 1
    factor = 30
    fig = make_subplots(rows=len(datamodules) * N_SERIES * 2, cols=1)
    for k, dm in enumerate(datamodules):
        data = dm.train_dataset.data[:N_SERIES]
        for j, series in enumerate(data):
            y = series[:, 0]
            imu = series[:, 1]
            n = (len(y) // factor) * factor
            y60 = y[:n].reshape(-1, factor).mean(axis=1)
            imu60 = imu[:n].reshape(-1, factor).mean(axis=1)
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(y60))),
                    y=y60,
                ),
                row=k * 2 * N_SERIES + 2 * j + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(y60))),
                    y=imu60,
                ),
                row=k * 2 * N_SERIES + 2 * j + 2,
                col=1,
            )

    fig.show()


def change_point_detection(datamodules):
    N_SERIES = 1
    for dm in datamodules:
        data = dm.train_dataset.data[:N_SERIES]
        for series in data:
            y = series[:, 0]
            model = "l2"
            algo = rpt.Pelt(model=model, min_size=20).fit(y)
            bkps = algo.predict(pen=10.0)  # tune 'pen' (see tips below)
            print(
                f"Number of CP: {len(bkps)}, Relative Score (len(bkps) / len(y)): {len(bkps) / len(y):.4}"
            )
            rpt.display(y, bkps)
            plt.show()


def adfuller_test(datamodules: List[LightningDataModule]):
    for datamodule in datamodules:
        print(f"DATASET: {datamodule.name}")
        data = datamodule.train_dataset.data
        stats, diff_stats = [], []
        p, diff_p = [], []
        for series in data:
            heartrate = series[:, 0]
            test = adfuller(heartrate, regression="ct")
            diff = np.diff(heartrate)
            diff_test = adfuller(diff, regression="ct")

            stats.append(test[0])
            p.append(test[1])

            diff_stats.append(diff_test[0])
            diff_p.append(diff_test[1])

            print(f"Test Statistic: {test[0]:.4f} | P Value: {test[1]:.4f}")
            print(
                f"DIFF Test Statistic: {diff_test[0]:.4f} | P Value: {diff_test[1]:.4f}"
            )

        print(f"Test Mean Stats {np.mean(stats):.4f} | Mean P: {np.mean(p):.4f}")
        print(
            f"DIFF Test Mean Stats {np.mean(diff_stats):.4f} | Mean P: {np.mean(diff_p):.4f}"
        )


def visualize_histogram(datamodules: List[LightningDataModule]):
    for datamodule in datamodules:
        data = datamodule.train_dataset.data
        logs = []
        for series in data:
            heartrate = series[:, 0]
            log_ratio = np.log(heartrate[1:] / heartrate[:-1])
            logs.append(log_ratio)

        concatenated = np.concatenate(logs)
        print(f"Mean {np.mean(concatenated)}")
        print(f"Std {np.std(concatenated)}")
        fig = px.histogram(concatenated, nbins=30, title=f"Histogram {datamodule.name}")
        fig.show()


def visualize_norm(datamodules: List[LightningDataModule]):
    n_series_per_dataset = 1
    pos = 1
    n_datasets = len(datamodules)
    row_names = []
    for d in datamodules:
        name = dataset_to_name[d.name]
        for _ in range(n_series_per_dataset):
            row_names.append(name + " HR")
            row_names.append(name + " Norm")

    fig = make_subplots(
        rows=2 * n_series_per_dataset * n_datasets, cols=1, row_titles=row_names
    )
    for j, datamodule in enumerate(datamodules):
        train_dataset = datamodule.train_dataset
        dataset = train_dataset.data
        pos = min(len(dataset) - 1, pos)
        dataset = dataset[pos : pos + n_series_per_dataset]
        offset = 2 * j * n_series_per_dataset
        mean = train_dataset.mean[0]
        std = train_dataset.std[0]
        for i, series in tqdm(enumerate(dataset)):
            heartrate = series[:, 0]
            globally = (heartrate - mean) / (std + 1e-8)
            locally, _, _ = local_z_norm_numpy(heartrate[np.newaxis, :, np.newaxis])
            locally = locally[0, :, 0]
            min_max, _, _ = min_max_norm_numpy(heartrate[np.newaxis, :, np.newaxis])
            min_max = min_max[0, :, 0]
            differenced = np.diff(heartrate)

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(heartrate))) * 2,
                    y=heartrate,
                    showlegend=True,
                    name="Heartrate",
                    opacity=1.0,
                ),
                row=2 * i + 1 + offset,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(globally))) * 2,
                    y=globally,
                    showlegend=True,
                    name="Global",
                    opacity=0.7,
                ),
                row=2 * i + 2 + offset,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(locally))) * 2,
                    y=locally,
                    showlegend=True,
                    name="Local",
                    opacity=0.7,
                ),
                row=2 * i + 2 + offset,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(differenced))) * 2,
                    y=differenced,
                    showlegend=True,
                    name="Difference",
                    opacity=0.7,
                ),
                row=2 * i + 2 + offset,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(min_max))) * 2,
                    y=min_max,
                    showlegend=True,
                    name="Min-Max",
                    opacity=0.7,
                ),
                row=2 * i + 2 + offset,
                col=1,
            )

    fig.update_xaxes(
        title_text="Seconds", row=2 * n_series_per_dataset * n_datasets, col=1
    )
    fig.show()


def forecastibility(datamodules: List[LightningDataModule]):
    sf = 0.5
    for datamodule in datamodules:
        dataset = datamodule.train_dataset.data
        # define length of window for STFFT
        if datamodule.name in ["ieee"]:
            nperseg = 32
        else:
            nperseg = 256
        results = []
        for x in dataset:
            heartrate = x[:, 0]
            H = ant.spectral_entropy(
                heartrate, sf=sf, method="welch", normalize=True, nperseg=nperseg
            )
            res = 1 - H
            results.append(res)

        print(
            f"Dataset {datamodule.name} | Mean {np.mean(results):.4f} | Std {np.std(results):.4f}"
        )


def partial_correlation(
    datamodules: List[LightningDataModule], nlags: int = 10, use_diff: bool = True
):
    row_titles = [dataset_to_name[d.name] for d in datamodules]
    fig = make_subplots(
        rows=len(datamodules),
        cols=1,
        shared_yaxes=True,
        vertical_spacing=0.05,
        row_titles=row_titles,
    )
    for i, datamodule in enumerate(datamodules):
        dataset = datamodule.train_dataset.data
        pcs = []
        for series in dataset:
            heartrate = series[:, 0]
            if use_diff:
                heartrate = np.diff(heartrate)
            partial_ac = pacf(heartrate, nlags=nlags)
            pcs.append(partial_ac)

        mean_pac = np.mean(np.stack(pcs), axis=0)
        std_pac = np.std(np.stack(pcs), axis=0)
        n = len(mean_pac)
        x_vals = list(range(n))

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=mean_pac,
                mode="lines+markers",
                error_y=dict(
                    type="data",
                    array=std_pac,
                    arrayminus=std_pac,
                    visible=True,
                ),
            ),
            row=i + 1,
            col=1,
        )
    for ann in fig.layout.annotations:
        ann.font.size = 24
        ann.font.color = "black"
        ann.text = f"<b>{ann.text}</b>"
    fig.update_yaxes(range=[-1, 1])
    fig.update_layout(showlegend=False)
    fig.show()


def print_infos(datamodules: List[LightningDataModule]):
    for datamodule in datamodules:
        dataset = datamodule.train_dataset.data

        def get_channel_stats(dataset):
            mins = [np.min(s) for s in dataset]
            maxs = [np.max(s) for s in dataset]
            means = [np.mean(s) for s in dataset]
            median = [np.median(s) for s in dataset]
            stds = [np.std(s) for s in dataset]
            return pd.DataFrame(
                {
                    "mean": means,
                    "median": median,
                    "std": stds,
                    "min": mins,
                    "max": maxs,
                }
            )

        # Stats for HR and Activity
        hr_stats = get_channel_stats([s[:, 0] for s in dataset])
        act_stats = get_channel_stats([s[:, 1] for s in dataset])
        print("HR Mean")
        print(hr_stats.mean())
        print("HR STDS")
        print(hr_stats.std())

        lengths = [len(s) for s in dataset]
        print(f"Total length of dataset {datamodule.name} is {sum(lengths)}")
        print(f"Min Length: {min(lengths)}")
        print(f"MaxLength: {max(lengths)}")


def max_pearson_corr(y, x, max_lag=20):
    lags = np.arange(-max_lag, max_lag + 1)
    corr = []
    for lag in lags:
        if lag < 0:
            corr.append(np.corrcoef(x[:lag], y[-lag:])[0, 1])
        elif lag > 0:
            corr.append(np.corrcoef(x[lag:], y[:-lag])[0, 1])
        else:
            corr.append(np.corrcoef(x, y)[0, 1])
    max_cor = np.abs(np.array(corr)).max()
    max_lag = lags[np.abs(np.array(corr)).argmax()]
    return float(max_cor), int(max_lag)


def max_pearson(datamodules: List[LightningDataModule], differencing: bool = True):
    print(f"Differencing = {differencing}")
    for datamodule in datamodules:
        print(f"Start computing Pearson for {datamodule.name}")
        dataset = datamodule.train_dataset.data

        pearsons = []
        best_lags = []
        for i, series in tqdm(enumerate(dataset)):
            heartrate = series[:, 0]
            activity = series[:, 1]
            if differencing:
                heartrate = np.diff(heartrate, n=1)
                activity = np.diff(activity, n=1)
            max_corr, best_lag = max_pearson_corr(heartrate, activity)
            pearsons.append(max_corr)
            best_lags.append(best_lag)
        print(list(zip(pearsons, best_lags)))
        mean_pearson = np.mean(pearsons)
        median_lag = np.median(best_lags)
        print(f"Mean pearson {mean_pearson} | median lag {median_lag} ")


def visualize_timeseries(datamodules: List[LightningDataModule]):
    n_series_per_dataset = 1
    pos = 1
    n_datasets = len(datamodules)
    row_names = []
    for d in datamodules:
        name = dataset_to_name[d.name]
        for _ in range(n_series_per_dataset):
            row_names.append(name + " HR")
            row_names.append(name + " ACT")

    fig = make_subplots(
        rows=2 * n_series_per_dataset * n_datasets,
        cols=1,
        row_titles=row_names,
        shared_xaxes=False,
    )
    for j, datamodule in enumerate(datamodules):
        dataset = datamodule.train_dataset.data
        pos = min(len(dataset) - 1, pos)
        dataset = dataset[pos : pos + n_series_per_dataset]
        offset = 2 * j * n_series_per_dataset
        for i, series in tqdm(enumerate(dataset)):
            heartrate = series[:, 0]
            activity = series[:, 1]
            r1 = 2 * i + 1 + offset  # HR row
            r2 = r1 + 1  # ACT row
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(series))) * 2,
                    y=heartrate,
                    showlegend=False,  # Legend redundant here
                ),
                row=r1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(series))) * 2,
                    y=activity,
                    showlegend=False,  # Legend redundant here
                ),
                row=r2,
                col=1,
            )
            parent_axis_id = "x" if r1 == 1 else f"x{r1}"
            fig.update_xaxes(matches=parent_axis_id, row=r2, col=1)
    fig.update_xaxes(
        title_text="Seconds", row=2 * n_series_per_dataset * n_datasets, col=1
    )
    fig.show()


def granger_test(datamodules: List[LightningDataModule]):
    lags = [1, 3, 5, 10, 20, 30]
    for datamodule in datamodules:
        print(f"Granger Causality Test for {datamodule.name}")
        dataset = datamodule.train_dataset.data
        mean_p: List[float] = []
        for i, series in tqdm(enumerate(dataset)):
            p_values: List[float] = []
            for lag in lags:
                gc_res = grangercausalitytests(series, [lag], verbose=False)
                p_value = gc_res[lag][0]["ssr_ftest"][1]
                p_values.append(round(p_value, 3))

            mean_p.append(np.mean(p_values))

        mean_means = np.mean(mean_p)
        std_means = np.std(mean_p)
        print(f"Mean: {mean_means:.4f} and Std: {std_means:.4f}")


def scatter_plots(datamodules: List[LightningDataModule]):
    lags = 10
    n_series_per_dataset = 1 if len(datamodules) == 3 else 5
    n_dataset = len(datamodules)
    fig = make_subplots(
        rows=n_series_per_dataset * n_dataset,
        cols=lags,
        column_titles=[f"Lag {i}" for i in range(lags)],
    )
    for j, datamodule in enumerate(datamodules):
        dataset = datamodule.train_dataset.data[:n_series_per_dataset]
        offset = n_series_per_dataset * j
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
                    row=i + 1 + offset,
                    col=lag + 1,
                )

        fig.update_layout(
            # title=f"Scatter Plots plotting Activity against Heartrate for {dataset}",
            height=200 * n_series_per_dataset * n_dataset,
            width=200 * lags,
            showlegend=False,
        )

    fig.show()


def mutual_information(
    datamodules: List[LightningDataModule], differencing: bool = True
):
    for datamodule in datamodules:
        print(f"Mutual Information Statistics for {datamodule.name}")
        dataset = datamodule.train_dataset.data
        mis = []
        for i, series in enumerate(dataset):
            heartrate = series[:, 0]
            activity = series[:, 1]
            if differencing:
                heartrate = np.diff(heartrate, n=1)
                activity = np.diff(activity, n=1)

            activity = activity.reshape(-1, 1)
            mi = mutual_info_regression(activity, heartrate)
            mis.append(mi[0])

        print(f"Mean {np.mean(mis)} | Std {np.std(mis)}")


def bds_test(datamodules: List[LightningDataModule]):
    n_datasets = len(datamodules)
    n_series_per_dataset = 1
    fig = make_subplots(rows=n_datasets * n_series_per_dataset, cols=1)
    for j, datamodule in enumerate(datamodules):
        print(f"BDS Test for {datamodule.name}")
        dataset = datamodule.train_dataset.data
        offset = n_series_per_dataset * j
        epsilon_res_stats = defaultdict(list)
        epsilon_res_p_val = defaultdict(list)
        for i, series in enumerate(dataset):
            heartrate = series[:, 0]
            heartrate = (heartrate - np.mean(heartrate)) / np.std(heartrate)
            #  order_selection = arma_order_select_ic(
            #      heartrate, max_ar=10, max_ma=10, ic="aic", trend="n"
            #  )
            #  best_order = order_selection.aic_min_order  # (p, q)

            print(f"\n=== Series {i} ===")

            model = ARIMA(heartrate, order=(10, 0, 10), trend="n")
            model_fit = model.fit()

            residuals = model_fit.resid
            if i < n_series_per_dataset:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(residuals))),
                        y=residuals,
                        mode="lines",
                    ),
                    row=i + 1 + offset,
                    col=1,
                )

            std = np.std(residuals)
            epsilons = np.array([0.5, 1.0, 1.5, 2.0])
            for epsilon in [epsilons[0]]:
                bds_result = bds(residuals, max_dim=3, epsilon=epsilon * std)
                epsilon_res_stats[epsilon].append(bds_result[0])
                epsilon_res_p_val[epsilon].append(bds_result[1])

        for key, item in epsilon_res_p_val.items():
            print(f"epsilon={key} * std")
            print(np.mean(np.stack(item), axis=0))

    fig.show()


def compute_catch22_correlation(datamodules: List[LightningDataModule]):
    for datamodule in datamodules:
        data = datamodule.train_dataset.data
        correlations: List[float] = []
        for series in data:
            heartrate = series[:, 0]
            activity = series[:, 1]
            h_res = pycatch22.catch22_all(heartrate)["values"]
            a_res = pycatch22.catch22_all(activity)["values"]
            correlation = np.corrcoef(h_res, a_res, rowvar=False)
            correlations.append(correlation[0, 1])

        final_cor = np.mean(correlations) + 1 / (1 + np.std(correlations))
        print(f"Mean {np.mean(correlations)}")
        print(f"STD {np.std(correlations)}")
        print(f"Dataset {datamodule.name} has cor = {final_cor}")


def pca_plot(datamodules: List[LightningDataModule]) -> None:
    point_size = 16

    def series_to_stats(series: NDArray[np.float32], nperseg: int = 256):
        LAG = 10
        heartrate = series[:, 0]
        activity = series[:, 1]
        mean = np.mean(heartrate)
        std = np.std(heartrate)
        minimum = np.min(heartrate)
        maximum = np.max(heartrate)
        median = np.median(heartrate)
        pacf_stat: List[float] = pacf(heartrate, nlags=3)
        pacf1 = pacf_stat[1]
        pacf2 = pacf_stat[2]
        pacf3 = pacf_stat[3]
        max_r, lag = max_pearson_corr(heartrate, activity)
        result = adfuller(heartrate, regression="ctt")
        granger = grangercausalitytests(series, [LAG])[LAG][0]["ssr_ftest"][0]
        mi = mutual_info_regression(activity.reshape(-1, 1), heartrate)

        sf = 0.5
        H = ant.spectral_entropy(
            heartrate, sf=sf, method="welch", normalize=True, nperseg=nperseg
        )
        res = 1 - H
        # TODO
        # return dict(mean=mean, std=std, minimum=minimum, maximum=maximum, median=median,pacf1=pacf1, pacf2=pacf2, pacf3=pacf3)

        return [
            mean,
            std,
            minimum,
            maximum,
            median,
            max_r,
            lag,
            pacf1,
            pacf2,
            pacf3,
            result[0],
            granger,
            mi,
            res,
        ]

    columns = [
        "mean",
        "std",
        "min",
        "max",
        "median",
        "max_r",
        "lag",
        "pacf1",
        "pacf2",
        "pacf3",
        "adf",
        "granger",
        "mi",
        "forecastibility",
        "Dataset",
    ]
    rows: List[float] = []
    for i, datamodule in enumerate(datamodules):
        data = datamodule.train_dataset.data
        dataset_name = dataset_to_name[datamodule.name]
        nperseg = 32 if datamodule.name == "ieee" else 256
        for series in data:
            series_stats = series_to_stats(series, nperseg=nperseg)
            series_stats += [dataset_name]  # dataset indicator
            rows.append(series_stats)

    df = pd.DataFrame(rows, columns=columns)

    color_col = "Dataset"  # column to color points by
    n_components = 2

    X = df.values[:, :-1]
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=0)
    Z = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], color_col: df[color_col]})

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Dataset",  # Now legend will show the dataset name
        symbol=color_col,  # This keeps your 3-color grouping separate
        opacity=0.85,
        labels={
            "PC1": f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
            "PC2": f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)",
        },
        # title="PCA projection",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )

    fig.update_traces(marker=dict(size=point_size))  # points in the plot
    fig.update_layout(coloraxis_showscale=False)

    fig.update_layout(
        xaxis_title=dict(
            text=f"<b>PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)</b>",
            font=dict(size=18),
        ),
        yaxis_title=dict(
            text=f"<b>PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)</b>",
            font=dict(size=18),
        ),
    )

    fig.update_layout(
        legend=dict(font=dict(size=24), itemsizing="constant"),
    )

    fig.show()

    n = X_scaled.shape[0]
    perplexity = int(min(30, max(5, (n - 1) // 3))) if n > 10 else max(2, n // 2)

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        n_iter=1000,
        random_state=42,
        verbose=0,
    )
    Z_tsne = tsne.fit_transform(X_scaled)

    tsne_df = pd.DataFrame(
        {
            "TSNE1": Z_tsne[:, 0],
            "TSNE2": Z_tsne[:, 1],
            color_col: pca_df[color_col],
        }
    )

    fig_tsne = px.scatter(
        tsne_df,
        x="TSNE1",
        y="TSNE2",
        color="Dataset",
        symbol=color_col,
        opacity=0.85,
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig_tsne.update_traces(marker=dict(size=point_size))
    fig_tsne.show()


# -----------------------------------------------------------------------------------------------------------------------
# BERKEN SCRIPT
def resample_uniform_from_time(
    time: np.ndarray, x: np.ndarray, fs_target: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolate onto a uniform grid at fs_target Hz. Returns (t_uniform, x_uniform)."""
    if time.ndim != 1 or x.ndim != 1 or len(time) != len(x):
        raise ValueError("time and x must be 1D arrays of equal length")
    if len(time) < 4:
        raise ValueError("need at least 4 samples to resample")
    t = np.asarray(time, dtype=float)
    x = np.asarray(x, dtype=float)
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    t_uniform = np.arange(t[0], t[-1], 1.0 / fs_target)
    xu = np.interp(t_uniform, t, x)
    return t_uniform, xu


def resample_uniform_from_fs(
    x: np.ndarray, fs: float, fs_target: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Uniform resample given constant sampling rate fs (Hz)."""
    n = len(x)
    t = np.arange(n, dtype=float) / float(fs)
    return resample_uniform_from_time(t, x, fs_target)


def standardize(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd == 0 or not np.isfinite(sd):
        return x - mu
    return (x - mu) / sd


def average_mutual_information(
    x: np.ndarray, max_lag: int, bins: int = 32
) -> np.ndarray:
    """Histogram-based AMI for lags 1..max_lag. Returns array length max_lag."""
    x = np.asarray(x, dtype=float)
    x = (x - np.nanmean(x)) / (np.nanstd(x) + 1e-12)
    hist, bin_edges = np.histogram(x, bins=bins, density=True)

    def bin_index(vals):
        return np.clip(np.digitize(vals, bin_edges) - 1, 0, bins - 1)

    idx = bin_index(x)
    ami = np.zeros(max_lag, dtype=float)
    for tau in range(1, max_lag + 1):
        a = idx[:-tau]
        b = idx[tau:]
        joint = np.zeros((bins, bins), dtype=float)
        for i, j in zip(a, b):
            joint[i, j] += 1.0
        joint /= joint.sum()
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        joint = np.where(joint > 0, joint, 1e-12)
        px = np.where(px > 0, px, 1e-12)
        py = np.where(py > 0, py, 1e-12)
        ami[tau - 1] = np.sum(joint * np.log(joint / (px[:, None] * py[None, :])))
    return ami


def first_local_min(arr: np.ndarray, min_index: int = 1) -> Optional[int]:
    """First index (1-based) of a strict local minimum. Returns None if none found."""
    for i in range(min_index, len(arr) - 1):
        if arr[i] < arr[i - 1] and arr[i] < arr[i + 1]:
            return i + 1
    return None


def autocorr_based_tau(x: np.ndarray, max_lag: int) -> int:
    """Fallback tau: first lag where autocorr drops below 1/e, else first zero crossing, else max_lag//4."""
    xz = x - np.mean(x)
    n = len(xz)
    ac = np.correlate(xz, xz, mode="full")[n - 1 : n - 1 + max_lag + 1]
    ac = ac / (ac[0] + 1e-12)
    for k in range(1, len(ac)):
        if ac[k] < 1.0 / math.e:
            return k
    for k in range(1, len(ac)):
        if ac[k] <= 0:
            return k
    return max(1, max_lag // 4)


def embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    """Delay embedding matrix (N_eff, m)."""
    n = len(x)
    N = n - (m - 1) * tau
    if N <= 2 * m:
        raise ValueError("time series too short for embedding with chosen m,tau")
    Y = np.zeros((N, m), dtype=float)
    for i in range(m):
        Y[:, i] = x[i * tau : i * tau + N]
    return Y


def false_nearest_neighbors(
    x: np.ndarray,
    tau: int,
    m_max: int = 10,
    Rtol: float = 15.0,
    Atol: float = 2.0,
    theiler: int = 10,
) -> Tuple[int, List[float]]:
    """
    Return (chosen m, list of %false for m=1..m_max) using Kennel et al. criteria.
    Fix: ensure the subsampled indices are valid for BOTH m and m+1 embeddings.
    """
    rng = np.random.default_rng(0)
    ratios = []
    m_chosen = m_max

    for m in range(1, m_max + 1):
        Y = embed(x, m, tau)
        N = len(Y)

        # candidate indices in the m-embedding
        idx = np.arange(N)
        if N > 4000:
            idx = rng.choice(N, size=4000, replace=False)

        # nearest neighbors in the m-embedding (on the subsample)
        Y_sub = Y[idx]
        D = np.sqrt(((Y_sub[:, None, :] - Y_sub[None, :, :]) ** 2).sum(axis=2))

        # Theiler window: mask near-diagonal (temporal neighbors)
        for i in range(len(idx)):
            lo = max(0, i - theiler)
            hi = min(len(idx), i + theiler + 1)
            D[i, lo:hi] = np.inf

        nn = np.argmin(D, axis=1)
        dist_m = D[np.arange(len(idx)), nn]

        if m < m_max:
            # Build (m+1)-embedding and restrict to rows valid in BOTH spaces
            Yp = embed(x, m + 1, tau)
            Np = len(Yp)

            # keep only those subsampled indices that are < Np
            valid_rows = np.where(idx < Np)[0]
            if len(valid_rows) < 2:
                # not enough points to evaluate false neighbors at this m; fall back
                ratios.append(ratios[-1] if ratios else 100.0)
                continue

            # restrict everything to valid_rows so we can index Yp consistently
            idx_v = idx[valid_rows]
            Yp_sub = Yp[idx_v]

            # also restrict the nn mapping to the same valid rows/cols
            # map from original subsample positions -> compact valid positions
            pos_map = -np.ones(len(idx), dtype=int)
            pos_map[valid_rows] = np.arange(len(valid_rows))
            nn_v = pos_map[nn[valid_rows]]
            dist_m_v = dist_m[valid_rows]

            # if any mapped neighbor is invalid (e.g., pointed to a dropped row), drop those rows
            good = nn_v >= 0
            if np.count_nonzero(good) < 2:
                ratios.append(ratios[-1] if ratios else 100.0)
                continue

            Yp_a = Yp_sub[good]
            Yp_b = Yp_sub[nn_v[good]]
            dist_m_good = dist_m_v[good]

            # distance growth in (m+1)-space
            Dp = np.sqrt(((Yp_a - Yp_b) ** 2).sum(axis=1))
            Ra = (Dp - dist_m_good) / (dist_m_good + 1e-12)

            false = (Ra > Rtol) | (Dp > Atol * np.std(x))
            ratio = 100.0 * np.mean(false)
        else:
            # for the last m, just repeat previous ratio
            ratio = ratios[-1] if ratios else 100.0

        ratios.append(ratio)
        if ratio < 2.0 and m_chosen == m_max:
            m_chosen = m

    return m_chosen, ratios


# --------- three metrics ---------


def rosenstein_lyapunov(
    x: np.ndarray,
    m: int,
    tau: int,
    theiler: int = 20,
    kmax: int = 50,
    fit_start: int = 1,
    fit_end: Optional[int] = None,
) -> Tuple[float, Tuple[int, int]]:
    """Largest Lyapunov exponent via Rosenstein method. Returns (lambda, (k0,k1))."""
    Y = embed(x, m, tau)
    N = len(Y)
    D = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))
    for i in range(N):
        lo = max(0, i - theiler)
        hi = min(N, i + theiler + 1)
        D[i, lo:hi] = np.inf
    nn = np.argmin(D, axis=1)
    kmax = min(kmax, N - 10)
    div = []
    for k in range(1, kmax + 1):
        valid = np.where((np.arange(N) + k < N) & (nn + k < N))[0]
        if len(valid) == 0:
            break
        d = np.linalg.norm(Y[valid + k] - Y[nn[valid] + k], axis=1)
        d = d[d > 1e-12]
        if len(d) == 0:
            break
        div.append(np.mean(np.log(d)))
    if not div:
        return float("nan"), (0, 0)
    div = np.array(div)
    if fit_end is None:
        fit_end = max(3, len(div) // 3)
    k0 = max(1, fit_start)
    k1 = min(len(div), fit_end)
    xk = np.arange(1, len(div) + 1, dtype=float)[k0 - 1 : k1]
    yk = div[k0 - 1 : k1]
    A = np.polyfit(xk, yk, 1)
    return float(A[0]), (k0, k1)


def zero_one_test(x: np.ndarray, n_c: int = 100, random_state: int = 0) -> float:
    """0–1 test for chaos (Gottwald–Melbourne). Returns median K over random c in (0, π)."""
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    N = len(x)
    if N < 200:
        return float("nan")
    ns = np.arange(1, N + 1)
    Ks = []
    for _ in range(n_c):
        c = rng.uniform(0, np.pi)
        pc = np.cumsum(x * np.cos(c * ns))
        qc = np.cumsum(x * np.sin(c * ns))
        M = (pc - np.mean(pc)) ** 2 + (qc - np.mean(qc)) ** 2
        nvec = (ns - ns.mean()) / (ns.std() + 1e-12)
        Mz = (M - M.mean()) / (M.std() + 1e-12)
        Ks.append(np.mean(nvec * Mz))
    return float(np.median(Ks))


def correlation_dimension(
    x: np.ndarray,
    m: int,
    tau: int,
    theiler: int = 20,
    n_samples: int = 4000,
    n_eps: int = 20,
) -> float:
    """Grassberger–Procaccia correlation dimension D2 from scaling of C(eps)."""
    Y = embed(x, m, tau)
    N = len(Y)
    if N > n_samples:
        idx = np.random.default_rng(0).choice(N, size=n_samples, replace=False)
        Y = Y[idx]
        N = len(Y)
    D = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))
    for i in range(N):
        lo = max(0, i - theiler)
        hi = min(N, i + theiler + 1)
        D[i, lo:hi] = np.inf
    finite = D[np.isfinite(D)]
    finite = finite[finite > 0]
    if len(finite) < 10:
        return float("nan")
    eps = np.quantile(finite, np.linspace(0.02, 0.4, n_eps))
    C = [np.mean((D < e) & np.isfinite(D)) for e in eps]
    C = np.array(C, dtype=float)
    mask = (C > 0.02) & (C < 0.2)
    if mask.sum() < 4:
        return float("nan")
    xe = np.log(eps[mask])
    ye = np.log(C[mask])
    return float(np.polyfit(xe, ye, 1)[0])


# --------- main API ---------


def analyze_hr_array(
    hr: np.ndarray,
    time: Optional[np.ndarray] = None,
    fs: Optional[float] = None,
    fs_target: float = 1.0,
    window_sec: int = 600,
    step_frac: float = 0.5,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Compute the three metrics from an HR array.
    Returns dict with: tau, m, LLE, zero_one_K, D2 and housekeeping.
    """
    hr = np.asarray(hr, dtype=float)
    if time is not None:
        tu, hru = resample_uniform_from_time(
            np.asarray(time, dtype=float), hr, fs_target
        )
    else:
        if fs is None:
            raise ValueError("Provide either 'time' or 'fs'.")
        tu, hru = resample_uniform_from_fs(hr, fs, fs_target)

    # Convert HR (BPM) to IBI (seconds) proxy and standardize
    hru = np.clip(hru, 20.0, 220.0)
    ibi = 60.0 / hru
    x = standardize(ibi)

    # Sliding windows aggregate by median
    win_len = int(window_sec * fs_target)
    step = max(1, int(win_len * step_frac))
    if len(x) < win_len:
        windows = [(0, len(x))]
    else:
        windows = [(i, i + win_len) for i in range(0, len(x) - win_len + 1, step)]

    rows = []
    for a, b in windows:
        xs = x[a:b]
        if len(xs) < 500:
            continue

        max_lag = min(60, len(xs) // 10)
        ami = average_mutual_information(xs, max_lag=max_lag, bins=32)
        tau = first_local_min(ami) or autocorr_based_tau(xs, max_lag=max_lag)
        tau = int(max(1, min(tau, 30)))

        m, _ = false_nearest_neighbors(xs, tau=tau, m_max=10, theiler=int(2 * tau))

        lle, (k0, k1) = rosenstein_lyapunov(
            xs, m=m, tau=tau, theiler=int(2 * tau), kmax=80
        )

        K01 = zero_one_test(xs, n_c=120, random_state=random_state)

        D2 = correlation_dimension(
            xs, m=m, tau=tau, theiler=int(2 * tau), n_samples=3000, n_eps=18
        )

        rows.append((tau, m, lle, k0, k1, K01, D2))

    if not rows:
        raise ValueError("Series too short after preprocessing to compute metrics.")

    A = np.array(rows, dtype=float)
    out = {
        "n_points_in": int(len(hr)),
        "n_points_resampled": int(len(hru)),
        "fs_target": float(fs_target),
        "tau": int(np.median(A[:, 0])),
        "m": int(np.median(A[:, 1])),
        "lle": float(np.median(A[:, 2])),
        "lle_fit_k0": int(np.median(A[:, 3])),
        "lle_fit_k1": int(np.median(A[:, 4])),
        "zero_one_K": float(np.median(A[:, 5])),
        "D2": float(np.nanmedian(A[:, 6])),
    }
    return out


def chaos_script(datamodules: List[LightningDataModule]):
    fs = 0.5
    for datamodule in datamodules:
        print(f"DATASET: {datamodule.name}")
        data = datamodule.train_dataset.data
        cols = defaultdict(list)
        for i, series in enumerate(data):
            print(f"Participant {i}")
            hr = 60.0 / series[:, 0]  # HR Shape (T, )
            if np.isinf(hr).any() or np.isnan(hr).any():
                print("ERROR in HR, either nan or inf")
                continue
            metrics = analyze_hr_array(hr, fs=fs, fs_target=1.0, window_sec=600)
            for k in ["tau", "m", "lle", "zero_one_K", "D2"]:
                print(k, ":", metrics[k])
                cols[k].append(metrics[k])

        df = pd.DataFrame.from_dict(cols)
        print("Raw Results:")
        print(df)
        print("Mean Results:")
        print(df.mean())


# ----------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
