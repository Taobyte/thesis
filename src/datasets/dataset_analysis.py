import argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import antropy as ant
import pycatch22


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
from typing import List
from numpy.typing import NDArray
from collections import defaultdict
from src.normalization import local_z_norm_numpy, min_max_norm_numpy

from src.constants import dataset_to_name

from src.utils import (
    compute_square_window,
    compute_input_channel_dims,
    get_optuna_name,
)


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


def max_pearson(datamodules: List[LightningDataModule]):
    for datamodule in datamodules:
        print(f"Start computing Pearson for {datamodule.name}")
        dataset = datamodule.train_dataset.data

        pearsons = []
        best_lags = []
        for i, series in tqdm(enumerate(dataset)):
            heartrate = series[:, 0]
            activity = series[:, 1]
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
        rows=2 * n_series_per_dataset * n_datasets, cols=1, row_titles=row_names
    )
    for j, datamodule in enumerate(datamodules):
        dataset = datamodule.train_dataset.data
        pos = min(len(dataset) - 1, pos)
        dataset = dataset[pos : pos + n_series_per_dataset]
        offset = 2 * j * n_series_per_dataset
        for i, series in tqdm(enumerate(dataset)):
            heartrate = series[:, 0]
            activity = series[:, 1]
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(series))) * 2,
                    y=heartrate,
                    showlegend=False,  # Legend redundant here
                ),
                row=2 * i + 1 + offset,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(series))) * 2,
                    y=activity,
                    showlegend=False,  # Legend redundant here
                ),
                row=2 * i + 2 + offset,
                col=1,
            )
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
    lags = 12
    n_series_per_dataset = 1
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


def mutual_information(datamodules: List[LightningDataModule]):
    for datamodule in datamodules:
        print(f"Mutual Information Statistics for {datamodule.name}")
        dataset = datamodule.train_dataset.data
        mis = []
        for i, series in enumerate(dataset):
            heartrate = series[:, 0]
            activity = series[:, 1].reshape(-1, 1)
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


def arch_test(datamodules: List[LightningDataModule]):
    pass


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


if __name__ == "__main__":
    OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("optuna_name", get_optuna_name)
    OmegaConf.register_new_resolver(
        "compute_input_channel_dims", compute_input_channel_dims
    )

    parser = argparse.ArgumentParser(description="WandB Results")

    def list_of_strings(arg):
        return arg.split(",")

    parser.add_argument(
        "--datasets",
        type=list_of_strings,
        required=False,
        default=["dalia", "wildppg", "ieee"],
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
                    "use_dynamic_features=True",
                    "use_heart_rate=True",
                ],
            )

        datamodule = instantiate(cfg.dataset.datamodule)
        datamodule.setup("fit")
        datamodules.append(datamodule)
    if args.type == "adfuller":
        adfuller_test(datamodules)
    elif args.type == "beliefppg":
        visualize_histogram(datamodules)
    elif args.type == "norm_viz":
        visualize_norm(datamodules)
    if args.type == "catch22":
        compute_catch22_correlation(datamodules)
    elif args.type == "pca":
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

        pca_df = pd.DataFrame(
            {"PC1": Z[:, 0], "PC2": Z[:, 1], color_col: df[color_col]}
        )

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
    elif args.type == "window_stats":
        window_length = 20
        fig = make_subplots(
            rows=len(dataset),
            cols=4,
            column_titles=["HR Mean", "HR Std", "ACT Mean", "ACT Std"],
        )
        for i, series in enumerate(dataset):
            heartrate = series[:, 0]
            activity = series[:, 1]
            hr_df = pd.DataFrame(heartrate)
            ac_df = pd.DataFrame(activity)

            hr_mean = hr_df.rolling(window=window_length).mean().dropna()[0].to_numpy()
            hr_std = hr_df.rolling(window=window_length).std().dropna()[0].to_numpy()

            ac_mean = ac_df.rolling(window=window_length).mean().dropna()[0].to_numpy()
            ac_std = ac_df.rolling(window=window_length).std().dropna()[0].to_numpy()

            x = list(range(len(hr_mean)))

            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=hr_mean,
                    mode="lines",
                ),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=hr_std,
                    mode="lines",
                ),
                row=i + 1,
                col=2,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ac_mean,
                    mode="lines",
                ),
                row=i + 1,
                col=3,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=ac_std,
                    mode="lines",
                ),
                row=i + 1,
                col=4,
            )

        fig.show()
    elif args.type == "bds":
        bds_test(datamodules)

    elif args.type == "arch":
        arch_test(datamodules)
