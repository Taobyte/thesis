import pandas as pd
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from typing import List, Tuple, Optional, Dict, Any
from tqdm import tqdm
from numpy.typing import NDArray
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
from omegaconf import OmegaConf
from hydra import initialize, compose
from hydra.utils import instantiate
from lightning import LightningDataModule
from collections import defaultdict
from scipy.signal import coherence as sp_coherence
from sklearn.feature_selection import mutual_info_regression

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning

from plotly.subplots import make_subplots

import plotly.graph_objects as go
import xgboost as xgb
from collections import Counter

from statsmodels.tsa.api import VAR


from src.utils import (
    get_optuna_name,
    compute_square_window,
    compute_input_channel_dims,
    get_min,
    resolve_str,
    ensemble_epochs,
    exo_channels_wildppg,
)


OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("optuna_name", get_optuna_name)
OmegaConf.register_new_resolver(
    "compute_input_channel_dims", compute_input_channel_dims
)
OmegaConf.register_new_resolver("min", get_min)
OmegaConf.register_new_resolver("str", resolve_str)
OmegaConf.register_new_resolver("ensemble_epochs", ensemble_epochs)
OmegaConf.register_new_resolver("exo_channels_wildppg", exo_channels_wildppg)

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=InterpolationWarning)


# ---------------------------------------------------------------------------------------------------------------------
# STATISTICAL MODEL PERFORMANCE COIMPARISON
# ---------------------------------------------------------------------------------------------------------------------


def _make_lag_matrix(x: NDArray, lags: List[int]) -> NDArray:
    """Return [x_{t-l} for l in lags] stacked column-wise, aligned to t."""
    cols = []
    for l in lags:
        if l == 0:
            cols.append(x.copy())
        else:
            z = np.empty_like(x)
            z[:l] = x[0]
            z[l:] = x[:-l]
            cols.append(z)
    return np.column_stack(cols)


def _standardize_train_apply(
    X_train: NDArray, X_val: NDArray
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-12
    return (X_train - mu) / sd, (X_val - mu) / sd, mu, sd


def _fit_best_arima_aic(
    y: NDArray, exog: Optional[NDArray], orders: List[Tuple[int, int, int]]
):
    """Fit SARIMAX over a small grid; return model with best AIC."""
    best = None
    best_aic = np.inf
    for p, d, q in orders:
        try:
            m = SARIMAX(
                y,
                exog=exog,
                order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            r = m.fit(disp=False)
            print(f"converged={r.mle_retvals.get('converged')}")
            if r.aic < best_aic and np.isfinite(r.aic):
                best, best_aic = r, r.aic
        except Exception:
            print(f"FIT FAILED for ({p}, {d}, {q})")
            continue

    return best


def _metrics(y_true: NDArray, y_pred: NDArray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(
        np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), 1e-8)))) * 100.0
    )
    return {"MAE": mae, "RMSE": rmse, "MAPE%": mape}


def _rolling_origin_eval(
    y: NDArray,
    X: Optional[NDArray],
    orders_grid: List[Tuple[int, int, int]],
    n_folds: int = 3,
    min_train: int = 600,
    val_len: int = 200,
) -> Dict[str, float]:
    """
    Expanding-window CV: for each fold, fit on [0:train_end) and predict next val_len.
    Returns averaged metrics.
    """
    T = len(y)
    if T < min_train + n_folds * val_len:
        # shrink if short series
        val_len = max(100, T // (n_folds + 2))
        min_train = max(300, T - (n_folds * val_len) - 50)
    best_coeffs = defaultdict(dict)
    metrics_accum = {"MAE": [], "RMSE": [], "MAPE%": []}
    for k in range(n_folds):
        train_end = min_train + k * val_len
        val_end = train_end + val_len
        if val_end > T:
            break

        y_tr, y_va = y[:train_end], y[train_end:val_end]
        if X is not None:
            X_tr, X_va = X[:train_end], X[train_end:val_end]
            # standardize exog based on train fold only
            X_tr_std, X_va_std, _, _ = _standardize_train_apply(X_tr, X_va)
        else:
            X_tr_std = X_va_std = None

        # fit best HR-only and HR+EXO on this fold
        print("-" * 25)
        print("Fit ARIMA")
        print("-" * 25)
        m_arima = _fit_best_arima_aic(y_tr, None, orders_grid)
        print("-" * 25)
        print("FIT ARIMAX")
        print("-" * 25)
        m_arimax = _fit_best_arima_aic(y_tr, X_tr_std, orders_grid)

        sig_vars = {
            name: (coef, pval)
            for name, coef, pval in zip(
                m_arimax.param_names, m_arimax.params, m_arimax.pvalues
            )
            if pval < 0.05 and name not in ["sigma2"]  # exclude sigma2
        }

        best_coeffs[k] = sig_vars

        # fallback: if fit failed, skip fold
        if m_arima is None:
            continue

        # Forecast val_len steps with optional exog
        yhat_arima = m_arima.get_forecast(steps=len(y_va)).predicted_mean
        if (m_arimax is not None) and (X_va_std is not None):
            yhat_arimax = m_arimax.get_forecast(
                steps=len(y_va), exog=X_va_std
            ).predicted_mean
        else:
            yhat_arimax = np.full_like(y_va, np.nan, dtype=float)

        # collect metrics
        met_arima = _metrics(y_va, yhat_arima)
        metrics_accum["MAE"].append((met_arima["MAE"],))
        metrics_accum["RMSE"].append((met_arima["RMSE"],))
        metrics_accum["MAPE%"].append((met_arima["MAPE%"],))

        if np.isfinite(yhat_arimax).all():
            met_arimax = _metrics(y_va, yhat_arimax)
            # store as tuple (hr_only, hr+exo) to compute deltas later
            metrics_accum["MAE"][-1] = (met_arima["MAE"], met_arimax["MAE"])
            metrics_accum["RMSE"][-1] = (met_arima["RMSE"], met_arimax["RMSE"])
            metrics_accum["MAPE%"][-1] = (met_arima["MAPE%"], met_arimax["MAPE%"])

    # average and compute deltas
    out = {}
    for k in ["MAE", "RMSE", "MAPE%"]:
        pairs = [p for p in metrics_accum[k] if len(p) == 2]
        if not pairs:
            continue
        hr_only = np.array([a for (a, b) in pairs], float)
        hr_exo = np.array([b for (a, b) in pairs], float)
        out[f"{k}_HR"] = float(hr_only.mean())
        out[f"{k}_HR+EXO"] = float(hr_exo.mean())
        out[f"{k}_Δabs"] = float(hr_exo.mean() - hr_only.mean())
        out[f"{k}_Δrel%"] = float(100.0 * (hr_exo.mean() / hr_only.mean() - 1.0))
    return out, best_coeffs


def statistical_performance_difference(datamodules: List[LightningDataModule]):
    # Small ARIMA grid: cheap but surprisingly strong
    ORDERS = [
        (1, 1, 1),
        (2, 1, 1),
        (1, 1, 2),
        (2, 1, 2),
        (3, 1, 1),
        (3, 1, 2),
    ]  # include d=1 options if needed

    # IMU lags in steps (2 s stride). {0,1,2,4} → {0,2,4,8} s
    IMU_LAGS = [0, 1, 2, 3]

    for dm in datamodules:
        print(f"\nDATASET: {dm.name}")
        data = dm.train_dataset.data
        series_results = []

        significant_coeffs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for j, series in tqdm(enumerate(data[:1])):
            y = series[:, 0].astype(float)  # HR
            x = series[:, 1].astype(float)  # base IMU feature you currently feed

            # Build a small, dynamic exogenous block:
            # level + first diff + short/long rolling std; then lag it.
            def roll_std(a, w):
                pad = np.repeat(a[:1], w - 1)
                aw = np.concatenate([pad, a])
                return np.array(
                    [aw[i - w + 1 : i + 1].std() for i in range(w - 1, len(aw))]
                )

            x_level = x
            x_diff = np.diff(x, prepend=x[0])
            x_std_s = roll_std(x, 2)  # ~2 s
            x_std_l = roll_std(x, 5)  # ~10 s
            # align lengths
            n = len(y)
            x_stack = np.column_stack(
                [z[:n] for z in [x_level, x_diff, x_std_s, x_std_l]]
            )

            # lag the whole block
            X = None
            for l in IMU_LAGS:
                X_l = _make_lag_matrix(
                    x_stack, [l]
                )  # returns same cols but lagged by l
                X = X_l if X is None else np.column_stack([X, X_l])

            X = _make_lag_matrix(x, IMU_LAGS)

            res, best_coeffs = _rolling_origin_eval(
                y, X, ORDERS, n_folds=3, min_train=2000, val_len=200
            )
            significant_coeffs[dm.name][j] = best_coeffs
            if res:
                series_results.append(res)

        if not series_results:
            print("No successful fits.")
            continue
        keys = series_results[0].keys()
        agg = {
            k: float(np.nanmean([r.get(k, np.nan) for r in series_results]))
            for k in keys
        }
        print("Avg over series:")
        for k, v in agg.items():
            print(f"  {k:>12}: {v:8.4f}")

        if "RMSE_Δrel%" in agg:
            print(
                f"→ Relative RMSE change (HR+EXO vs HR-only): {agg['RMSE_Δrel%']:.2f}% "
                f"(negative = improvement)"
            )

        print(significant_coeffs)

        for dataset_name, series_res in significant_coeffs.items():
            best_coeffs: set[str] = set()
            for _, items in series_res.items():
                for _, t in items.items():
                    best_coeffs.update(t.keys())

            print(f"Significant Coefficients for {dataset_name}")
            print(best_coeffs)


# ---------------------------------------------------------------------------------------------------------------------
# CROSS CORRELATION FUNCTION
# ---------------------------------------------------------------------------------------------------------------------


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
        # print(list(zip(pearsons, best_lags)))
        mean_pearson = np.mean(pearsons)
        median_lag = np.median(best_lags)
        print(f"Mean pearson {mean_pearson} | median lag {median_lag} ")


def cross_correlation(datamodules):
    max_pearson(datamodules, differencing=False)
    max_pearson(datamodules, differencing=True)


# ---------------------------------------------------------------------------------------------------------------------
# GRANGER CAUSALITY TEST
# ---------------------------------------------------------------------------------------------------------------------


def _stationarity_votes(x: np.ndarray, alpha: float = 0.05):
    """
    Returns (all_agree_stationary, details_dict) where 'all_agree_stationary' is True
    iff ADF and PP both reject unit root (p<alpha) AND KPSS fails to reject stationarity (p>=alpha).
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 10:  # too short to be confident
        return False, {"adf_p": np.nan, "pp_p": np.nan, "kpss_p": np.nan}

    adf_p = adfuller(x, autolag="AIC", regression="c")[1]
    pp_p = phillips_perron(x, trend="c")[1]
    kpss_p = kpss(x, regression="c", nlags="auto")[1]

    adf_stationary = (not np.isnan(adf_p)) and (adf_p < alpha)
    pp_stationary = (not np.isnan(pp_p)) and (pp_p < alpha)
    kpss_stationary = (not np.isnan(kpss_p)) and (kpss_p >= alpha)

    all_agree = adf_stationary and pp_stationary and kpss_stationary
    return all_agree, {"adf_p": adf_p, "pp_p": pp_p, "kpss_p": kpss_p}


def _adf_kpss_pp_diff_once_if_needed(
    x: np.ndarray,
    p_thresh: float = 0.05,
    return_details: bool = False,
):
    """
    Difference once only if the three-test vote does NOT agree on stationarity.
    Returns (series, d) or (series, d, details) if return_details=True.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size <= 2:
        return (
            (x, 0, {"adf_p": np.nan, "pp_p": np.nan, "kpss_p": np.nan})
            if return_details
            else (x, 0)
        )

    agree_stationary, details = _stationarity_votes(x, alpha=p_thresh)

    if agree_stationary:
        out = (x, 0, details) if return_details else (x, 0)
    else:
        dx = np.diff(x)
        out = (dx, 1, details) if return_details else (dx, 1)
    return out


def _fdr_bh(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros_like(p, dtype=bool)
    kmax = np.max(np.where(passed))
    mask = np.zeros_like(p, dtype=bool)
    mask[order[: kmax + 1]] = True
    return mask


def granger_imu_to_hr(
    datamodules: List[Any],
    maxlags: int = 30,
    do_stationarity_check: bool = True,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Tests IMU ⇒ HR per series (2 columns: [HR, IMU]).
    Chooses VAR lag via AIC (≤ maxlags). Returns DataFrame of results and prints a summary.
    """
    rows = []
    for dm in datamodules:
        data = dm.train_dataset.data
        for si, series in enumerate(data):
            if series.ndim != 2 or series.shape[1] < 2:
                continue
            hr_raw = series[:, 0].astype(float)
            imu_raw = series[:, 1].astype(float)

            if do_stationarity_check:
                hr, d_hr, hr_tests = _adf_kpss_pp_diff_once_if_needed(
                    hr_raw, p_thresh=0.05, return_details=True
                )
                imu, d_imu, imu_tests = _adf_kpss_pp_diff_once_if_needed(
                    imu_raw, p_thresh=0.05, return_details=True
                )
                # print("HR tests:", hr_tests, "IMU tests:", imu_tests)
                print("d_hr", d_hr, "d_imu", d_imu)
            else:
                hr, imu, d_hr, d_imu = hr_raw, imu_raw, 0, 0

            n = min(len(hr), len(imu))
            if n < max(50, maxlags + 10):
                continue
            df = (
                pd.DataFrame({"HR": hr[:n], "IMU": imu[:n]})
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(df) < max(50, maxlags + 10):
                continue

            # Select lag by AIC and fit VAR
            try:
                sel = VAR(df).select_order(maxlags=maxlags)
                k_ar = int(sel.hqic) if np.isfinite(sel.aic) else min(maxlags, 4)
                k_ar = max(1, min(k_ar, maxlags))
                print(f"aic: {sel.aic}, bic: {sel.bic}, hqic: {sel.hqic}")
            except Exception:
                k_ar = min(maxlags, 4)
            try:
                model = VAR(df).fit(k_ar)
            except Exception:
                # fallback to smaller lag if needed
                tried = False
                for k in range(min(4, k_ar), 0, -1):
                    try:
                        model = VAR(df).fit(k)
                        k_ar = k
                        tried = True
                        break
                    except Exception:
                        continue
                if not tried:
                    continue

            # Granger test: do past IMU lags jointly improve HR?
            try:
                gc = model.test_causality(caused="HR", causing="IMU", kind="f")
                p_xy = float(gc.pvalue)
                stat_xy = float(gc.test_statistic)
            except Exception:
                p_xy, stat_xy = np.nan, np.nan

            rows.append(
                {
                    "dataset": dm.name,
                    "series_idx": si,
                    "n_obs": len(df),
                    "lag_order": k_ar,
                    "p_IMU_to_HR": p_xy,
                    "F_IMU_to_HR": stat_xy,
                    "d_HR": d_hr,
                    "d_IMU": d_imu,
                }
            )

    if not rows:
        print("No series processed.")
        return pd.DataFrame()

    res = pd.DataFrame(rows)

    # Print per-dataset summary with FDR across series
    for ds, sub in res.groupby("dataset"):
        pvals = sub["p_IMU_to_HR"].dropna().values
        if pvals.size == 0:
            print(f"{ds}: no valid Granger results.")
            continue
        sig_mask = _fdr_bh(pvals, alpha=alpha)
        frac_sig = float(sig_mask.mean())
        print(f"\n{ds} — VAR HQIC lag (median): {np.median(sub['lag_order']):.0f}")
        print(
            f"IMU ⇒ HR: median p={np.median(pvals):.3g} | FDR α={alpha}: {100 * frac_sig:.1f}% series significant"
        )

    return res


# ---------------------------------------------------------------------------------------------------------------------
# XGBOOST FEATURE IMPORTANCE
# ---------------------------------------------------------------------------------------------------------------------


def _build_supervised_hr_imu(
    y: NDArray,
    x: NDArray,
    hr_lags: List[int],
    exo_lags: List[int],
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create lagged features for HR (autoregressive) and IMU (exogenous) to predict y[t+horizon].
    Returns X, y_target, feature_names (in the same order as X columns).
    """
    # Build HR lag block
    X_hr = _make_lag_matrix(y, hr_lags)  # shape (T, len(hr_lags))
    hr_names = [f"hr_lag{l}" for l in hr_lags]

    # Build IMU lag block
    X_exo = _make_lag_matrix(x, exo_lags)  # shape (T, len(exo_lags))
    exo_names = [f"imu_level_lag{l}" for l in exo_lags]

    # Target is future HR
    y_target = np.column_stack([np.roll(y, -horizon) for h in range(1, horizon + 1)])

    # Align: we must drop the last `horizon` rows because target is rolled
    T = len(y)
    valid_upto = T - horizon
    X = np.column_stack([X_hr[:valid_upto], X_exo[:valid_upto]])
    y_target = y_target[:valid_upto]

    # Also drop the first max lag so lags are valid
    max_lag = max((hr_lags + exo_lags) or [0])
    X = X[max_lag:, :]
    y_target = y_target[max_lag:]

    feat_names = hr_names + exo_names
    return X, y_target, feat_names


def _temporal_split(X: np.ndarray, y: np.ndarray, train=0.7, val=0.1):
    """
    70/10/20 temporal split.
    """
    n = len(y)
    n_tr = int(n * train)
    n_va = int(n * val)
    X_tr, y_tr = X[:n_tr], y[:n_tr]
    X_va, y_va = X[n_tr : n_tr + n_va], y[n_tr : n_tr + n_va]
    X_te, y_te = X[n_tr + n_va :], y[n_tr + n_va :]
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def _xgb_train_and_importance(
    X_tr,
    y_tr,
    X_va,
    y_va,
    feature_names: List[str],
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=6,
):
    early_stop = xgb.callback.EarlyStopping(
        rounds=20,
        metric_name="rmse",
        data_name="validation_0",
        save_best=True,
        maximize=False,
    )
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
        n_jobs=0,
        eval_metric="rmse",
        callbacks=[early_stop],
    )
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )

    # Importance by "gain" (most informative)
    booster = model.get_booster()
    raw_gain = booster.get_score(
        importance_type="gain"
    )  # dict: feature index name -> gain

    # Map xgb feature names (f0,f1,...) to your feature_names
    fmap = {f"f{i}": name for i, name in enumerate(feature_names)}
    gain_named = {fmap[k]: v for k, v in raw_gain.items() if k in fmap}

    return model, gain_named


def xgb_feature_importance(
    datamodules: List[LightningDataModule],
    hr_lags: List[int],
    exo_lags: List[int],
    horizon: int = 1,
    max_series: Optional[int] = 5,
):
    """
    For each dataset and (up to) `max_series` series:
      - build lagged HR+IMU features
      - temporal 70/10/20 split
      - train XGB and record per-feature gain importance
    Prints per-dataset:
      - top features
      - grouped importance (HR vs IMU)
      - importance aggregated by lag (e.g., lag1, lag2, ...)
    """
    for dm in datamodules:
        print(f"\n[XGB] DATASET: {dm.name}")
        data = dm.train_dataset.data
        agg_gain = Counter()
        agg_gain_hr_vs_exo = Counter()
        agg_gain_by_lag = Counter()

        n_series = len(data) if max_series is None else min(len(data), max_series)
        for si in range(n_series):
            series = data[si]
            y = series[:, 0].astype(float)  # HR
            x = series[:, 1].astype(float)  # IMU (level)

            # build supervised
            X, y_target, feat_names = _build_supervised_hr_imu(
                y, x, hr_lags=hr_lags, exo_lags=exo_lags, horizon=horizon
            )

            if len(y_target) < 70:
                continue
            (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = _temporal_split(X, y_target)

            # train + importance
            _, gain_named = _xgb_train_and_importance(
                X_tr, y_tr, X_va, y_va, feat_names
            )

            # aggregate
            agg_gain.update(gain_named)

            # HR vs IMU buckets
            for name, g in gain_named.items():
                bucket = "HR" if name.startswith("hr_lag") else "IMU"
                agg_gain_hr_vs_exo[bucket] += g
                # by lag number
                lag = int(name.split("lag")[-1])
                agg_gain_by_lag[lag] += g

        # normalize to percentages
        total_gain = sum(agg_gain.values()) or 1.0

        def pct(x):
            return 100.0 * x / total_gain

        print("\nTop features (by total gain across series):")
        for name, g in agg_gain.most_common(15):
            print(f"  {name:>16}: {g:.4f}  ({pct(g):5.1f}%)")

        print("\nGrouped importance (HR vs IMU):")
        for grp in ["HR", "IMU"]:
            g = agg_gain_hr_vs_exo[grp]
            print(f"  {grp:>3}: {g:.4f}  ({pct(g):5.1f}%)")

        print("\nImportance by lag (aggregated over HR+IMU):")
        for lag, g in sorted(agg_gain_by_lag.items()):
            print(f"  lag {lag:>2}: {g:.4f}  ({pct(g):5.1f}%)")


# ---------------------------------------------------------------------------------------------------------------------
# COHERENCE
# ---------------------------------------------------------------------------------------------------------------------


def summarize_coherence(
    datamodules,
    fs=0.5,
    nperseg=256,
    bands_hz=None,  # list of (lo, hi) in Hz; clipped to Nyquist
    dataset_attr_name="name",
):
    """
    datamodules: iterable of objects with .train_dataset.data shaped (N_series, T, C)
                 HR assumed at channel 0, IMU at channel 1
    fs: sampling rate in Hz (0.5 Hz => Nyquist 0.25 Hz)
    nperseg: window length for Welch coherence (auto-capped by series length)
    bands_hz: list of (low, high) in Hz for band-averaged coherence
              If None, uses bands suited for fs=0.5: [(0.00,0.05), (0.05,0.15), (0.15, fs/2)]
    dataset_attr_name: attribute to pull a readable dataset name from the datamodule
    Returns:
      per_series_df: one row per (dataset, series_idx) with band averages
      per_dataset_df: aggregated (nanmean) coherence per band across series
    """
    nyq = fs / 2.0
    if bands_hz is None:
        # fs=0.5 -> Nyquist=0.25. Split into very-low, low, mid bands.
        bands_hz = [(0.00, 0.05), (0.05, 0.15), (0.15, nyq)]
    # Clip band highs to Nyquist and drop empty/invalid bands
    bands_hz = [(lo, min(hi, nyq)) for (lo, hi) in bands_hz if lo < min(hi, nyq)]
    band_names = [f"{lo:.2f}-{hi:.2f}Hz" for lo, hi in bands_hz]

    per_series_rows = []

    for dm_i, dm in enumerate(datamodules):
        ds_name = getattr(dm, dataset_attr_name, f"dataset_{dm_i}")
        data = dm.train_dataset.data  # expected shape: (N, T, C)
        N = len(data)

        for s_idx, series in enumerate(data):
            # series: (T, C), HR in col 0, IMU in col 1
            hr = np.asarray(series[:, 0], dtype=float)
            imu = np.asarray(series[:, 1], dtype=float)

            # clean
            keep = np.isfinite(hr) & np.isfinite(imu)
            hr = hr[keep]
            imu = imu[keep]
            if hr.size < 8:  # too short to say anything
                row = {"dataset": ds_name, "series_idx": s_idx, "n_used": int(hr.size)}
                row.update({bn: np.nan for bn in band_names})
                per_series_rows.append(row)
                continue

            seg = int(min(nperseg, hr.size))  # cap by series length
            # Compute coherence (returns f up to Nyquist)
            f, Cxy = sp_coherence(hr, imu, fs=fs, nperseg=seg)

            row = {"dataset": ds_name, "series_idx": s_idx, "n_used": int(hr.size)}
            for (lo, hi), bn in zip(bands_hz, band_names):
                mask = (f >= lo) & (f < hi) if hi < nyq else (f >= lo) & (f <= hi)
                row[bn] = float(np.nanmean(Cxy[mask])) if np.any(mask) else np.nan
            # overall mean coherence across all bins
            row["overall_mean"] = float(np.nanmean(Cxy)) if Cxy.size else np.nan
            per_series_rows.append(row)

    per_series_df = pd.DataFrame(per_series_rows)

    # Aggregate per dataset
    agg_cols = band_names + ["overall_mean"]
    per_dataset_df = per_series_df.groupby("dataset", as_index=False)[agg_cols].mean(
        numeric_only=True
    )

    return per_series_df, per_dataset_df


def coherence_fn(datamodules):
    per_series_df, per_dataset_df = summarize_coherence(datamodules, fs=0.5)

    print(per_dataset_df)


# ---------------------------------------------------------------------------------------------------------------------
# SCATTER PLOTS
# ---------------------------------------------------------------------------------------------------------------------


def _align_imux_hr_y(hr: np.ndarray, imu: np.ndarray, lag: int):
    """
    Return (imu_aligned, hr_aligned) given lag.
    Convention: positive lag means X (IMU) leads Y (HR) by `lag`.
    """
    if lag < 0:
        # HR leads; drop last -lag from HR, drop first -lag from IMU
        return imu[:lag], hr[-lag:]
    elif lag > 0:
        # IMU leads; drop first lag from HR, drop first lag from IMU's tail
        return imu[lag:], hr[:-lag]
    else:
        return imu, hr


def plot_scatter(datamodules):
    dataset_to_series = {
        "dalia": (2, 1, 8, 1),  # (series_idx, raw_lag, diff_lag, bj_lag)
        "wildppg": (0, 0, 1, 0),
        "ieee": (0, 1, 9, 3),
    }

    fig = make_subplots(
        rows=3,
        cols=3,
        column_titles=["No Processing", "Differenced", "Pre-Whitened"],
        row_titles=[dm.name for dm in datamodules],
    )

    for j, dm in enumerate(datamodules, start=1):
        data = dm.train_dataset.data
        s_idx, n_l, d_l, bj_l = dataset_to_series[dm.name]
        series = data[s_idx]
        hr = series[:, 0].astype(float)
        imu = series[:, 1].astype(float)

        # 1) Raw
        x0, y0 = _align_imux_hr_y(hr=hr, imu=imu, lag=n_l)
        if len(x0) and len(y0):
            fig.add_trace(
                go.Scattergl(
                    x=x0,
                    y=y0,
                    mode="markers",
                    marker=dict(size=3, opacity=0.5),
                    name=f"{dm.name} s{s_idx} (lag={n_l})",
                    showlegend=False,
                ),
                row=j,
                col=1,
            )

        # 2) Differenced
        hr_diff = np.diff(hr, n=1)
        imu_diff = np.diff(imu, n=1)
        x1, y1 = _align_imux_hr_y(hr=hr_diff, imu=imu_diff, lag=d_l)
        if len(x1) and len(y1):
            fig.add_trace(
                go.Scattergl(
                    x=x1,
                    y=y1,
                    mode="markers",
                    marker=dict(size=3, opacity=0.5),
                    name=f"{dm.name} s{s_idx} Δ (lag={d_l})",
                    showlegend=False,
                ),
                row=j,
                col=2,
            )

        # 3) BJ pre-whitened (fit on IMU, filter both; IMU on x-axis, HR on y-axis)
        try:
            imu_pw, hr_pw, _ = _bj_prewhiten_filter_xy_stable(imu, hr)
        except NameError:
            # fallback if you used the non-*_stable name
            imu_pw, hr_pw, _ = _bj_prewhiten_filter_xy(imu, hr)

        x2, y2 = _align_imux_hr_y(hr=hr_pw, imu=imu_pw, lag=bj_l)
        if len(x2) and len(y2):
            fig.add_trace(
                go.Scattergl(
                    x=x2,
                    y=y2,
                    mode="markers",
                    marker=dict(size=3, opacity=0.5),
                    name=f"{dm.name} s{s_idx} BJ (lag={bj_l})",
                    showlegend=False,
                ),
                row=j,
                col=3,
            )

    # Axis titles
    for c in (1, 2, 3):
        fig.update_xaxes(title_text="IMU", row=3, col=c)
    for r in (1, 2, 3):
        fig.update_yaxes(title_text="HR", row=r, col=1)

    fig.update_layout(height=900, width=1200, title_text="IMU vs HR at chosen lags")
    fig.show()


# Conditional Mutual Information


def make_supervised_windows(hr, imu, L=30, H=3, imu_lags=1):
    """
    hr, imu: 1D numpy arrays (same length) for a single subject/sequence
    L: lookback length used as HR_past
    H: forecast horizon steps ahead (e.g., H=1,3,5,...)
    imu_lags: number of causal IMU lags to include (1 => current t)

    Returns:
      Z: (N, L)   HR past (t-L+1 ... t)
      X: (N, imu_lags)  IMU lags (t, t-1, ..., t-imu_lags+1)
      y: (N,)  HR future at t+H
    """
    assert hr.ndim == 1 and imu.ndim == 1 and len(hr) == len(imu)
    T = len(hr)
    max_lag = max(L, imu_lags - 1)
    # last usable t is T-H-1 ; first usable t is max_lag
    t_start = max_lag
    t_end = T - H - 1
    if t_end < t_start:
        return np.empty((0, L)), np.empty((0, imu_lags)), np.empty((0,))

    idx = np.arange(t_start, t_end + 1)
    N = len(idx)

    # HR past Z
    Z = np.stack([hr[idx - k] for k in range(L, 0, -1)], axis=1)  # shape (N, L)

    # IMU lags X (current to past)
    X = np.stack([imu[idx - k] for k in range(imu_lags)], axis=1)  # (N, imu_lags)

    # target y = future HR at t+H
    y = hr[idx + H]
    return Z, X, y


def conditional_mutual_info(
    hr, imu, L=30, H=3, imu_lags=1, n_neighbors=5, random_state=0
):
    """
    Estimate I(IMU ; HR_future | HR_past) via difference of mutual infos:
      I([IMU,Z]; y) - I(Z; y)
    using sklearn's mutual_info_regression (kNN estimator).
    Returns: cmi_estimate (nats), N_effective
    """
    Z, X, y = make_supervised_windows(hr, imu, L=L, H=H, imu_lags=imu_lags)
    if len(y) == 0:
        return np.nan, 0

    # Concatenate features
    ZX = np.concatenate([Z, X], axis=1)

    # sklearn returns MI in nats (natural log base)
    I_ZY = mutual_info_regression(
        Z, y, n_neighbors=n_neighbors, random_state=random_state
    )
    I_ZY_total = float(np.sum(I_ZY))  # sum over Z dims approximates I(Z; y)

    I_ZX_Y = mutual_info_regression(
        ZX, y, n_neighbors=n_neighbors, random_state=random_state
    )
    I_ZX_Y_total = float(np.sum(I_ZX_Y))  # approximates I([Z,X]; y)

    cmi = I_ZX_Y_total - I_ZY_total
    # CMI can't be < 0 theoretically; clip tiny negatives due to estimation noise
    return max(0.0, cmi), len(y)


def cmi_averaged(datamodules):
    for dm in datamodules:
        data = dm.train_dataset.data
        cmis = []
        for series in tqdm(data):
            hr = series[:, 0]
            imu = series[:, 1]
            cmi, _ = conditional_mutual_info(hr, imu)
            cmis.append(cmi)
        print(f"Dataset: {dm.name}")
        print(f"Average CMI: {np.mean(cmis)}")


def main():
    datamodules: List[LightningDataModule] = []
    for dataset in ["dalia", "wildppg", "ieee"]:
        with initialize(version_base=None, config_path="../config/"):
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

    # statistical_performance_difference(datamodules)
    # cross_correlation(datamodules)
    # granger_imu_to_hr(datamodules)

    HR_LAGS = [1, 2, 3, 4, 5, 10, 15, 20, 30]
    IMU_LAGS = [1, 2, 3, 5, 10]
    # xgb_feature_importance(
    #     datamodules, hr_lags=HR_LAGS, exo_lags=IMU_LAGS, horizon=3, max_series=20
    # )
    # plot_scatter(datamodules)
    # coherence_fn(datamodules)
    cmi_averaged(datamodules)


if __name__ == "__main__":
    main()
