from typing import List

# Metrics
METRICS = ["MSE", "MAE", "DIRACC", "MASE", "ND", "NRMSE", "SMAPE"]

# Human-readable names for metrics
metric_to_name = {
    "MSE": "Mean Squared Error",
    "MAE": "Mean Absolute Deviation",
    "DIRACC": "Directional Accuracy",
    "MASE": "Mean Absolute Scaled Error",
    "ND": "Normalized Difference",
    "NRMSE": "Normalized RMSE",
    "SMAPE": "Symmetric MAPE",
}

# Models
MODELS: List[str] = [
    "linear",
    "mole",
    "msar",
    "kalmanfilter",
    "xgboost",
    "gp",
    "mlp",
    "timesnet",
    "simpletm",
    "adamshyper",
    "patchtst",
    "timexer",
    "gpt4ts",
    "nbeatsx",
]

BASELINES: List[str] = [
    "linear",
    "msar",
    "mole",
    "kalmanfilter",
    "xgboost",
    "gp",
    "mlp",
]

DL: List[str] = [
    "timesnet",
    "simpletm",
    "adamshyper",
    "patchtst",
    "timexer",
    "gpt4ts",
    "nbeatsx",
]

# Colors for plotting
model_colors = {
    "linear": "#4477AA",
    "mole": "#EE6677",
    "msar": "#228833",
    "kalmanfilter": "#CCBB44",
    "gp": "#66CCEE",
    "xgboost": "#AA3377",
    "mlp": "#BBBBBB",
    "timesnet": "#332288",
    "simpletm": "#88CCEE",
    "adamshyper": "#44AA99",
    "patchtst": "#117733",
    "timexer": "#999933",
    "gpt4ts": "#DDCC77",
    "nbeatsx": "#CC6677",
    "dl": "#0072B2",
    "baseline": "#D55E00",
}

dataset_colors = {
    "dalia": "#7F3C8D",
    "wildppg": "#11A579",
    "ieee": "#E73F74",
}


# Human-readable names for models
model_to_name = {
    "timesnet": "TimesNet",
    "gpt4ts": "GPT4TS",
    "adamshyper": "AdaMSHyper",
    "simpletm": "SimpleTM",
    "patchtst": "PatchTST",
    "timexer": "TimeXer",
    "nbeatsx": "NBeatsX",
    "gp": "Gaussian Process",
    "kalmanfilter": "Kalman Filter",
    "linear": "Linear Regression",
    "msar": "MSAR",
    "mole": "MoLE",
    "xgboost": "XGBoost",
    "mlp": "MLP",
}

model_to_abbr = {
    "timesnet": "TNET",
    "gpt4ts": "GPT4TS",
    "adamshyper": "AMSH",
    "simpletm": "STM",
    "patchtst": "PTST",
    "timexer": "TXER",
    "nbeatsx": "NBX",
    "gp": "GP",
    "kalmanfilter": "KF",
    "linear": "LR",
    "msar": "MSAR",
    "mole": "MoLE",
    "xgboost": "XGB",
    "mlp": "MLP",
}

# Human-readable names for datasets
dataset_to_name = {
    "dalia": "DaLiA",
    "wildppg": "WildPPG",
    "ieee": "IEEE",
    "ldalia": "Dalia",
    "lwildppg": "LWildPPG",
    "lieee": "LIEEE",
}


# Human-readable names for experiments
experiment_to_name = {
    "endo_only": "Endogenous Only",
    "endo_exo": "Endogenous & Exogenous",
}
