# Metrics
test_metrics = ["MSE", "MAE", "DIRACC", "MASE", "ND", "NRMSE"]

# Colors for plotting
model_colors = [
    "blue",
    "green",
    "orange",
    "red",
    "purple",
    "pink",
    "brown",
    "olive",
    "cyan",
    "magenta",
    "lime",
    "teal",
    "navy",
    "gold",
    "gray",
]

# Human-readable names for metrics
metric_to_name = {
    "MSE": "Mean Squared Error",
    "MAE": "Mean Absolute Deviation",
    "DIRACC": "Directional Accuracy",
    "MASE": "Mean Absolute Percentage Error",
    "ND": "Normalized Difference",
    "NRMSE": "Normalized RMSE",
}

# Human-readable names for models
model_to_name = {
    "timesnet": "TimesNet",
    "gpt4ts": "GPT4TS",
    "adamshyper": "AdaMSHyper",
    "timellm": "TimeLLM",
    "pattn": "PAttn",
    "simpletm": "SimpleTM",
    "elastst": "ElasTST",
    "timexer": "TimeXer",
    "gp": "Gaussian Process",
    "bnn": "Bayesian Neural Network",
    "kalmanfilter": "Kalman Filter",
    "linear": "Linear Regression",
    "hmm": "Hidden Markov Model",
    "xgboost": "XGBoost",
    "mlp": "MLP",
    "hlinear": "HLinear",
    "hxgboost": "HXGBoost",
}

# Human-readable names for datasets
dataset_to_name = {
    "dalia": "DaLiA",
    "wildppg": "WildPPG",
    "ieee": "IEEE",
    "mhc6mwt": "My Heart Counts Six-Minute Walk Test",
}

# Human-readable names for experiments
experiment_to_name = {
    "endo_only": "Endogenous Only",
    "endo_exo": "Endogenous & Exogenous",
}
