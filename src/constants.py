# Metrics
test_metrics = ["test_MSE", "test_MAE", "test_cross_correlation", "test_dir_acc_single"]

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
    "test_MSE": "Mean Squared Error",
    "test_MAE": "Mean Absolute Deviation",
    "test_cross_correlation": "Cross Correlation",
    "test_dir_acc_single": "Directional Accuracy",
}

name_to_title = {
    "MSE": "MSE",
    "MAE": "MAE",
    "cross_correlation": "Pearson Correlation",
    "dir_acc_full": "Dir Acc Full",
    "dir_acc_single": "Dir Acc Single",
    "sMAPE": "sMAPE",
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
    "gp": "Gaussian Process",
    "bnn": "Bayesian Neural Network",
    "kalmanfilter": "Kalman Filter",
    "linear": "Linear Regression",
    "hmm": "Hidden Markov Model",
    "xgboost": "XGBoost",
}

# Human-readable names for datasets
dataset_to_name = {
    "dalia": "DaLiA",
    "wildppg": "WildPPG",
    "ieee": "IEEE",
    "mhc6mwt": "My Heart Counts Six-Minute Walk Test",
}
