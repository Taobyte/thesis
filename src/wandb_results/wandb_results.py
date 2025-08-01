import argparse

from src.wandb_results.metric_table import plot_tables
from src.wandb_results.activity_ablation import dynamic_feature_ablation
from src.wandb_results.look_back_ablation import visualize_look_back_window_difference
from src.wandb_results.normalization_ablation import visualize_normalization_difference
from src.wandb_results.utils import create_params_file_from_optuna


def main():
    parser = argparse.ArgumentParser(description="WandB Results")

    def list_of_ints(arg):
        return [int(i) for i in arg.split(",")]

    def list_of_strings(arg):
        return arg.split(",")

    parser.add_argument(
        "--type",
        type=str,
        choices=["table", "viz", "activity_ablation", "norm_ablation", "optuna"],
        required=True,
        default="table",
        help="Plot type. Must be either 'table', 'viz' or 'activity_ablation' .",
    )

    parser.add_argument(
        "--dataset",
        type=list_of_strings,
        required=True,
        default=["dalia"],
        help="Dataset must be one of the following: dalia, ieee, wildppg, chapman, ucihar, usc, capture24",
    )

    parser.add_argument(
        "--normalization",
        type=str,
        choices=[
            "none",
            "global",
            "local",
        ],
        required=False,
        default=None,
        help="Normalization must be 'none', 'global' or 'local' ",
    )

    parser.add_argument(
        "--look_back_window",
        type=list_of_ints,
        required=True,
        default=[5],
        help="Lookback window size",
    )

    parser.add_argument(
        "--prediction_window",
        type=list_of_ints,
        required=True,
        default=[3],
        help="Prediction window size",
    )

    parser.add_argument(
        "--use_heart_rate",
        required=False,
        action="store_true",
        help="get runs for heart rate only has an effect for datasets: dalia, ieee & wildppg",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=False,
        default="endo_exo",
        help="experiment name",
    )

    parser.add_argument(
        "--models",
        required=False,
        default=None,
        type=list_of_strings,
        help="Pass in the models you want to visualize the prediction and look back window. Must be separated by commas , without spaces between the model names! (Correct Example: timesnet,xgboost| Wrong Example: gpt4ts, timellm )",
    )

    parser.add_argument(
        "--save_html",
        required=False,
        action="store_true",
        help="save html plot for sharing",
    )

    parser.add_argument(
        "--window_statistic",
        choices=["mean", "var", "power"],
        required=False,
        default=None,
        type=str,
        help="Which window statistic for the heartrate value to use. Can be 'mean', 'var' or 'power'. Only supported by DaLia dataset at the moment.",
    )

    parser.add_argument(
        "--use_std",
        required=False,
        action="store_true",
        help="plot standard deviation",
    )
    parser.add_argument(
        "--table",
        required=False,
        action="store_true",
        help="Plot table.",
    )

    args = parser.parse_args()

    if args.type == "table":
        plot_tables(
            args.dataset,
            args.look_back_window,
            args.prediction_window,
            args.use_heart_rate,
            args.experiment,
        )
    elif args.type == "viz":
        visualize_look_back_window_difference(
            args.dataset,
            args.models,
            args.look_back_window,
            args.prediction_window,
            args.use_heart_rate,
            args.use_dynamic_features,
            args.use_static_features,
            args.normalization,
            save_html=args.save_html,
            start_time="2025-6-24",
            use_std=args.use_std,
        )
    elif args.type == "activity_ablation":
        dynamic_feature_ablation(
            datasets=args.dataset,
            models=args.models,
            look_back_window=args.look_back_window,
            prediction_window=args.prediction_window,
            use_heart_rate=args.use_heart_rate,
            start_time="2025-7-14",
            normalization=args.normalization,
            save_html=args.save_html,
            window_statistic=args.window_statistic,
            use_std=args.use_std,
            table=args.table,
        )
    elif args.type == "norm_ablation":
        visualize_normalization_difference(
            args.dataset,
            args.models,
            args.look_back_window,
            args.prediction_window,
            args.use_heart_rate,
            args.use_dynamic_features,
            args.use_static_features,
            save_html=args.save_html,
        )

    elif args.type == "optuna":
        create_params_file_from_optuna(models=args.models, start_time="2025-6-24")


if __name__ == "__main__":
    main()
