import argparse

from src.constants import MODELS

from src.wandb_results.horizon_ablation import (
    horizon_exo_difference,
    pw_ablation_table,
    best_model_viz_horizon_ablation,
)
from src.wandb_results.look_back_ablation import (
    visualize_look_back_window_difference,
    best_model_viz_lbw_ablation,
)
from src.wandb_results.exo_norm_ablation import (
    exo_norm_ablation_table,
    exo_norm_ablation_heatmap,
)
from local_global_exo_diff import local_global_diff
from src.wandb_results.normalization_ablation import plot_normalization_table
from src.wandb_results.efficiency import plot_efficiency_table
from src.wandb_results.utils import create_params_file_from_optuna
from src.wandb_results.mitbih_results import plot_mitbih_metric_table
from src.wandb_results.qualitative_results import (
    plot_predictions,
    plot_best_exo_improvement,
)


def main():
    parser = argparse.ArgumentParser(description="WandB Results")

    def list_of_ints(arg: str) -> list[int]:
        return [int(i) for i in arg.split(",")]

    def list_of_strings(arg: str) -> list[str]:
        return arg.split(",")

    parser.add_argument(
        "--type",
        type=str,
        choices=[
            "norm_ablation",
            "viz",
            "best_lbw_viz",
            "horizon_ablation",
            "horizon_viz",
            "horizon_exo_diff",
            "exo_norm_ablation_table",
            "exo_norm_ablation_heatmap",
            "local_global",
            "efficiency",
            "mitbih",
            "predictions",
            "exo_endo_preds",
            "optuna",
        ],
        required=True,
        default="table",
        help="Plot type. Must be either 'table', 'viz' or 'activity_ablation' .",
    )

    parser.add_argument(
        "--dataset",
        type=list_of_strings,
        required=False,
        default=["dalia", "wildppg", "ieee"],
        help="Dataset must be one of the following: dalia, ieee, wildppg, chapman, ucihar, usc, capture24",
    )

    parser.add_argument(
        "--look_back_window",
        type=list_of_ints,
        required=False,
        default=[5],
        help="Lookback window size",
    )

    parser.add_argument(
        "--prediction_window",
        type=list_of_ints,
        required=False,
        default=[3],
        help="Prediction window size",
    )

    parser.add_argument(
        "--models",
        required=False,
        default=MODELS,
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

    parser.add_argument(
        "--baselines",
        required=False,
        action="store_true",
        help="Use Baseline models.",
    )

    parser.add_argument(
        "--dls",
        required=False,
        action="store_true",
        help="Use DL models.",
    )

    parser.add_argument(
        "--is_local",
        required=False,
        action="store_true",
        help="Process local datasets for horizon ablation.",
    )

    parser.add_argument(
        "--metric",
        type=str,
        required=False,
        default="MASE",
        help="Target metric for the plots.",
    )

    args = parser.parse_args()
    if args.type == "norm_ablation":
        plot_normalization_table(
            datasets=args.dataset,
            prediction_window=args.prediction_window,
            models=args.models,
            start_time="2025-10-08",
            use_std=args.use_std,
        )
    elif args.type == "viz":
        visualize_look_back_window_difference(
            args.dataset,
            args.look_back_window,
            args.prediction_window,
            start_time="2025-10-08T12:00:00Z",
            save_html=args.save_html,
            use_std=args.use_std,
            models=args.models,
        )
    elif args.type == "best_lbw_viz":
        best_model_viz_lbw_ablation(
            args.dataset,
            args.models,
            args.look_back_window,
            args.prediction_window,
            metric=args.metric,
            start_time="2025-10-08T12:00:00Z",
            use_std=args.use_std,
        )
    elif args.type == "horizon_ablation":
        pw_ablation_table(
            args.dataset,
            args.models,
            args.look_back_window,
            args.prediction_window,
            metric=args.metric,
            start_time="2025-09-10",
            baselines=args.baselines,
            dls=args.dls,
            is_local=args.is_local,
        )
    elif args.type == "horizon_exo_diff":
        horizon_exo_difference(
            args.dataset,
            args.models,
            args.look_back_window,
            args.prediction_window,
            start_time="2025-09-10",
        )
    elif args.type == "horizon_viz":
        best_model_viz_horizon_ablation(
            args.dataset,
            args.models,
            args.look_back_window,
            args.prediction_window,
            metric=args.metric,
            start_time="2025-09-10",
            use_std=args.use_std,
        )

    elif args.type == "exo_norm_ablation_table":
        exo_norm_ablation_table(
            args.dataset,
            args.models,
            args.look_back_window,
            args.prediction_window,
            metric=args.metric,
            start_time="2025-09-10",
            baselines=args.baselines,
            dls=args.dls,
        )
    elif args.type == "exo_norm_ablation_heatmap":
        exo_norm_ablation_heatmap(
            args.dataset,
            args.models,
            args.look_back_window,
            args.prediction_window,
            metric=args.metric,
            start_time="2025-09-10",
        )
    elif args.type == "local_global":
        local_global_diff(
            args.dataset,
            args.models,
            args.look_back_window,
            args.prediction_window,
            start_time="2025-09-17",
        )
    elif args.type == "efficiency":
        plot_efficiency_table(
            dataset=args.dataset,
            look_back_window=args.look_back_window,
            prediction_window=args.prediction_window,
            models=args.models,
        )
    elif args.type == "mitbih":
        plot_mitbih_metric_table()
    elif args.type == "predictions":
        plot_predictions(
            args.dataset,
            args.look_back_window,
            args.prediction_window,
            args.models,
            args.experiment,
        )
    elif args.type == "exo_endo_preds":
        plot_best_exo_improvement(
            args.dataset,
            args.look_back_window,
            args.prediction_window,
            args.models,
        )
    elif args.type == "optuna":
        create_params_file_from_optuna(
            models=args.models, start_time="2025-8-31T21:30:00+02:00"
        )


if __name__ == "__main__":
    main()
