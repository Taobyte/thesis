def plot_efficiency_table(
    dataset: list[str] = ["dalia"],
    look_back_window: int = 30,
    prediction_window: int = 3,
    models: list[str] = [],
):
    eff_metrics = [
        "params",
        "peak_vram_gib_train",
        "peak_vram_gib_infer",
        "train_seconds_per_epoch_med",
        "epochs_to_best",
        "time_to_best_min",
        "total_fit_minutes",
        "inference_ms_per_window",
    ]

    pass
