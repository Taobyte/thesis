import time
import json
import math
import numpy as np
import torch
import wandb

from pathlib import Path
from numpy.typing import NDArray
from lightning import Callback
from lightning.pytorch.loggers import WandbLogger


class EfficiencyCallback(Callback):
    def __init__(self, latency_warmup=50, latency_iters=500, log_prefix=""):
        self.latency_warmup = latency_warmup
        self.latency_iters = latency_iters
        self.log_prefix = log_prefix
        # state
        self._fit_t0 = None
        self._epoch_t0 = None
        self.epoch_times = []
        self.best_epoch = None
        self.best_val = math.inf
        self.peak_vram_bytes_train = 0
        self.peak_vram_bytes_infer = 0

    # ---- training time & peak VRAM (training) ----
    def on_fit_start(self, trainer, pl_module):
        self._fit_t0 = time.perf_counter()
        self.epoch_times.clear()
        self.best_epoch = None
        self.best_val = math.inf
        self.peak_vram_bytes_train = 0
        self.peak_vram_bytes_infer = 0

    def on_train_epoch_start(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(pl_module.device)
        self._epoch_t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        sec = time.perf_counter() - self._epoch_t0
        self.epoch_times.append(sec)
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_reserved(pl_module.device)
            self.peak_vram_bytes_train = max(self.peak_vram_bytes_train, peak)
        # log per-epoch time
        pl_module.log(
            self.log_prefix + "train_seconds_per_epoch",
            sec,
            prog_bar=False,
            logger=True,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        cm = trainer.callback_metrics
        key = "val_loss_epoch"
        val = float(cm[key])
        if val < self.best_val:
            self.best_val = val
            self.best_epoch = int(trainer.current_epoch)

    def on_fit_end(self, trainer, pl_module):
        total_sec = time.perf_counter() - self._fit_t0
        med_epoch = (
            float(np.median(self.epoch_times)) if self.epoch_times else float("nan")
        )
        num_devices = getattr(trainer, "num_devices", 1) or 1
        gpu_hours = (
            total_sec / 3600.0
        ) * num_devices  # good proxy when using 1 GPU per job

        # ---- inference latency (single-window) ----
        latency_ms = float("nan")
        if hasattr(trainer, "datamodule") and trainer.datamodule is not None:
            trainer.datamodule.setup("test")
            try:
                dl = trainer.datamodule.test_dataloader()
                batch = next(iter(dl))
                _, x, _ = batch
                device = pl_module.device
                x = x.to(device, non_blocking=True) if torch.is_tensor(x) else x
                pl_module.eval()
                with torch.no_grad():
                    # warm-up
                    for _ in range(self.latency_warmup):
                        _ = pl_module.model_forward(x)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(self.latency_iters):
                        _ = pl_module.model_forward(x)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    latency_ms = (
                        (time.perf_counter() - t0) * 1000.0 / self.latency_iters
                    )
                    # peak VRAM during inference
                    if torch.cuda.is_available():
                        self.peak_vram_bytes_infer = max(
                            self.peak_vram_bytes_infer,
                            torch.cuda.max_memory_reserved(device),
                        )
            except Exception as e:
                print(f"[EfficiencyCallback] Latency measurement skipped: {e}")

        # convert & log
        B = 1024**3
        metrics = {
            self.log_prefix + "params": sum(
                p.numel() for p in pl_module.parameters() if p.requires_grad
            ),
            self.log_prefix + "peak_vram_gib_train": self.peak_vram_bytes_train / B,
            self.log_prefix + "peak_vram_gib_infer": self.peak_vram_bytes_infer / B,
            self.log_prefix + "train_seconds_per_epoch_med": med_epoch,
            self.log_prefix + "epochs_to_best": (
                self.best_epoch if self.best_epoch is not None else -1
            ),
            self.log_prefix + "time_to_best_min": (
                sum(self.epoch_times[: self.best_epoch + 1]) / 60.0
                if self.best_epoch is not None
                else float("nan")
            ),
            self.log_prefix + "total_fit_minutes": total_sec / 60.0,
            self.log_prefix + "gpu_hours": gpu_hours,
            self.log_prefix + "inference_ms_per_window": latency_ms,
        }
        # prefer logger, else print
        try:
            if trainer.logger is not None:
                trainer.logger.log_metrics(metrics, step=trainer.global_step)
        except Exception:
            pass
        print(
            "[Efficiency]",
            {k.replace(self.log_prefix, ""): v for k, v in metrics.items()},
        )


class PredictionCallback(Callback):
    def __init__(self, filename: str = "test_predictions.npz"):
        self.filename = filename

    def on_test_end(self, trainer, pl_module):
        datamodule = trainer.datamodule
        test_dataset = datamodule.test_dataset
        ground_truth: list[NDArray[np.float32]] = test_dataset.data

        dl = datamodule.test_dataloader()
        pl_module.eval()
        device = pl_module.device
        predictions: list[NDArray[np.float32]] = []
        with torch.no_grad():
            for look_back_window, _ in dl:
                look_back_window = look_back_window.to(device, non_blocking=True)
                preds = pl_module.model_forward(look_back_window)
                denorm_preds = pl_module._denormalize_tensor(preds)
                preds_numpy = denorm_preds.detach().cpu().float().numpy()
                predictions.append(preds_numpy)

        stacked_preds = np.concatenate(predictions, axis=0).astype(np.float32)
        metrics: dict[str, list[float]] = pl_module.metric_full
        result_arrays = {
            "preds": stacked_preds,  # (N, H, C)
            "gt_series": np.asarray(ground_truth, dtype=object),  # ragged → object
        }

        outdir = Path(trainer.default_root_dir) / "predictions"
        outdir.mkdir(parents=True, exist_ok=True)
        npz_path = outdir / self.filename
        np.savez_compressed(npz_path, **result_arrays)

        metrics_path = outdir / (npz_path.stem + "_metrics.json")

        with open(metrics_path, "w") as f:
            json.dump(metrics, f)  # metrics is your dict[str, list[float]]

        # Log to W&B
        wb_logger = trainer.logger
        assert isinstance(wb_logger, WandbLogger)
        run = wb_logger.experiment

        art = wandb.Artifact(f"test_predictions-{run.id}", type="predictions")
        art.add_file(str(npz_path))
        art.add_file(str(metrics_path))
        run.log_artifact(art)
