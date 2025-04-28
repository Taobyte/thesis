import pytest
import lightning as L
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.utils import compute_square_window

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("compute_square_window", compute_square_window)


max_epochs = 50
freq = 64
l_b_w_ppg = [3 * freq, 4 * freq, 5 * freq]
p_w_ppg = [1 * freq, 2 * freq, 3 * freq]

l_b_w_hr = [3]
p_w_hr = [1]
# models = ["elastst", "simpletm", "adamshyper", "timesnet", "gpt4ts", "pattn", "timellm"]
models = ["timellm"]


def _test_model_overfitting(
    model_name: str,
    look_back_window: int,
    prediction_window: int,
):
    overrides = ["dataset=dalia"]
    overrides += [
        f"look_back_window={look_back_window}",
        f"prediction_window={prediction_window}",
    ]
    overrides += ["use_heart_rate=True"]
    overrides += [f"model={model_name}"]
    with initialize(version_base=None, config_path="../config/"):
        config = compose(config_name="config", overrides=overrides)
    datamodule = instantiate(
        config.dataset.datamodule,
        batch_size=1,
    )
    datamodule.setup("fit")
    model = instantiate(config.model.model)

    pl_model = instantiate(
        config.model.pl_model,
        model=model,
        global_mean=datamodule.train_dataset.global_mean,
        global_std=datamodule.train_dataset.global_std,
    )
    trainer = L.Trainer(
        max_epochs=max_epochs,
        overfit_batches=1,
        enable_progress_bar=True,
    )
    trainer.fit(pl_model, datamodule=datamodule)

    final_train_loss = trainer.callback_metrics.get("train_loss")
    print(
        f"[{model_name}] lbw={look_back_window}, pw={prediction_window} → Final loss: {final_train_loss}"
    )

    assert final_train_loss is not None and final_train_loss < 0.3, (
        "Model failed to overfit"
    )


def _test_model_inference(
    model_name: str,
    look_back_window: int,
    prediction_window: int,
    use_activity_info: bool = False,
):
    overrides = ["dataset=dalia"]
    overrides += [
        f"look_back_window={look_back_window}",
        f"prediction_window={prediction_window}",
    ]
    overrides = [f"use_activity_info={str(use_activity_info)}"]
    overrides += ["use_heart_rate=True"]
    overrides += [f"model={model_name}"]
    with initialize(version_base=None, config_path="../config/"):
        config = compose(config_name="config", overrides=overrides)
    datamodule = instantiate(
        config.dataset.datamodule,
        batch_size=1,
    )

    model = instantiate(config.model.model)

    pl_model = instantiate(
        config.model.pl_model,
        model=model,
    )
    trainer = L.Trainer(fast_dev_run=1)
    trainer.fit(pl_model, datamodule=datamodule)


@pytest.mark.parametrize("look_back_window", l_b_w_hr)
@pytest.mark.parametrize("prediction_window", p_w_hr)
@pytest.mark.parametrize("model", models)
def test_all_inference(look_back_window, prediction_window, model):
    _test_model_inference(model, look_back_window, prediction_window, True)


def test_elastst():
    _test_model_overfitting("elastst")


def test_timesnet():
    _test_model_overfitting("timesnet")


@pytest.mark.parametrize("look_back_window", l_b_w_hr)
@pytest.mark.parametrize("prediction_window", p_w_hr)
def test_adamshyper(look_back_window, prediction_window):
    _test_model_overfitting("adamshyper", look_back_window, prediction_window)


@pytest.mark.parametrize("look_back_window", l_b_w_hr)
@pytest.mark.parametrize("prediction_window", p_w_hr)
def test_simpletm(look_back_window, prediction_window):
    _test_model_overfitting("simpletm", look_back_window, prediction_window)
