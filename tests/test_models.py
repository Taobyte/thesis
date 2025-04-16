import pytest
import lightning as L
from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval)
max_epochs = 50
freq = 64
l_b_w_ppg = [3 * freq, 4 * freq, 5 * freq]
p_w_ppg = [1 * freq, 2 * freq, 3 * freq]

l_b_w_hr = [10, 15, 20]
p_w_hr = [1, 2, 3]
# l_b_w_hr = [10]
# p_w_hr = [1]


def _test_model(model_name: str, look_back_window: int, prediction_window: int):
    overrides = ["dataset=dalia"]
    overrides += [
        f"look_back_window={look_back_window}",
        f"prediction_window={prediction_window}",
    ]
    overrides += ["dataset.datamodule.use_heart_rate=True"]
    overrides += [f"model={model_name}"]
    overrides += ["model.pl_model.learning_rate=0.0001"]
    with initialize(version_base=None, config_path="../config/"):
        config = compose(config_name="config", overrides=overrides)
    datamodule = instantiate(
        config.dataset.datamodule,
        batch_size=1,
    )
    model = instantiate(config.model.model)

    pl_model = instantiate(config.model.pl_model, model=model)
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


def test_elastst():
    _test_model("elastst")


def test_timesnet():
    _test_model("timesnet")


@pytest.mark.parametrize("look_back_window", l_b_w_hr)
@pytest.mark.parametrize("prediction_window", p_w_hr)
def test_adamshyper(look_back_window, prediction_window):
    _test_model("adamshyper", look_back_window, prediction_window)


@pytest.mark.parametrize("look_back_window", l_b_w_hr)
@pytest.mark.parametrize("prediction_window", p_w_hr)
def test_simpletm(look_back_window, prediction_window):
    _test_model("simpletm", look_back_window, prediction_window)
