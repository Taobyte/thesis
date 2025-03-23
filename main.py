import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
import lightning as L
import wandb

from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.models.mlp import MLPBaseline
from src.datasets import Mode, DaLiADataset


@hydra.main(version_base="1.2", config_path="config", config_name="config.yaml")
def main(config: DictConfig):
    print(config)
    L.seed_everything(config.seed)

    wandb.init(
        project="thesis",
        entity="ckeusch",  # <-- your personal username
    )

    wandb_logger = WandbLogger(project="thesis")

    train_dl = DataLoader(
        DaLiADataset(
            look_back_window=config.look_back_window,
            prediction_window=config.prediction_window,
            mode=Mode.TRAIN,
            **config.dataset,
        ),
        batch_size=32,
        num_workers=4,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        DaLiADataset(
            look_back_window=config.look_back_window,
            prediction_window=config.prediction_window,
            mode=Mode.VAL,
            **config.dataset,
        ),
        batch_size=32,
    )
    print(len(val_dl))
    mlp_baseline = MLPBaseline(
        config.look_back_window,
        config.prediction_window,
        config.model.hidden_channels,
        config.model.dropout,
    )
    trainer = L.Trainer(logger=wandb_logger)
    trainer.fit(model=mlp_baseline, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    main()
