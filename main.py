import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import torch
import lightning as L
import wandb

from lightning.pytorch.loggers import WandbLogger


@hydra.main(version_base="1.2", config_path="config", config_name="config.yaml")
def main(config: DictConfig):
    print(config)
    L.seed_everything(config.seed)

    datamodule = instantiate(config.dataset.datamodule)


if __name__ == "__main__":
    main()
