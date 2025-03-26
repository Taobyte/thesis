import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

import lightning as L
from lightning.pytorch.loggers import WandbLogger


@hydra.main(version_base="1.2", config_path="config", config_name="config.yaml")
def main(config: DictConfig):
    print(config)
    L.seed_everything(config.seed)

    wandb_logger = WandbLogger(project="thesis")

    datamodule = instantiate(config.dataset.datamodule)
    model = instantiate(config.model.model)
    pl_model = instantiate(config.model.pl_model, model=model)

    trainer = L.Trainer(logger=wandb_logger)
    trainer.fit(pl_model, datamodule=datamodule)


if __name__ == "__main__":
    main()
