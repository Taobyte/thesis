import sys
from pathlib import Path

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, root_dir)

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm


def test_dataset():
    with initialize(version_base=None, config_path="../../config/"):
        cfg = compose(config_name="config", overrides=["dataset=dalia"])

    datamodule = instantiate(cfg.dataset)["datamodule"]
    print(datamodule)
    datamodule.setup("fit")
    test_dl = datamodule.val_dataloader()

    for batch in tqdm(test_dl):
        x, y = batch
