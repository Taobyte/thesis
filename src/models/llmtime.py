import lightning as L


class LLMTime(L.LightningModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def train_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        return None
