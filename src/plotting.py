import torch
import matplotlib.pyplot as plt
import wandb
from lightning.pytorch.loggers import WandbLogger


def plot_prediction_wandb(
    x: torch.Tensor, y: torch.Tensor, preds: torch.Tensor, wandb_logger: WandbLogger
):
    look_back_window = x[0][:, 0].cpu().detach()
    target = y[0][:, 0].cpu().detach()
    prediction = preds[0][:, 0].cpu().detach()
    ground_truth = torch.cat((look_back_window, target), dim=0).numpy()
    prediction = torch.cat((look_back_window, prediction), dim=0).numpy()

    fig, ax = plt.subplots()
    ax.plot(ground_truth, label="Ground Truth")
    ax.plot(prediction, label="Prediction")
    ax.legend()

    wandb_logger.experiment.log({"test/predictions": wandb.Image(fig)})

    plt.close(fig)
