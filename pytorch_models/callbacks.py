import re
from datetime import timedelta
from typing import Union

from lightning.pytorch.callbacks import (
    Callback,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import CometLogger
from rich.progress import ProgressColumn
from rich.style import Style
from rich.text import Text


def get_callbacks(config):
    """
    Function to get training callbacks

    Parameters
    ----------
    config: DotMap instance with the current configuration.

    Returns
    -------
    List of callbacks
    """
    callbacks = []
    # callbacks.append(DeviceStatsMonitor())
    callbacks.append(LearningRateMonitor())
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=config.callbacks.checkpoint_dir,
        filename="model-epoch_{epoch:02d}-val_loss_{val_loss:.3f}",
        auto_insert_metric_name=False,
        save_top_k=config.callbacks.checkpoint_top_k,
    )
    callbacks.append(checkpoint_callback)
    if config.callbacks.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                min_delta=config.callbacks.es_min_delta,
                patience=config.callbacks.es_patience,
            )
        )
    if config.callbacks.stochastic_weight_averaging:
        callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2))
    if config.comet.enable:
        callbacks.append(CometMLCustomLogs())
    callbacks.append(BetterProgressBar())
    return callbacks


class CometMLCustomLogs(Callback):
    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                if isinstance(logger, CometLogger):
                    try:
                        print("Saving labels distribution to comet experiment.")
                        labels_dist_figures = (
                            trainer.datamodule.get_label_distributions(mode="train")
                        )
                        logger.experiment.log_figure(
                            "Training labels distribution",
                            labels_dist_figures["train"],
                            step=0,
                        )
                        logger.experiment.log_figure(
                            "Validation labels distribution",
                            labels_dist_figures["val"],
                            step=0,
                        )
                    except AttributeError as e:
                        pass


class RemainingTimeColumn(ProgressColumn):
    """Show total remaining time in training"""

    max_refresh = 1.0

    def __init__(self, style: Union[str, Style]) -> None:
        self.style = style
        self.estimated_time_per_epoch = None
        super().__init__()

    def render(self, task) -> Text:
        if "Epoch" in task.description:
            # fetch current epoch number from task description
            m = re.search(r"Epoch (\d+)/(\d+)", task.description)
            current_epoch, total_epoch = int(m.group(1)), int(m.group(2))

            elapsed = task.finished_time if task.finished else task.elapsed
            remaining = task.time_remaining

            if remaining:
                time_per_epoch = elapsed + remaining
                if self.estimated_time_per_epoch is None:
                    self.estimated_time_per_epoch = time_per_epoch
                else:
                    # smooth the time_per_epoch estimation
                    self.estimated_time_per_epoch = (
                        0.99 * self.estimated_time_per_epoch + 0.01 * time_per_epoch
                    )

                remaining_total = (
                    self.estimated_time_per_epoch * (total_epoch - current_epoch - 1)
                    + remaining
                )

                return Text(
                    f"{timedelta(seconds=int(remaining_total))}", style=self.style
                )

        else:
            return Text("")


class BetterProgressBar(RichProgressBar):
    def configure_columns(self, trainer) -> list:
        columns = super().configure_columns(trainer)
        columns.insert(4, RemainingTimeColumn(style=self.theme.time))
        return columns
