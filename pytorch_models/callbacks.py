from lightning.pytorch.callbacks import (
    Callback,
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    StochasticWeightAveraging,
)
from lightning.pytorch.loggers import CometLogger


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
