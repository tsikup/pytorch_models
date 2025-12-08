import re
from datetime import timedelta
from typing import Union, List

from torch.cuda import current_device
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

from zeus.monitor import ZeusMonitor


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
    try:
        if config.energy_monitoring:
            callbacks.append(EnergyMonitorCallback())
    except AttributeError:
        pass
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

class EnergyMonitorCallback(Callback):
    def __init__(self, gpu_indices: List[int] = None) -> None:
        super().__init__()
        self.gpu_indices = gpu_indices if gpu_indices is not None else [current_device()]
        self.zeus_monitor = ZeusMonitor(gpu_indices=self.gpu_indices)
        self.time = dict()
        self.energy = dict()
        self._metrics = dict()
        for mode in ["train", "val"]:
            self.time[mode] = dict(epoch=[], batch=[])
            self.energy[mode] = dict(epoch=[], batch=[])
        for mode in ["test", "predict", "entire_fit"]:
            self.time[mode] = []
            self.energy[mode] = []

    @property
    def metrics(self):
        self._compute_metrics()
        return self._metrics

    def _compute_metrics(self):
        def _mean(values):
            return sum(values) / len(values) if values else None
        for mode in ['train', 'val']:
            self._metrics[f'avg_{mode}_time_epoch'] = _mean(self.time[mode]['epoch'])
            self._metrics[f'avg_{mode}_energy_epoch'] = _mean(self.energy[mode]['epoch'])
            self._metrics[f'avg_{mode}_time_batch'] = _mean(self.time[mode]['batch'])
            self._metrics[f'avg_{mode}_energy_batch'] = _mean(self.energy[mode]['batch'])
        self._metrics['entire_fit_time'] = _mean(self.time['entire_fit'])
        self._metrics['entire_fit_energy'] = _mean(self.energy['entire_fit'])
        self._metrics['predict_time'] = _mean(self.time['predict'])
        self._metrics['predict_energy'] = _mean(self.energy['predict'])
        self._metrics['test_time'] = _mean(self.time['test'])
        self._metrics['test_energy'] = _mean(self.energy['test'])

    def on_fit_start(self, trainer, pl_module):
        self.zeus_monitor.begin_window("entire_fit")

    def on_fit_end(self, trainer, pl_module):
        mes = self.zeus_monitor.end_window("entire_fit")
        self.time['entire_fit'].append(mes.time)
        self.energy['entire_fit'].append(mes.total_energy)

    def on_train_epoch_start(self, trainer, pl_module):
        self.zeus_monitor.begin_window("train_epoch")

    def on_train_epoch_end(self, trainer, pl_module):
        mes = self.zeus_monitor.end_window("train_epoch")
        self.time['train']['epoch'].append(mes.time)
        self.energy['train']['epoch'].append(mes.total_energy)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.zeus_monitor.begin_window("train_batch")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        mes = self.zeus_monitor.end_window("train_batch")
        self.time['train']['batch'].append(mes.time)
        self.energy['train']['batch'].append(mes.total_energy)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.zeus_monitor.begin_window("val_epoch")

    def on_validation_epoch_end(self, trainer, pl_module):
        mes = self.zeus_monitor.end_window("val_epoch")
        self.time['val']['epoch'].append(mes.time)
        self.energy['val']['epoch'].append(mes.total_energy)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.zeus_monitor.begin_window("val_batch")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        mes = self.zeus_monitor.end_window("val_batch")
        self.time['val']['batch'].append(mes.time)
        self.energy['val']['batch'].append(mes.total_energy)

    def on_test_start(self, trainer, pl_module):
        self.zeus_monitor.begin_window("test")

    def on_test_end(self, trainer, pl_module):
        mes = self.zeus_monitor.end_window("test")
        self.time['test'].append(mes.time)
        self.energy['test'].append(mes.total_energy)

    def on_predict_start(self, trainer, pl_module):
        self.zeus_monitor.begin_window("predict")

    def on_predict_end(self, trainer, pl_module):
        mes = self.zeus_monitor.end_window("predict")
        self.time['predict'].append(mes.time)
        self.energy['predict'].append(mes.total_energy)




