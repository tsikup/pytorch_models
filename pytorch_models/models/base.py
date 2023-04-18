from typing import Any, Callable, List, Optional, Union

import lightning as L
import numpy as np
import torch
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dotmap import DotMap
from lightning.pytorch.core.optimizer import LightningOptimizer
from torch_optimizer import AdaBound, SWATS, DiffGrad, AggMo, Apollo

from pytorch_models.losses.losses import get_loss
from pytorch_models.optim.lookahead import Lookahead
from pytorch_models.optim.utils import get_warmup_factor
from pytorch_models.utils.metrics.metrics import get_metrics
from ranger21 import Ranger21
from timm.optim import AdamP, AdaBelief
from torch.optim import (
    ASGD,
    SGD,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    NAdam,
    Optimizer,
    RAdam,
    SparseAdam,
)
from torch.optim.lr_scheduler import (
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
    MultiStepLR,
    ExponentialLR,
    CosineAnnealingLR,
    CyclicLR,
)


# TODO: inference sliding window
#  https://github.com/YtongXie/CoTr/blob/main/CoTr_package/CoTr/network_architecture/neural_network.py


class BaseModel(L.LightningModule):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        in_channels: Union[int, None],
        segmentation=False,
    ):
        super().__init__()
        self.config = config

        # log hyperparameters
        self.save_hyperparameters()

        self.n_classes = n_classes
        self.in_channels = in_channels
        self.segmentation = segmentation
        self.dim = self.config.model.input_shape

        # Hyperparameters
        self.learning_rate = self.config.trainer.optimizer_params.learning_rate
        self.batch_size = self.config.trainer.batch_size

        # Get Loss
        self.loss = get_loss(
            config_losses=config.trainer.loss,
            classes_loss_weights=config.trainer.classes_loss_weights,
            multi_loss_weights=config.trainer.multi_loss_weights,
        )

        # Get metrics
        self.sync_dist = (
            True
            if self.config.trainer.sync_dist
            and (torch.cuda.device_count() > 1 or self.config.devices.nodes > 1)
            else False
        )

        self.train_metrics = get_metrics(
            config,
            n_classes=self.n_classes,
            dist_sync_on_step=False,
            mode="train",
            segmentation=self.segmentation,
        ).clone(prefix="train_")

        self.val_metrics = get_metrics(
            config,
            n_classes=self.n_classes,
            dist_sync_on_step=self.sync_dist,
            mode="val",
            segmentation=self.segmentation,
        ).clone(prefix="val_")

        self.test_metrics = get_metrics(
            config,
            n_classes=self.n_classes,
            dist_sync_on_step=self.sync_dist,
            mode="test",
            segmentation=self.segmentation,
        ).clone(prefix="test_")

    def l_regularisation(self, l=2):
        l_reg = None

        for W in self.parameters():
            if l_reg is None:
                l_reg = W.norm(l)
            else:
                l_reg = l_reg + W.norm(l)
        return l_reg

    def l1_regularisation(self, l_w=1.0):
        return l_w * self.l_regularisation(1)

    def l2_regularisation(self, l_w=1.0):
        return l_w * self.l_regularisation(2)

    def l12_regularisation(self, l1_w=0.3, l2_w=0.7):
        return l1_w * self.l_regularisation(1) + l2_w * self.l_regularisation(2)

    def forward(self, batch):
        # Batch
        images, target = batch
        # Prediction
        logits = self._forward(images)
        # Loss (on logits)
        loss = self.loss.forward(logits, target.float())

        # Sigmoid or Softmax activation
        if self.n_classes == 1:
            preds = logits.sigmoid()
        else:
            preds = torch.nn.functional.softmax(logits, dim=1)
        return {"images": images, "target": target, "preds": preds, "loss": loss}

    def configure_optimizers(self):
        optimizer = self._get_optimizer()
        if self.config.trainer.lr_scheduler is not None:
            scheduler = self._get_scheduler(optimizer)
            return [optimizer], [scheduler]
        else:
            return optimizer

    def _get_optimizer(self):
        SUPPORTED_OPTIMIZERS = {
            "adam": Adam,
            "ranger": Ranger21,
            "adamp": AdamP,
            "radam": RAdam,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "adamw": AdamW,
            "sgd": SGD,
            "asgd": ASGD,
            "nadam": NAdam,
            "sparseadam": SparseAdam,
            "adabound": AdaBound,
            "swats": SWATS,
            "diffgrad": DiffGrad,
            "aggmo": AggMo,
            "adabelief": AdaBelief,
            "apollo": Apollo,
        }

        OPTIM_ARGS = {
            "ranger": {
                "weight_decay": 1e-4
                if self.config.trainer.weight_decay == "default"
                else self.config.trainer.weight_decay,
                "lookahead_active": True,
                "use_madgrad": True,
                "use_adabelief": True,
                "softplus": True,
                "using_gc": True,
                "using_normgc": True,
                "normloss_active": True,
                "use_adaptive_gradient_clipping": True,
                "use_cheb": True,
                "use_warmup": False,
                "num_warmup_iterations": None,
                "warmdown_active": True,
            }
        }

        assert self.config.trainer.optimizer.lower() in SUPPORTED_OPTIMIZERS.keys(), (
            f"Unsupported Optimizer: {self.configs.model.optimizer}\n"
            f"Supported Optimizers: {SUPPORTED_OPTIMIZERS.keys()}"
        )

        if self.config.trainer.optimizer.lower() == "sgd":
            optimizer = SGD(
                self.parameters(),
                **self.config.trainer.optimizer_params,
            )
        elif self.config.trainer.optimizer.lower() == "ranger":
            optimizer = Ranger21(
                self.parameters(),
                num_epochs=self.config.trainer.epochs,
                num_batches_per_epoch=self.config.trainer.num_batches_per_epoch,
                **OPTIM_ARGS["ranger"],
                **self.config.trainer.optimizer_params,
            )
        else:
            optimizer = SUPPORTED_OPTIMIZERS[self.config.trainer.optimizer.lower()](
                self.parameters(),
                **self.config.trainer.optimizer_params,
            )

        if self.config.trainer.lookahead:
            optimizer = Lookahead(optimizer=optimizer, k=5, alpha=0.5)

        # if self.config.trainer.optimizer_lars:
        #     assert (
        #         not self.config.trainer.lookahead
        #     ), "Lookahead and LARS cannot be used together."
        #     # Layer-wise Adaptive Rate Scaling for large batch training.
        #     # Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
        #     # I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
        #     # Implements the LARS learning rate scheme presented in the paper above. This
        #     # optimizer is useful when scaling the batch size to up to 32K without
        #     # significant performance degradation. It is recommended to use the optimizer
        #     # in conjunction with:
        #     #     - Gradual learning rate warm-up
        #     #     - Linear learning rate scaling
        #     #     - Poly rule learning rate decay
        #     optimizer = LARS(optimizer=optimizer, eps=1e-8, trust_coef=0.001)

        return optimizer

    def _get_scheduler(self, optimizer, scheduler_name=None):
        SUPPORTED_SCHEDULERS = {
            "plateau": (ReduceLROnPlateau, "epoch"),
            "step": (StepLR, "epoch"),
            "multistep": (MultiStepLR, "epoch"),
            "exp": (ExponentialLR, "epoch"),
            "cosineannealing": (CosineAnnealingLR, "step"),
            "cosineannealingwarmuprestarts": (CosineAnnealingWarmupRestarts, "step"),
            "cyclic": (CyclicLR, "step"),
            "onecycle": (OneCycleLR, "step"),
        }

        if scheduler_name is None:
            scheduler_name = self.config.trainer.lr_scheduler.lower()

        assert scheduler_name in SUPPORTED_SCHEDULERS.keys(), (
            f"Unsupported Scheduler: {scheduler_name}\n"
            f"Supported Schedulers: {list(SUPPORTED_SCHEDULERS.keys())}"
        )

        lr_scheduler, interval = SUPPORTED_SCHEDULERS[scheduler_name]
        lr_scheduler = lr_scheduler(
            optimizer, **self.config.trainer.lr_scheduler_params
        )

        lr_scheduler = {"scheduler": lr_scheduler, "interval": interval, "frequency": 1}

        if scheduler_name == "plateau":
            lr_scheduler["monitor"] = self.config.trainer.lr_scheduler_metric_monitor

        return lr_scheduler

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Union[Optimizer, LightningOptimizer],
        optimizer_closure: Optional[Callable[[], Any]] = None,
    ) -> None:
        # update params
        optimizer.step(closure=optimizer_closure)
        # manually warm up lr without a scheduler
        if (
            self.config.trainer.lr_warmup
            and self.trainer.global_step < self.config.trainer.lr_warmup_period
        ):
            lr_scale = get_warmup_factor(
                self.trainer.global_step,
                self.config.trainer.lr_warmup_period,
                mode=self.config.trainer.lr_warmup_mode,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.learning_rate

    def _forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        images, target, preds, loss = (
            output["images"],
            output["target"],
            output["preds"],
            output["loss"],
        )
        self._log_metrics(preds, target, loss, "train")
        return {"loss": loss, "preds": preds, "target": target}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        images, target, preds, loss = (
            output["images"],
            output["target"],
            output["preds"],
            output["loss"],
        )
        self._log_metrics(preds, target, loss, "val")
        return {"val_loss": loss, "val_preds": preds, "val_target": target}

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)

        print(output)

        images, target, preds, loss = (
            output["images"],
            output["target"],
            output["preds"],
            output["loss"],
        )

        self._log_metrics(preds, target, loss, "test")

        return {"test_loss": loss, "test_preds": preds, "test_target": target}

    def _compute_metrics(self, preds, target, mode):
        if mode == "val":
            metrics = self.val_metrics
        elif mode == "train":
            metrics = self.train_metrics
        elif mode in ["eval", "test"]:
            metrics = self.test_metrics
        if self.n_classes in [1, 2]:
            metrics(preds.view(-1), target.view(-1))
        else:
            metrics(preds, target.view(-1))
            # metrics(preds, self._one_hot_target(preds, target))

    def _log_metrics(self, preds, target, loss, mode):
        on_step = False if mode != "train" else True
        # https://github.com/Lightning-AI/lightning/issues/13210
        sync_dist = self.sync_dist and (
            mode == "val" or mode == "test" or mode == "eval"
        )
        if mode == "val":
            metrics = self.val_metrics
        elif mode == "train":
            metrics = self.train_metrics
        elif mode == "test":
            metrics = self.test_metrics

        self._compute_metrics(preds, target, mode)
        self.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=False, logger=True
        )
        self.log(
            f"{mode}_loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=sync_dist,
        )


class BaseMILModel(BaseModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
    ):
        super(BaseMILModel, self).__init__(
            config, n_classes=n_classes, in_channels=None, segmentation=False
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, target = batch
        # Prediction
        logits = self._forward(features)
        # Loss (on logits)
        loss = self.loss.forward(logits, target.float())
        # Sigmoid or Softmax activation
        if self.n_classes == 1:
            preds = logits.sigmoid()
        else:
            preds = torch.nn.functional.softmax(logits, dim=1)
        return {"target": target, "preds": preds, "loss": loss}

    def _forward(self, x):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        target, preds, loss = (
            output["target"],
            output["preds"],
            output["loss"],
        )
        self._log_metrics(preds, target, loss, "train")
        return {"loss": loss, "preds": preds, "target": target}

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        target, preds, loss = (
            output["target"],
            output["preds"],
            output["loss"],
        )
        self._log_metrics(preds, target, loss, "val")
        return {"val_loss": loss, "val_preds": preds, "val_target": target}

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)

        target, preds, loss = (
            output["target"],
            output["preds"],
            output["loss"],
        )

        self._log_metrics(preds, target, loss, "test")

        return {"test_loss": loss, "test_preds": preds, "test_target": target}

    def predict_step(self, batch, batch_idx):
        _, labels = batch
        output = self.forward(batch, is_predict=True)

        return output["preds"], labels


class EnsembleInferenceModel(BaseModel):
    def __init__(
        self,
        base_class: L.LightningModule,
        model_ckpts: List[str],
        config: DotMap,
        n_classes: int,
        in_channels: int,
        segmentation=False,
        ensemble_mode="mean",
        model_weights: List[float] = None,
    ):
        super(EnsembleInferenceModel, self).__init__(
            config, n_classes, in_channels, segmentation
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.ensemble_mode = ensemble_mode

        self.models = [base_class.load_from_checkpoint(ckpt) for ckpt in model_ckpts]

        self.model_weights = torch.from_numpy(
            np.array(
                model_weights
                if model_weights is not None
                else [1 for _ in range(len(model_ckpts))]
            )
        ).to(device)

        self.in_channels = self.models[0].in_channels

        for model in self.models:
            model.to(device)
            model.eval()

    def aggregate_predictions(self, preds):
        if self.ensemble_mode == "mean":
            preds = torch.cat(preds, dim=-1)
            weighted_preds = preds * self.model_weights
            weighted_preds = torch.sum(weighted_preds) / sum(self.model_weights)
            return weighted_preds

    def forward(self, batch):
        return self._forward(batch)

    def _forward(self, batch):
        _, target = batch

        outcomes = [model.forward(batch, is_predict=True) for model in self.models]

        preds = self.aggregate_predictions([o["preds"] for o in outcomes])

        return dict(preds=preds, target=target)

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)

        target, preds = output["target"], output["preds"]

        self._log_metrics(preds, target, 0, "test")

        return {"test_preds": preds, "test_target": target}

    def predict_step(self, batch, batch_idx):
        output = self.forward(batch, is_predict=True)

        return output["preds"]
