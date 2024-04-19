from typing import Any, Callable, Dict, List, Optional, Union

import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from dotmap import DotMap
from lightning.pytorch.core.optimizer import LightningOptimizer
from pytorch_models.losses.losses import get_loss
from pytorch_models.models.multimodal.clinical import ClinicalModel
from pytorch_models.models.multimodal.two_modalities import IntegrateTwoModalities
from pytorch_models.models.utils import (
    LinearWeightedSum,
    LinearWeightedTransformationSum,
)
from pytorch_models.optim.lookahead import Lookahead
from pytorch_models.optim.utils import get_warmup_factor
from pytorch_models.utils.metrics.metrics import get_metrics
from pytorch_models.utils.survival import CoxSurvLoss, MyDeepHitLoss, NLLSurvLoss
from pytorch_models.utils.tensor import aggregate_features, pad_col
from ranger21 import Ranger21
from timm.optim import AdaBelief, AdamP
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
    CosineAnnealingLR,
    CyclicLR,
    ExponentialLR,
    MultiStepLR,
    OneCycleLR,
    ReduceLROnPlateau,
    StepLR,
)
from torch_optimizer import SWATS, AdaBound, AggMo, Apollo, DiffGrad
from torchvision.transforms.functional import center_crop

# TODO: inference sliding window
#  https://github.com/YtongXie/CoTr/blob/main/CoTr_package/CoTr/network_architecture/neural_network.py


class BaseModel(L.LightningModule):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        in_channels: Union[int, None],
        segmentation=False,
        automatic_optimization=True,
    ):
        super().__init__()
        self.config = config

        self.automatic_optimization = automatic_optimization

        # log hyperparameters
        self.save_hyperparameters()

        self.n_classes = n_classes
        assert self.n_classes > 0, "n_classes must be greater than 0"
        self.in_channels = in_channels
        self.segmentation = segmentation

        # Hyperparameters
        self.learning_rate = self.config.trainer.optimizer_params.lr
        self.batch_size = self.config.trainer.batch_size
        self.l1_reg_weight = self.config.trainer.l1_reg_weight
        self.l2_reg_weight = self.config.trainer.l2_reg_weight
        if self.l1_reg_weight:
            self.l1_reg_weight = float(self.l1_reg_weight)
        if self.l2_reg_weight:
            self.l2_reg_weight = float(self.l2_reg_weight)

        # Get Loss
        self.loss = get_loss(
            config_losses=config.trainer.loss,
            n_classes=self.n_classes,
            classes_loss_weights=config.trainer.classes_loss_weights,
            multi_loss_weights=config.trainer.multi_loss_weights,
            samples_per_cls=config.trainer.samples_per_class,
            reduction="mean",
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

    def forward(self, batch, is_predict=False):
        # Batch
        images, target = batch
        # Prediction
        logits = self._forward(images)
        # Loss (on logits)
        loss = None
        if not is_predict:
            if len(target.shape) > 1 and self.n_classes > 1:
                if target.shape[1] == 1:
                    target = target.squeeze(1)
                else:
                    target = torch.argmax(target, dim=1)
            if self.n_classes > 1:
                loss = self.loss.forward(logits, target)
            else:
                loss = self.loss.forward(logits, target.float())

            if self.l1_reg_weight:
                loss = loss + self.l1_regularisation(l_w=self.l1_reg_weight)

            if self.l2_reg_weight:
                loss = loss + self.l2_regularisation(l_w=self.l2_reg_weight)

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

    def _get_optimizer(self, params=None):
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
                self.parameters() if params is None else params,
                **self.config.trainer.optimizer_params,
            )
        elif self.config.trainer.optimizer.lower() == "ranger":
            optimizer = Ranger21(
                self.parameters() if params is None else params,
                num_epochs=self.config.trainer.epochs,
                num_batches_per_epoch=self.config.trainer.num_batches_per_epoch,
                **OPTIM_ARGS["ranger"],
                **self.config.trainer.optimizer_params,
            )
        else:
            optimizer = SUPPORTED_OPTIMIZERS[self.config.trainer.optimizer.lower()](
                self.parameters() if params is None else params,
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
            if len(preds.shape) == 2 and preds.shape[1] > 1:
                preds = preds[:, 1]
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
        if isinstance(loss, dict):
            for key, value in loss.items():
                self.log(
                    f"{mode}_{key}",
                    value,
                    on_step=on_step,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                    sync_dist=sync_dist,
                    batch_size=self.batch_size,
                )
        else:
            self.log(
                f"{mode}_loss",
                loss,
                on_step=on_step,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=sync_dist,
                batch_size=self.batch_size,
            )


class BaseSegmentationModel(BaseModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        in_channels: int,
        interpolate_output: bool = False,
    ):
        super(BaseSegmentationModel, self).__init__(
            config, n_classes=n_classes, in_channels=in_channels, segmentation=True
        )
        self.interpolate_output = interpolate_output

    def forward(self, batch, is_predict=False):
        # Batch
        images, target = batch
        # Prediction
        logits = self._forward(images)
        # Loss (on logits)
        loss = None
        if not is_predict:
            if logits.shape[-2] < target.shape[-2]:
                target = center_crop(target, logits.shape[-2:])
            if len(target.shape) == 4 and self.n_classes > 1:
                if target.shape[1] == 1:
                    target = target.squeeze(1)
                else:
                    target = torch.argmax(target, dim=1)
            if self.n_classes > 1:
                loss = self.loss.forward(logits, target)
            else:
                loss = self.loss.forward(logits, target.float())

            if self.l1_reg_weight:
                loss = loss + self.l1_regularisation(l_w=self.l1_reg_weight)

            if self.l2_reg_weight:
                loss = loss + self.l2_regularisation(l_w=self.l2_reg_weight)

        if self.interpolate_output:
            preds = F.interpolate(
                logits, (images.shape[-2], images.shape[-1]), mode="bilinear"
            )
        else:
            preds = logits
        if self.n_classes == 1:
            preds = preds.sigmoid()
            out = torch.where(preds > 0.5, 1, 0)
        else:
            preds = torch.nn.functional.softmax(preds, dim=1)
            out = torch.argmax(preds, dim=1)
        return {
            "images": images,
            "target": target,
            "preds": preds,
            "loss": loss,
            "out": out,
            "logits": logits,
        }


class BaseMILModel(BaseModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        multires_aggregation: Union[Dict[str, str], str, None] = None,
        size: List[int] = None,
        n_resolutions: int = 1,
    ):
        super(BaseMILModel, self).__init__(
            config, n_classes=n_classes, in_channels=None, segmentation=False
        )

        self.multires_aggregation = multires_aggregation
        self.n_resolutions = n_resolutions

        if self.config.model.classifier != "clam":
            if self.multires_aggregation == "linear":
                assert size is not None
                self.linear_agg = LinearWeightedTransformationSum(
                    size[0], self.n_resolutions
                )
            elif self.multires_aggregation == "linear_2":
                assert size is not None
                self.linear_agg = LinearWeightedSum(size[0], self.n_resolutions)

    def forward(self, batch, is_predict=False):
        # Batch
        features, target = batch["features"], batch["labels"]
        # Prediction
        logits = self._forward(features)
        # Loss (on logits)
        loss = None
        if not is_predict:
            loss = self.loss.forward(
                logits, target.float() if logits.shape[1] == 1 else target.view(-1)
            )
        # Sigmoid or Softmax activation
        if self.n_classes == 1:
            preds = logits.sigmoid()
        else:
            preds = torch.nn.functional.softmax(logits, dim=1)
        return {
            "target": target,
            "preds": preds,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch):
        logits = []
        for singlePatientFeatures in features_batch:
            h: List[torch.Tensor] = [
                singlePatientFeatures[key] for key in singlePatientFeatures
            ]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _logits = self.model.forward(h)
            logits.append(_logits)
        return torch.vstack(logits)

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
        output = self.forward(batch, is_predict=True)
        return output["preds"], batch["labels"], batch["slide_name"]


class BaseClinicalMultimodalMILModel(BaseMILModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: List[int],
        size_cat: int,
        size_cont: int,
        clinical_layers: List[int],
        multimodal_odim: int,
        embed_size: list = None,
        batch_norm: bool = True,
        bilinear_scale_dim1: int = 1,
        bilinear_scale_dim2: int = 1,
        dropout: float = 0.5,
        multires_aggregation: Union[Dict[str, str], str, None] = None,
        multimodal_aggregation: str = "concat",
        n_resolutions: int = 1,
    ):
        super(BaseClinicalMultimodalMILModel, self).__init__(
            config=config,
            n_classes=n_classes,
            multires_aggregation=multires_aggregation,
            size=size,
            n_resolutions=n_resolutions,
        )
        self.multimodal_aggregation = multimodal_aggregation
        self.multimodal_odim = multimodal_odim
        self.size_cat = size_cat
        self.size_cont = size_cont
        self.clinical_layers = clinical_layers
        self.embed_size = embed_size
        self.batch_norm = batch_norm

        self.clinical_model = ClinicalModel(
            size_cat=size_cat,
            size_cont=size_cont,
            layers=clinical_layers,
            embed_size=embed_size,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        if config.model.classifier != "clam":
            if multimodal_aggregation == "concat":
                self.multimodal_odim = size[-1] + clinical_layers[-1]
            elif multimodal_aggregation == "kron":
                self.multimodal_odim = size[-1] * clinical_layers[-1]

            self.integration_model = IntegrateTwoModalities(
                dim1=size[-1],
                dim2=clinical_layers[-1],
                odim=self.multimodal_odim,
                method=multimodal_aggregation,
                bilinear_scale_dim1=bilinear_scale_dim1,
                bilinear_scale_dim2=bilinear_scale_dim2,
                dropout=dropout,
            )

    def _forward(self, features_batch):
        clinical_cat = []
        clinical_cont = []
        imaging = []

        for singlePatientFeatures in features_batch:
            clinical_cat.append(singlePatientFeatures.pop("clinical_cat", None))
            clinical_cont.append(singlePatientFeatures.pop("clinical_cont", None))
            h: List[torch.Tensor] = [
                singlePatientFeatures[key] for key in singlePatientFeatures
            ]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            imaging.append(self.model.forward_imaging(h))
        clinical = self.clinical_model(
            torch.stack(clinical_cat, dim=0), torch.stack(clinical_cont, dim=0)
        )
        mmfeats = self.integration_model(torch.stack(imaging, dim=0), clinical)
        logits = self.model.forward(mmfeats)
        return logits


class BaseSurvModel(BaseModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        in_channels: Union[int, None],
        loss_type="cox",
    ):
        super(BaseSurvModel, self).__init__(
            config, n_classes=n_classes, in_channels=in_channels, segmentation=False
        )
        self.loss = None
        del self.loss
        self.loss_type = loss_type
        self._define_loss(loss_type)

        self.train_metrics = get_metrics(
            self.config,
            n_classes=self.n_classes,
            dist_sync_on_step=False,
            mode="train",
            segmentation=False,
            survival=True,
            cindex_method="pycox"
            if loss_type
            in [
                "nll_pmf_loss",
                "deephit_loss",
                "deephit_single_loss",
            ]
            else "counts",
            cuts=self.config.dataset.cuts,
        ).clone(prefix="train_")

        self.val_metrics = get_metrics(
            self.config,
            n_classes=self.n_classes,
            dist_sync_on_step=False,
            mode="val",
            segmentation=False,
            survival=True,
            cindex_method="pycox"
            if loss_type
            in [
                "nll_pmf_loss",
                "deephit_loss",
                "deephit_single_loss",
            ]
            else "counts",
            cuts=self.config.dataset.cuts,
        ).clone(prefix="val_")

        self.test_metrics = get_metrics(
            self.config,
            n_classes=self.n_classes,
            dist_sync_on_step=False,
            mode="test",
            segmentation=False,
            survival=True,
            cindex_method="pycox"
            if loss_type
            in [
                "nll_pmf_loss",
                "deephit_loss",
                "deephit_single_loss",
            ]
            else "counts",
            cuts=self.config.dataset.cuts,
        ).clone(prefix="test_")

    def _define_loss(self, loss_type, alpha=0.5, sigma=1.0):
        if loss_type == "cox_loss":
            self._coxloss = CoxSurvLoss()
        elif loss_type == "cox_loss__ce":
            assert (
                self.config.model.classifier == "clam"
            ), "Implemented only for CLAM models for now.."
            self._coxloss = CoxSurvLoss()
            self._ce = torch.nn.BCEWithLogitsLoss()
        elif loss_type.startswith("nll"):
            self._nll_loss = NLLSurvLoss(type=loss_type)
        elif loss_type == "deephit_single_loss":
            self._deephistloss = MyDeepHitLoss(single=True, alpha=alpha, sigma=sigma)
        elif loss_type == "deephit_loss":
            self._deephistloss = MyDeepHitLoss(single=False, alpha=alpha, sigma=sigma)
        else:
            raise NotImplementedError

    def compute_loss(self, survtime, event, logits, S):
        if self.loss_type == "cox_loss":
            return self._coxloss(logits=logits, survtimes=survtime, events=event)
        elif self.loss_type == "cox_loss__ce":
            assert (isinstance(logits, list) or isinstance(logits, tuple)) and len(
                logits
            ) == 2
            return self._coxloss(
                logits=logits[0], survtimes=survtime, events=event
            ) + self._ce(logits[1], event.float())
        elif self.loss_type in [
            "nll_loss",
            "nll_logistic_hazard_loss",
            "nll_pmf_loss",
            "nll_pc_hazard_loss",
        ]:
            return self._nll_loss(logits=logits, S=S, Y=survtime, c=1 - event)
        elif self.loss_type in ["deephit_loss", "deephit_single_loss"]:
            return self._deephistloss(survtimes=survtime, events=event, logits=logits)
        else:
            raise NotImplementedError

    def forward(self, batch, is_predict=False):
        raise NotImplementedError

    def predict_pmf(self, logits):
        pmf = pad_col(logits).softmax(1)[:, :-1]
        surv = 1 - pmf.cumsum(1)
        risk = -torch.sum(surv, dim=1)
        return dict(pmf=pmf, surv=surv, risk=risk)

    def predict_deephit(self, logits):
        if self.loss_type == "deephit_loss":
            pmf = pad_col(logits.view(logits.size(0), -1)).softmax(1)[:, :-1]
            pmf = pmf.view(logits.shape).transpose(0, 1).transpose(1, 2)
            cif = pmf.cumsum(1)
            surv = 1.0 - cif.sum(0)
            risk = -torch.sum(surv, dim=1)
            # # Calculate risk at time T
            # # https://github.com/chl8856/DeepHit/blob/master/summarize_results.py
            # # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            # risk = np.sum(pred[:,:,:(eval_horizon+1)], axis=2) #risk score until EVAL_TIMES
            return dict(pmf=pmf, cif=cif, surv=surv, risk=risk)
        return self.predict_pmf(logits)

    def predict_pchazard(self, logits, sub=1):
        raise NotImplementedError

    def predict_logistic_hazard(self, logits, epsilon=1e-7):
        hazards = logits.sigmoid()
        # surv = (1 - logits).add(epsilon).log().cumsum(1).exp()
        surv = (1 - logits).cumprod(1)
        risk = -torch.sum(surv, dim=1)
        return dict(hazards=hazards, surv=surv, risk=risk)

    def _calculate_surv_risk(self, logits):
        if self.loss_type.startswith("cox_loss"):
            # https://github.com/mahmoodlab/MCAT/blob/master/models/model_coattn.py
            return dict(hazards=logits.sigmoid(), surv=None, risk=None)
        elif self.loss_type in ["nll_loss", "nll_logistic_hazard_loss"]:
            return self.predict_logistic_hazard(logits)
        elif self.loss_type in ["deephit_single_loss", "deephit_loss"]:
            # https://github.com/havakv/pycox
            return self.predict_deephit(logits)
        elif self.loss_type == "nll_pmf_loss":
            return self.predict_pmf(logits)
        raise NotImplementedError

    def _compute_metrics(self, hazards, S, events, survtimes, mode):
        if mode == "val":
            metrics = self.val_metrics
        elif mode == "train":
            metrics = self.train_metrics
        elif mode in ["eval", "test"]:
            metrics = self.test_metrics
        metrics(hazards, S, events, survtimes)

    def _log_metrics(self, hazards, S, events, survtimes, loss, mode):
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

        self._compute_metrics(hazards, S, events, survtimes, mode)
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
            batch_size=self.batch_size,
        )

    def training_step(self, batch, batch_idx):
        output = self.forward(batch)
        event, survtime, hazards, risk, S, loss = (
            output["event"],
            output["survtime"],
            output["hazards"],
            output["risk"],
            output["S"],
            output["loss"],
        )
        self._log_metrics(
            risk if risk is not None else hazards, S, event, survtime, loss, "train"
        )
        return {
            "loss": loss,
            "hazards": hazards,
            "risk": risk,
            "S": S,
            "event": event,
            "survtime": survtime,
        }

    def validation_step(self, batch, batch_idx):
        output = self.forward(batch)
        event, survtime, hazards, risk, S, loss = (
            output["event"],
            output["survtime"],
            output["hazards"],
            output["risk"],
            output["S"],
            output["loss"],
        )
        self._log_metrics(
            risk if risk is not None else hazards, S, event, survtime, loss, "val"
        )
        return {
            "val_loss": loss,
            "val_hazards": hazards,
            "val_risk": risk,
            "val_S": S,
            "val_event": event,
            "val_survtime": survtime,
        }

    def test_step(self, batch, batch_idx):
        output = self.forward(batch)

        event, survtime, hazards, risk, S, loss = (
            output["event"],
            output["survtime"],
            output["hazards"],
            output["risk"],
            output["S"],
            output["loss"],
        )

        self._log_metrics(
            risk if risk is not None else hazards, S, event, survtime, loss, "test"
        )

        return {
            "test_loss": loss,
            "test_hazards": hazards,
            "test_risk": risk,
            "test_S": S,
            "event": event,
            "survtime": survtime,
        }

    def predict_step(self, batch, batch_idx):
        output = self.forward(batch, is_predict=True)
        return output


class BaseMILSurvModel(BaseSurvModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        loss_type="cox",
        size: List[int] = None,
        multires_aggregation: Union[Dict[str, str], str, None] = None,
        n_resolutions: int = 1,
    ):
        super(BaseMILSurvModel, self).__init__(
            config, n_classes=n_classes, in_channels=None, loss_type=loss_type
        )

        self.multires_aggregation = multires_aggregation
        self.n_resolutions = n_resolutions

        if self.config.model.classifier != "clam":
            if self.multires_aggregation == "linear":
                assert size is not None
                self.linear_agg = LinearWeightedTransformationSum(
                    size[0], self.n_resolutions
                )
            elif self.multires_aggregation == "linear_2":
                assert size is not None
                self.linear_agg = LinearWeightedSum(size[0], self.n_resolutions)

    def forward(self, batch, is_predict=False):
        # Batch
        features, event, survtime = (
            batch["features"],
            batch["event"],
            batch["survtime"],
        )

        if not self.loss_type.startswith("cox_loss") or (
            len(survtime.shape) > 1 and survtime.shape[1] > 1
        ):
            survtime = torch.argmax(survtime, dim=1).view(-1, 1)

        # Prediction
        logits = self._forward(features)
        res = self._calculate_surv_risk(logits)
        hazards, S, risk = (
            res.pop("hazards", None),
            res.pop("surv", None),
            res.pop("risk", None),
        )
        pmf, cif = res.pop("pmf", None), res.pop("cif", None)

        # Loss (on logits)
        loss = None
        if not is_predict:
            loss = self.compute_loss(survtime, event, logits, S)
            if self.l1_reg_weight:
                loss = loss + self.l1_regularisation(l_w=self.l1_reg_weight)

        return {
            "event": event,
            "survtime": survtime,
            "hazards": hazards,
            "risk": risk,
            "S": S,
            "pmf": pmf,
            "cif": cif,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch):
        logits = []
        for singlePatientFeatures in features_batch:
            h: List[torch.Tensor] = [
                singlePatientFeatures[key] for key in singlePatientFeatures
            ]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _logits = self.model.forward(h)
            logits.append(_logits)
        return torch.vstack(logits)


class BaseClinicalMultimodalMILSurvModel(BaseMILSurvModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        size: List[int],
        size_cat: int,
        size_cont: int,
        clinical_layers: List[int],
        multimodal_odim: int,
        embed_size: list = None,
        batch_norm: bool = True,
        dropout: float = 0.5,
        bilinear_scale_dim1: int = 1,
        bilinear_scale_dim2: int = 1,
        loss_type="cox",
        multires_aggregation: Union[Dict[str, str], str, None] = None,
        multimodal_aggregation: str = "concat",
        n_resolutions: int = 1,
    ):
        super(BaseClinicalMultimodalMILSurvModel, self).__init__(
            config=config,
            n_classes=n_classes,
            loss_type=loss_type,
            size=size,
            multires_aggregation=multires_aggregation,
            n_resolutions=n_resolutions,
        )

        self.multimodal_aggregation = multimodal_aggregation
        self.multimodal_odim = multimodal_odim
        self.size_cat = size_cat
        self.size_cont = size_cont
        self.clinical_layers = clinical_layers
        self.embed_size = embed_size
        self.batch_norm = batch_norm

        self.clinical_model = ClinicalModel(
            size_cat=size_cat,
            size_cont=size_cont,
            layers=clinical_layers,
            embed_size=embed_size,
            dropout=dropout,
            batch_norm=batch_norm,
        )

        if config.model.classifier != "clam":
            if multimodal_aggregation == "concat":
                self.multimodal_odim = size[-1] + clinical_layers[-1]
            elif multimodal_aggregation == "kron":
                self.multimodal_odim = size[-1] * clinical_layers[-1]

            self.integration_model = IntegrateTwoModalities(
                dim1=size[-1],
                dim2=clinical_layers[-1],
                odim=self.multimodal_odim,
                method=multimodal_aggregation,
                bilinear_scale_dim1=bilinear_scale_dim1,
                bilinear_scale_dim2=bilinear_scale_dim2,
                dropout=dropout,
            )

    def _forward(self, features_batch):
        clinical_cat = []
        clinical_cont = []
        imaging = []
        for singlePatientFeatures in features_batch:
            clinical_cat.append(singlePatientFeatures.pop("clinical_cat", None))
            clinical_cont.append(singlePatientFeatures.pop("clinical_cont", None))
            h: List[torch.Tensor] = [
                singlePatientFeatures[key] for key in singlePatientFeatures
            ]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            imaging.append(self.model.forward_imaging(h))
        clinical = self.clinical_model(
            torch.stack(clinical_cat, dim=0), torch.stack(clinical_cont, dim=0)
        )
        mmfeats = self.integration_model(torch.stack(imaging, dim=0), clinical)
        logits = self.model.forward(mmfeats)
        return logits


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
