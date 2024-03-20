from typing import Dict, List, Tuple, Union

import torch
from dotmap import DotMap
from pytorch_models.models.base import BaseMILSurvModel
from pytorch_models.models.classification.dtfd import DTFD
from pytorch_models.utils.survival import coxloss
from pytorch_models.utils.tensor import aggregate_features


class DTFD_PL_Surv(BaseMILSurvModel):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        loss_type="cox",
        size: Union[List[int], Tuple[int, int]] = None,
        K: int = 1,
        n_bags=3,
        dropout=0.25,
        multires_aggregation: Union[None, str] = None,
        n_resolutions: int = 1,
    ):
        super(DTFD_PL_Surv, self).__init__(
            config,
            n_classes=n_classes,
            loss_type=loss_type,
            size=size,
            multires_aggregation=multires_aggregation,
            n_resolutions=n_resolutions,
        )

        assert len(size) == 3, "size must be a tuple of size 3"

        self.multires_aggregation = multires_aggregation

        self.model = DTFD(
            size=size,
            n_classes=self.n_classes,
            K=K,
            n_bags=n_bags,
            dropout=dropout,
        )

    def forward(self, batch, is_predict=False):
        # Batch
        features, event, survtime = (
            batch["features"],
            batch["event"],
            batch["survtime"],
        )

        if self.loss_type != "cox_loss" or (
            len(survtime.shape) > 1 and survtime.shape[1] > 1
        ):
            survtime = torch.argmax(survtime, dim=1).view(-1, 1)

        # Prediction
        logits, sub_logits = self._forward(
            features
        )  ### batch_size, batch_size x numGroup x fs

        logits = torch.sigmoid(logits)
        sub_logits = torch.sigmoid(sub_logits)

        S, risk = None, None
        sub_S, sub_risk = None, None
        if self.n_classes > 1 and logits.shape[1] > 1:
            raise NotImplementedError
            S = torch.cumprod(1 - logits, dim=2)
            risk = -torch.sum(S, dim=2).detach().cpu().numpy()
            sub_S = torch.cumprod(1 - sub_logits, dim=2)
            sub_risk = -torch.sum(sub_S, dim=2).detach().cpu().numpy()

        sub_event = event.repeat(1, sub_logits.shape[1]).reshape(
            -1
        )  ### batch_size x numGroup -> batch_size * numGroup x 1
        sub_survtime = survtime.repeat(1, sub_logits.shape[1]).reshape(
            -1
        )  ### batch_size x numGroup -> batch_size * numGroup x 1
        sub_logits = sub_logits.reshape(
            -1
        )  ### batch_size x numGroup -> batch_size * numGroup x 1

        loss = self.compute_loss(survtime, event, logits, S)
        loss += self.compute_loss(sub_survtime, sub_event, sub_logits, S)
        if self.l1_reg_weight:
            loss = loss + self.l1_regularisation(l_w=self.l1_reg_weight)

        return {
            "event": event.squeeze(),
            "survtime": survtime.squeeze(),
            "hazards": logits,
            "risk": risk,
            "S": S,
            "loss": loss,
            "slide_name": batch["slide_name"],
        }

    def _forward(self, features_batch: List[Dict[str, torch.Tensor]]):
        logits = []
        sub_logits = []
        for features in features_batch:
            h: List[torch.Tensor] = [features[key] for key in features]
            if self.multires_aggregation in ["linear", "linear_2"]:
                h = self.linear_agg(h)
            else:
                h: torch.Tensor = aggregate_features(
                    h, method=self.multires_aggregation
                )
            if len(h.shape) == 3:
                h = h.squeeze(dim=0)
            _logits, _sub_logits = self.model.forward(h)
            logits.append(_logits.squeeze())
            sub_logits.append(torch.stack(_sub_logits).squeeze())
        return torch.stack(logits), torch.stack(sub_logits)
