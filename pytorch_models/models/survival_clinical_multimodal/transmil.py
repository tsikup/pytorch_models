from pytorch_models.models.base import BaseClinicalMultimodalMILSurvModel
from pytorch_models.models.classification_clinical_multimodal.transmil import (
    TransMIL_Clinical_Multimodal,
)


class TransMIL_PL_Surv(BaseClinicalMultimodalMILSurvModel):
    def __init__(
        self,
        config,
        n_classes,
        loss_type="cox",
        size=(1024, 512),
        size_clinical=None,
        multires_aggregation=None,
        multimodal_aggregation="concat",
        n_resolutions: int = 1,
        dropout=0.5,
    ):
        self.multires_aggregation = multires_aggregation
        super(TransMIL_PL_Surv, self).__init__(
            config,
            n_classes=n_classes,
            loss_type=loss_type,
            size=size,
            multires_aggregation=multires_aggregation,
            n_resolutions=n_resolutions,
            multimodal_aggregation=multimodal_aggregation,
            size_clinical=size_clinical,
        )

        assert (
            self.n_classes == 1
        ), "Survival model should have 1 output class (i.e. hazard)"

        self.model = TransMIL_Clinical_Multimodal(
            n_classes=n_classes,
            size=size,
            size_clinical=size_clinical,
            multimodal_aggregation=multimodal_aggregation,
            dropout=dropout,
        )
