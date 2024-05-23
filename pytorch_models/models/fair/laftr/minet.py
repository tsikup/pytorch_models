from typing import List, Tuple, Union

from dotmap import DotMap
from pytorch_models.models.base_fair import BaseMILModel_LAFTR
from pytorch_models.models.classification.minet import get_minet_model


class MINET_LAFTR_PL(BaseMILModel_LAFTR):
    def __init__(
        self,
        config: DotMap,
        n_classes: int,
        n_groups: int,
        size: Union[List[int], Tuple[int, int]] = None,
        dropout: bool = True,
        pooling_mode="max",
        multires_aggregation: Union[None, str] = None,
        n_resolutions: int = 1,
        SensWeights=None,
        LabelSensWeights=None,
        adversary_size: int = 32,
        model_var: str = "eqodd",
        aud_steps: int = 1,
        class_coeff: float = 1.0,
        fair_coeff: float = 1.0,
        gradient_clip_value: float = 0.5,
        gradient_clip_algorithm: str = "norm",
    ):
        super(MINET_LAFTR_PL, self).__init__(
            config,
            n_classes=n_classes,
            n_groups=n_groups,
            hidden_size=None,
            size=size,
            multires_aggregation=multires_aggregation,
            n_resolutions=n_resolutions,
            adversary_size=adversary_size,
            model_var=model_var,
            aud_steps=aud_steps,
            class_coeff=class_coeff,
            fair_coeff=fair_coeff,
            SensWeights=SensWeights,
            LabelSensWeights=LabelSensWeights,
            gradient_clip_value=gradient_clip_value,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

        self.size = size
        self.dropout = dropout
        self.pooling_mode = pooling_mode

        self.model = get_minet_model(
            config,
            self.n_classes,
            size,
            pooling_mode,
            return_features=False,
            return_preds=False,
        )

        self.discriminator = get_minet_model(
            config,
            self.n_groups,
            size,
            pooling_mode,
            return_features=False,
            return_preds=False,
        )
