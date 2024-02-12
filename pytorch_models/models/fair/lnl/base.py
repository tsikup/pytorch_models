from pytorch_models.models.fair.utils import grad_reverse
from torch import nn


class _BaseLNL(nn.Module):
    def __init__(self):
        super(_BaseLNL, self).__init__()

    def forward(self, x, is_adv=True):
        logits, feats = self.main_model(x, return_features=True)
        if not is_adv:
            feats_aux = grad_reverse(feats)
        else:
            feats_aux = feats
        logits_aux = self.aux_model(feats_aux)
        return logits, logits_aux
