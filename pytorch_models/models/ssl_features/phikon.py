import torch
from transformers import ViTModel


class PhiKonModel(torch.nn.Module):
    def __init__(self):
        super(PhiKonModel, self).__init__()
        self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

    def forward(self, x):
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0, :]  # (1, 768) shape
        return features


def phikon():
    return PhiKonModel()
