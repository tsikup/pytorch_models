from dotmap import DotMap
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from ..base import BaseModel


# TODO: HuggingFace with PL
#  https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/text-transformers.html
class MySegformer(BaseModel):
    def __init__(self, config: DotMap, backbone="nvidia/mit-b3"):
        super(MySegformer, self).__init__(config)
        self.model = SegformerForSemanticSegmentation.from_pretrained(backbone)

    # def shared_step(self, batch):
    #     # Batch
    #     images, target = batch
    #     # Prediction
    #     logits = self.forward(images)
    #     logits = nn.functional.interpolate(logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
    #     # Loss (on logits)
    #     loss = self.loss.forward(logits, target)
    #     # Sigmoid or Softmax activation
    #     if self.config.dataset.num_classes == 1:
    #         preds = logits.sigmoid()
    #     else:
    #         preds = torch.nn.functional.softmax(logits, dim=1)
    #     return {'images': images, 'target': target, 'preds': preds, 'loss': loss}

    # def forward(self, x):
    #     y = self.model(x)
    #     if self.mode in ['test', 'inference']:
    #         y = nn.functional.interpolate(y, size=self.config.model.output_shape[-2:], mode="bilinear", align_corners=False)
    #     return y


# if __name__ == '__main__':
#     import torch
#     import json
#     from dotmap import DotMap
#     from definitions import ROOT_DIR
#
#     with open(os.path.join(ROOT_DIR, 'assets/configs/training_seg_config.json'), 'r') as f:
#         config = DotMap(json.load(f))
#
#     a = torch.rand(1, 3, 512, 512)
#     feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/mit-b3")
#     # model = MySegformer(config)
#     # a = feature_extractor(images=a, return_tensors="pt")['pixel_values']
#     y = model(a)
#     print(y.logits.shape)
#     # print(y['pixel_values'].shape)
