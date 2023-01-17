import torchvision

from ..base import BaseModel


class FCN_ResNet50(BaseModel):
    def __init__(self, config):
        super(FCN_ResNet50, self).__init__(config)
        self.net = torchvision.models.segmentation.fcn_resnet50(
            pretrained=self.config.model.pretrained,
            progress=True,
            num_classes=self.n_classes,
        )

    def forward(self, x):
        return self.net(x)
