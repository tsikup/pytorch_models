from transformers import ViTModel


def phikon():
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    return model
