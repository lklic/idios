import requests
from PIL import Image

import torch

from transformers import CLIPModel, CLIPProcessor


def load_image_from_url(url):
    MIN_SIZE = 150
    MAX_SIZE = 1000

    with Image.open(requests.get(url, stream=True).raw) as image:
        if min(image.size) < MIN_SIZE:
            raise ValueError("Images must have their dimensions above 150 x 150 pixels")
        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE))
        return image


# https://github.com/kingyiusuen/clip-image-search/blob/80e36511dbe1969d3989989b220c27f08d30a530/clip_image_search/clip_feature_extractor.py
class CLIPFeatures:
    def __init__(self, model_name):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def get_text_features(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = inputs.to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.tolist()
        return text_features

    @torch.no_grad()
    def get_image_features(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.tolist()
        return image_features

    def extract(self, image):
        return self.get_image_features(image)[0]


features = {
    "vit_b32": CLIPFeatures("openai/clip-vit-base-patch32"),
    # "vit_l14": CLIPFeatures("openai/clip-vit-large-patch14"),
}
