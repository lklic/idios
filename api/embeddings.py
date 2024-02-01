import requests
from PIL import Image

import torch

from transformers import CLIPModel, CLIPProcessor

import cv2 as cv
import numpy as np


def load_image_from_url(url):
    MIN_SIZE = 150
    # not necessarily useful, as feature preprocessing might take care of it
    MAX_SIZE = 1000

    with Image.open(requests.get(url, stream=True).raw) as image:
        if min(image.size) < MIN_SIZE:
            raise ValueError("Images must have their dimensions above 150 x 150 pixels")

        if max(image.size) > MAX_SIZE:
            image.thumbnail((MAX_SIZE, MAX_SIZE))  # noop if <= MAX_SIZE
            return image

        return image.copy()  # ensure the image data is not released


# Based on https://github.com/kingyiusuen/clip-image-search/blob/80e36511dbe1969d3989989b220c27f08d30a530/clip_image_search/clip_feature_extractor.py
class CLIP:
    def __init__(self, model_name):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def get_text_embedding(self, text):
        inputs = self.processor(text=text, return_tensors="pt")
        inputs = inputs.to(self.device)
        text_embedding = self.model.get_text_features(**inputs)
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
        text_embedding = text_embedding.tolist()
        return text_embedding[0]

    @torch.no_grad()
    def get_image_embedding(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        image_embedding = self.model.get_image_features(**inputs)
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
        image_embedding = image_embedding.tolist()
        return image_embedding[0]


class Sift:
    def __init__(self, cardinality):
        self.cardinality = cardinality
        self.sift = cv.SIFT_create()

    def get_image_embedding(self, image):
        gray = cv.cvtColor(np.array(image), cv.COLOR_RGB2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)

        sorted_features = sorted(
            zip(descriptors, keypoints), key=lambda x: x[1].response, reverse=True
        )
        top_features = sorted_features[: self.cardinality]

        return [
            (
                descriptor,
                "_".join(
                    [
                        str(round(x, 2))
                        for x in [keypoint.pt[0], keypoint.pt[1], keypoint.angle]
                    ]
                ),
            )
            for descriptor, keypoint in top_features
        ]


embeddings = {
    "vit_b32": CLIP("openai/clip-vit-base-patch32"),
    # "vit_l14": CLIP("openai/clip-vit-large-patch14"),
    "sift20": Sift(20),
    "sift100": Sift(100),
}
