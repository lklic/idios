import requests
from PIL import Image

import torch
import torchvision.models as models
import torchvision.transforms as transforms

from transformers import CLIPModel, CLIPProcessor


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

class ResNet50:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # We use only the part up to the AdaptiveAvgPool2d layer (included)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def get_image_embedding(self, image):
        image = self.transform(image).unsqueeze(0).to(self.device)
        features = self.model(image)
        # Flatten the output to a single vector of 2048 dimensions
        features = features.view(features.size(0), -1)  # This reshapes the tensor to [1, 2048]
        return features.squeeze().tolist()  # Squeeze to remove the batch dimension and convert to list


embeddings = {
    "vit_b32": CLIP("openai/clip-vit-base-patch32"),
    # "vit_l14": CLIP("openai/clip-vit-large-patch14"),
    "resnet50": ResNet50(),
}
