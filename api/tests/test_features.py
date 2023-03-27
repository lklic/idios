import pytest

import random
from PIL import Image

from ..features import load_image_from_url, features


def test_load_image_from_url():
    image = load_image_from_url("https://picsum.photos/512")
    assert image.size == (512, 512)


def test_load_image_from_url_and_resize():
    image = load_image_from_url("https://picsum.photos/1024")
    assert image.size == (1000, 1000)


def test_load_image_from_url_and_reject_small_ones():
    with pytest.raises(ValueError) as exc_info:
        load_image_from_url("https://picsum.photos/128")
    assert (
        str(exc_info.value)
        == "Images must have their dimensions above 150 x 150 pixels"
    )


def test_extract_vit_b32():
    random.seed(2023)
    image_data = bytes([random.randint(0, 255) for _ in range(500 * 500 * 3)])
    image = Image.frombytes("RGB", (500, 500), image_data)

    b32 = features["vit_b32"]
    embedding = b32.extract(image)
    assert 512 == len(embedding)
    assert pytest.approx(-1.00633253128035) == sum(embedding)
