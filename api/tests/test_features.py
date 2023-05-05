import pytest
import functools

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


@functools.cache
def batch_test_images():
    batch_test_ids = (
        "001,003,009,010,016,017,018,019,020,021,"
        "024,027,030,034,035,044,045,048,050,052,"
        "053,055,057,302,303,307,308,309,311,315,"
        "316,317,318,319,320,322,325,328,334,335,"
        "339,348,350,353,354,355,356,358,359,360,"
        "364,367,368,369,376,377,379,381,384,391,"
        "393,415,416,421,422,430,431,435,439,444,"
        "445,447,448,449,450,451,452,453,454,457,"
        "458,460,461,463,464,465,469,470,477,483,"
        "490,491,497,498,499,601,602,603,605,606,"
    )
    batch_test_urls = [
        f"https://artresearch-iiif.s3.eu-west-1.amazonaws.com/marburg/XKH141{i}.jpg"
        for i in batch_test_ids.split(",")
    ]
    return [load_image_from_url(url) for url in batch_test_urls[:10]]


@pytest.mark.benchmark
def test_individual_extract(benchmark):
    def extract_each():
        for image in batch_test_images():
            features["vit_b32"].extract(image)

    benchmark(extract_each)


@pytest.mark.benchmark
def test_batch_extract(benchmark):
    benchmark(features["vit_b32"].extract, batch_test_images())
