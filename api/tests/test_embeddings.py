import pytest
import functools

import random
from PIL import Image
import math

from ..embeddings import load_image_from_url, embeddings


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


def squared_l2(v):
    return sum([x * x for x in v])


def test_vit_b32_get_image_embedding():
    random.seed(2023)
    image_data = bytes([random.randint(0, 255) for _ in range(500 * 500 * 3)])
    image = Image.frombytes("RGB", (500, 500), image_data)

    b32 = embeddings["vit_b32"]
    embedding = b32.get_image_embedding(image)
    assert pytest.approx(1) == squared_l2(embedding)
    assert 512 == len(embedding)
    assert pytest.approx(-1.00633253128035) == sum(embedding)


def test_vit_b32_get_text_embedding():
    b32 = embeddings["vit_b32"]
    embedding = b32.get_text_embedding("some random text")
    assert pytest.approx(1) == squared_l2(embedding)
    assert 512 == len(embedding)
    assert pytest.approx(1.223632167381993) == sum(embedding)


def test_sift_get_image_embedding():
    random.seed(2023)
    image_data = bytes([random.randint(0, 255) for _ in range(500 * 500 * 3)])
    image = Image.frombytes("RGB", (500, 500), image_data)

    sift = embeddings["sift100"]
    features = sift.get_image_embedding(image)
    assert len(features) > 1
    for descriptor, location in features:
        assert len(descriptor) == 128
    assert pytest.approx(4372) == sum(features[0][0])
    assert "256.36_319.86_190.76" == features[0][1]


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
            embeddings["vit_b32"].get_image_embedding(image)

    benchmark(extract_each)


@pytest.mark.benchmark
def test_batch_extract(benchmark):
    benchmark(embeddings["vit_b32"].get_image_embedding, batch_test_images())


def test_unresized_image():
    url0 = "https://artresearch-iiif.s3.eu-west-1.amazonaws.com/marburg/XKH141001.jpg"
    url1 = "https://artresearch-iiif.s3.eu-west-1.amazonaws.com/marburg/fm3003035.jpg"
    url2 = "https://artresearch-iiif.s3.eu-west-1.amazonaws.com/marburg/gm1159076.jpg"

    images = [load_image_from_url(url) for url in [url0, url1, url2]]
    [embeddings["vit_b32"].get_image_embedding(image) for image in images]
