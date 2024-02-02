import pytest
from ..commands import commands
from embeddings import embeddings
from milvus import collections, get_collection
from common import DIMENSIONS, INDEX_PARAMS, SEARCH_PARAMS, CARDINALITIES
from pymilvus import utility
import numpy as np

from unittest.mock import patch


TEST_URLS = [
    "https://iiif.itatti.harvard.edu/iiif/2/yashiro!letters-jp!letter_001.pdf/full/full/0/default.jpg",
    "https://iiif.artresearch.net/iiif/2/zeri!151200%21150872_g.jpg/full/full/0/default.jpg",
    "https://ids.lib.harvard.edu/ids/iiif/44405790/full/full/0/native.jpg",
]
SMALL_URL = "https://picsum.photos/128"


@pytest.fixture
def mock_model():
    REAL_MODEL_NAME = "vit_b32"
    TEST_MODEL_NAME = "mock_" + REAL_MODEL_NAME
    if utility.has_collection(TEST_MODEL_NAME):
        utility.drop_collection(TEST_MODEL_NAME)

    with patch.dict(
        DIMENSIONS, {TEST_MODEL_NAME: DIMENSIONS[REAL_MODEL_NAME]}
    ), patch.dict(
        INDEX_PARAMS, {TEST_MODEL_NAME: INDEX_PARAMS[REAL_MODEL_NAME]}
    ), patch.dict(
        SEARCH_PARAMS, {TEST_MODEL_NAME: SEARCH_PARAMS[REAL_MODEL_NAME]}
    ), patch.dict(
        CARDINALITIES, {TEST_MODEL_NAME: CARDINALITIES[REAL_MODEL_NAME]}
    ):
        with patch.dict(
            collections,
            {
                TEST_MODEL_NAME: get_collection(
                    TEST_MODEL_NAME, DIMENSIONS[REAL_MODEL_NAME]
                )
            },
        ), patch.dict(embeddings, {TEST_MODEL_NAME: embeddings[REAL_MODEL_NAME]}):
            yield TEST_MODEL_NAME


@pytest.fixture
def mock_features():
    REAL_MODEL_NAME = "sift20"
    TEST_MODEL_NAME = "mock_" + REAL_MODEL_NAME
    if utility.has_collection(TEST_MODEL_NAME):
        utility.drop_collection(TEST_MODEL_NAME)

    with patch.dict(
        DIMENSIONS, {TEST_MODEL_NAME: DIMENSIONS[REAL_MODEL_NAME]}
    ), patch.dict(
        INDEX_PARAMS, {TEST_MODEL_NAME: INDEX_PARAMS[REAL_MODEL_NAME]}
    ), patch.dict(
        SEARCH_PARAMS, {TEST_MODEL_NAME: SEARCH_PARAMS[REAL_MODEL_NAME]}
    ), patch.dict(
        CARDINALITIES, {TEST_MODEL_NAME: CARDINALITIES[REAL_MODEL_NAME]}
    ):
        with patch.dict(
            collections,
            {
                TEST_MODEL_NAME: get_collection(
                    TEST_MODEL_NAME, DIMENSIONS[REAL_MODEL_NAME]
                )
            },
        ), patch.dict(embeddings, {TEST_MODEL_NAME: embeddings[REAL_MODEL_NAME]}):
            yield TEST_MODEL_NAME


def test_crud(mock_model):
    metadata = {"tags": ["text"], "language": "japanese"}

    assert [] == commands["list_images"](mock_model)

    commands["insert_images"](mock_model, [TEST_URLS[0]], [metadata])

    assert [TEST_URLS[0]] == commands["list_images"](mock_model)

    assert 1 == commands["count"](mock_model)

    assert [
        {
            "similarity": pytest.approx(55.82546989213125),
            "metadata": metadata,
            "url": TEST_URLS[0],
        }
    ] == commands["search_by_url"](mock_model, TEST_URLS[1])
    assert [
        {
            "similarity": pytest.approx(29.19090986251831, rel=1e-3),
            "metadata": metadata,
            "url": TEST_URLS[0],
        }
    ] == commands["search_by_text"](mock_model, "a black and white text in japanese")
    assert [
        {
            "similarity": pytest.approx(17.36249327659607),
            "metadata": metadata,
            "url": TEST_URLS[0],
        }
    ] == commands["search_by_text"](mock_model, "a cute colorful cat")

    commands["remove_images"](mock_model, [TEST_URLS[0]])

    assert [] == commands["list_images"](mock_model)

    assert 0 == commands["count"](mock_model)


def test_crud_local_features(mock_features):
    metadata = {"tags": ["text"], "language": "japanese"}

    assert [] == commands["list_images"](mock_features)

    commands["insert_images"](mock_features, [TEST_URLS[0]], [metadata])

    assert [TEST_URLS[0]] == commands["list_images"](mock_features)

    assert 1 == commands["count"](mock_features)

    assert [
        {
            "similarity": 100,
            "metadata": metadata,
            "url": TEST_URLS[0],
        }
    ] == commands["search_by_url"](mock_features, TEST_URLS[1])

    commands["remove_images"](mock_features, [TEST_URLS[0]])

    assert [] == commands["list_images"](mock_features)

    assert 0 == commands["count"](mock_features)


def test_insert_nothing(mock_model):
    commands["insert_images"](mock_model, [], [])


def test_search_more_results(mock_model):
    urls = [f"url{i}" for i in range(1000)]
    commands["insert_images"](
        mock_model,
        urls,
        [None] * 1000,
        [[0] * 512] * 1000,
    )
    results = commands["search_by_url"](mock_model, TEST_URLS[1])
    assert 10 == len(results)
    results = commands["search_by_url"](mock_model, TEST_URLS[1], 100)
    assert 100 == len(results)
    commands["remove_images"](mock_model, urls)


def test_remove_multiple_images(mock_model):
    assert [] == commands["list_images"](mock_model)

    commands["insert_images"](
        mock_model,
        [TEST_URLS[0], TEST_URLS[1]],
        [None] * 2,
        [[0] * 512] * 2,
    )

    assert [TEST_URLS[1], TEST_URLS[0]] == commands["list_images"](mock_model)

    commands["remove_images"](mock_model, [TEST_URLS[0], TEST_URLS[1]])

    assert [] == commands["list_images"](mock_model)

    assert 0 == commands["count"](mock_model)


def test_insert_without_replacing(mock_model):
    commands["insert_images"](
        mock_model,
        [TEST_URLS[0]],
        [None],
        [[0] * 512],
    )
    assert [TEST_URLS[0]] == commands["list_images"](mock_model)

    assert {"added": [TEST_URLS[1]], "found": [TEST_URLS[0]]} == commands[
        "insert_images"
    ](
        mock_model,
        [TEST_URLS[0], TEST_URLS[1]],
        [None] * 2,
        None,
        replace_existing=False,
    )
    result = commands["list_images"](mock_model, output_fields=["url", "embedding"])
    assert result[0]["url"] == TEST_URLS[1]
    assert result[1]["url"] == TEST_URLS[0]
    assert result[1]["embedding"] == [0] * 512

    commands["remove_images"](mock_model, [TEST_URLS[0], TEST_URLS[1]])


def test_compare():
    assert pytest.approx(55.82546989213125) == commands["compare"](
        "vit_b32", TEST_URLS[0], TEST_URLS[1]
    )


def test_image_too_small():
    with pytest.raises(ValueError) as exc_info:
        commands["insert_images"]("vit_b32", [SMALL_URL], [None])
    assert (
        str(exc_info.value)
        == "Images must have their dimensions above 150 x 150 pixels"
    )


def test_image_left_too_small():
    with pytest.raises(ValueError) as exc_info:
        commands["compare"]("vit_b32", SMALL_URL, TEST_URLS[1])
    assert (
        str(exc_info.value)
        == "Images must have their dimensions above 150 x 150 pixels"
    )


def test_image_right_too_small():
    with pytest.raises(ValueError) as exc_info:
        commands["compare"]("vit_b32", TEST_URLS[0], SMALL_URL)
    assert (
        str(exc_info.value)
        == "Images must have their dimensions above 150 x 150 pixels"
    )


def test_list_with_cursor(mock_model):
    commands["insert_images"](
        mock_model,
        [TEST_URLS[0], TEST_URLS[1]],
        [None] * 2,
        [[0] * 512] * 2,
    )
    assert [TEST_URLS[1], TEST_URLS[0]] == commands["list_images"](mock_model)

    assert [TEST_URLS[0]] == commands["list_images"](mock_model, TEST_URLS[1])

    commands["remove_images"](mock_model, [TEST_URLS[0], TEST_URLS[1]])


def test_list_with_limit(mock_model):
    commands["insert_images"](
        mock_model,
        [TEST_URLS[0], TEST_URLS[1]],
        [None] * 2,
        [[0] * 512] * 2,
    )
    assert [TEST_URLS[1], TEST_URLS[0]] == commands["list_images"](mock_model)

    assert [TEST_URLS[1]] == commands["list_images"](mock_model, None, 1)

    commands["remove_images"](mock_model, [TEST_URLS[0], TEST_URLS[1]])


def test_list_with_cursor_and_limit(mock_model):
    commands["insert_images"](
        mock_model,
        [TEST_URLS[0], TEST_URLS[1], TEST_URLS[2]],
        [None] * 3,
        [[0] * 512] * 3,
    )
    assert [
        TEST_URLS[2],
        TEST_URLS[1],
        TEST_URLS[0],
    ] == commands[
        "list_images"
    ](mock_model)

    assert [TEST_URLS[1]] == commands["list_images"](mock_model, TEST_URLS[2], 1)

    commands["remove_images"](mock_model, [TEST_URLS[0], TEST_URLS[1], TEST_URLS[2]])


def test_list_with_output_fields(mock_model):
    embedding = np.random.rand(512)
    embedding = embedding / np.sum(embedding)

    commands["insert_images"](
        mock_model,
        [TEST_URLS[0]],
        [{"meta": "data"}],
        [embedding],
    )

    result = commands["list_images"](
        mock_model, "", 10, ["url", "embedding", "metadata"]
    )

    assert all(isinstance(value, float) for value in result[0]["embedding"])
    assert np.all(result[0]["embedding"] == pytest.approx(embedding))
    assert result[0]["url"] == TEST_URLS[0]
    assert result[0]["metadata"] == {"meta": "data"}

    commands["remove_images"](mock_model, [TEST_URLS[0]])


def test_ping():
    assert commands["ping"]() == "pong"
