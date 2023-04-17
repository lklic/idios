import pytest
from ..commands import commands


TEST_URLS = [
    "https://iiif.itatti.harvard.edu/iiif/2/yashiro!letters-jp!letter_001.pdf/full/full/0/default.jpg",
    "https://iiif.artresearch.net/iiif/2/zeri!151200%21150872_g.jpg/full/full/0/default.jpg",
    "https://ids.lib.harvard.edu/ids/iiif/44405790/full/full/0/native.jpg",
]


def test_crud():
    metadata = {"tags": ["text"], "language": "japanese"}
    commands["insert_image"]("vit_b32", TEST_URLS[0], metadata)

    with pytest.raises(ValueError) as exc_info:
        commands["insert_image"]("vit_b32", TEST_URLS[0], metadata)
    assert str(exc_info.value) == "Url already in collection."

    assert [TEST_URLS[0]] == commands["list_urls"]("vit_b32")

    assert 1 == commands["count"]("vit_b32")

    assert [
        {
            "similarity": pytest.approx(11.650939784262505),
            "metadata": metadata,
            "url": TEST_URLS[0],
        }
    ] == commands["search"]("vit_b32", TEST_URLS[1])

    commands["remove_image"]("vit_b32", TEST_URLS[0])

    assert [] == commands["list_urls"]("vit_b32")

    assert 0 == commands["count"]("vit_b32")


def test_compare():
    assert pytest.approx(11.650939784262505) == commands["compare"](
        "vit_b32", TEST_URLS[0], TEST_URLS[1]
    )


def test_image_too_small():
    with pytest.raises(ValueError) as exc_info:
        commands["insert_image"]("vit_b32", "https://picsum.photos/128", None)
    assert (
        str(exc_info.value)
        == "Images must have their dimensions above 150 x 150 pixels"
    )


def test_image_left_too_small():
    with pytest.raises(ValueError) as exc_info:
        commands["compare"]("vit_b32", "https://picsum.photos/128", TEST_URLS[1])
    assert (
        str(exc_info.value)
        == "Images must have their dimensions above 150 x 150 pixels"
    )


def test_image_right_too_small():
    with pytest.raises(ValueError) as exc_info:
        commands["compare"]("vit_b32", TEST_URLS[0], "https://picsum.photos/128")
    assert (
        str(exc_info.value)
        == "Images must have their dimensions above 150 x 150 pixels"
    )


def test_list_with_cursor():
    commands["insert_image"]("vit_b32", TEST_URLS[0], None)
    commands["insert_image"]("vit_b32", TEST_URLS[1], None)
    assert [TEST_URLS[1], TEST_URLS[0]] == commands["list_urls"]("vit_b32")

    assert [TEST_URLS[0]] == commands["list_urls"]("vit_b32", TEST_URLS[1])

    commands["remove_image"]("vit_b32", TEST_URLS[1])
    commands["remove_image"]("vit_b32", TEST_URLS[0])
