import pytest
from fastapi.testclient import TestClient

from ..main import app

client = TestClient(app)


TEST_URLS = [
    "https://iiif.itatti.harvard.edu/iiif/2/yashiro!letters-jp!letter_001.pdf/full/full/0/default.jpg",
    "https://iiif.artresearch.net/iiif/2/zeri!151200%21150872_g.jpg/full/full/0/default.jpg",
    "https://ids.lib.harvard.edu/ids/iiif/44405790/full/full/0/native.jpg",
]


def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == "pong"
    assert response.text == '"pong"'


def test_crud():
    response = client.post(
        "/models/vit_b32/add",
        json={
            "url": TEST_URLS[0],
            "metadata": {"tags": ["text"], "language": "japanese"},
        },
    )
    assert response.status_code == 204

    response = client.get("/models/vit_b32/urls")
    assert response.status_code == 200
    assert [TEST_URLS[0]] == response.json()

    # response = client.get("/models/vit_b32/count")
    # assert response.status_code == 200
    # assert 1 == response.json()

    response = client.post(
        "/models/vit_b32/search",
        json={"url": TEST_URLS[1]},
    )
    assert response.status_code == 200
    assert [
        {
            "distance": pytest.approx(0.8834905624389648),
            "metadata": {"tags": ["text"], "language": "japanese"},
            "url": TEST_URLS[0],
        }
    ] == response.json()

    response = client.post(
        "/models/vit_b32/remove",
        json={"url": TEST_URLS[0]},
    )
    assert response.status_code == 204

    response = client.get("/models/vit_b32/urls")
    assert response.status_code == 200
    assert [] == response.json()

    # response = client.get("/models/vit_b32/count")
    # assert response.status_code == 200
    # assert 0 == response.json()


def test_compare():
    response = client.post(
        "/models/vit_b32/compare",
        json={"url_left": TEST_URLS[0], "url_right": TEST_URLS[1]},
    )
    assert response.status_code == 200
    assert pytest.approx(0.8834906264850583) == response.json()


def test_image_too_small():
    response = client.post(
        "/models/vit_b32/add",
        json={"url": "https://picsum.photos/128"},
    )
    print(response.json())
    assert response.status_code == 422
    assert {
        "detail": [
            {
                "loc": ["body", "url"],
                "msg": "Images must have their dimensions above 150 x 150 pixels",
                "type": "parameter_error",
            }
        ]
    } == response.json()


def test_image_left_too_small():
    response = client.post(
        "/models/vit_b32/compare",
        json={"url_left": "https://picsum.photos/128", "url_right": TEST_URLS[1]},
    )
    print(response.json())
    assert response.status_code == 422
    assert {
        "detail": [
            {
                "loc": ["body", "url_left"],
                "msg": "Images must have their dimensions above 150 x 150 pixels",
                "type": "parameter_error",
            }
        ]
    } == response.json()


def test_image_right_too_small():
    response = client.post(
        "/models/vit_b32/compare",
        json={"url_right": "https://picsum.photos/128", "url_left": TEST_URLS[0]},
    )
    print(response.json())
    assert response.status_code == 422
    assert {
        "detail": [
            {
                "loc": ["body", "url_right"],
                "msg": "Images must have their dimensions above 150 x 150 pixels",
                "type": "parameter_error",
            }
        ]
    } == response.json()
