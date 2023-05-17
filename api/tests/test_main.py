import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from ..main import app

client = TestClient(app)


@pytest.fixture
def mock_rpc():
    with patch("rpc_client.RpcClient.__call__") as mock_rpc:
        yield mock_rpc


def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == "pong"
    assert response.text == '"pong"'


def test_add_image_success(mock_rpc):
    mock_rpc.return_value = None
    response = client.post(
        "/models/vit_b32/add",
        json={
            "url": "http://example.com/image.jpg",
            "metadata": {"tags": ["cat", "cute"]},
        },
    )
    assert response.status_code == 204
    mock_rpc.assert_called_once_with(
        "insert_images",
        ["vit_b32", ["http://example.com/image.jpg"], [{"tags": ["cat", "cute"]}]],
    )


def test_add_image_success_no_metadata(mock_rpc):
    mock_rpc.return_value = None
    response = client.post(
        "/models/vit_b32/add",
        json={"url": "http://example.com/image.jpg"},
    )
    assert response.status_code == 204
    mock_rpc.assert_called_once_with(
        "insert_images",
        ["vit_b32", ["http://example.com/image.jpg"], [None]],
    )


def test_add_image_invalid_url(mock_rpc):
    response = client.post(
        "/models/vit_b32/add",
        json={"url": "invalid_url", "metadata": {"tags": ["cat", "cute"]}},
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "loc": ["body", "url"],
                "msg": "invalid or missing URL scheme",
                "type": "value_error.url.scheme",
            }
        ]
    }
    mock_rpc.assert_not_called()


def test_add_image_invalid_metadata_json(mock_rpc):
    response = client.post(
        "/models/vit_b32/add",
        content='{"url": "https://example.com/image.jpg", "metadata": "invalid json string"}',
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "loc": ["body", "metadata"],
                "msg": "value is not a valid dict",
                "type": "type_error.dict",
            }
        ]
    }
    mock_rpc.assert_not_called()


def test_add_image_metadata_too_long(mock_rpc):
    response = client.post(
        "/models/vit_b32/add",
        json={
            "url": "http://example.com/image.jpg",
            "metadata": {
                "chicken?": "chicken" * (int(2**16 / 7) - 2),
            },
        },
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": [
            {
                "loc": ["body", "metadata"],
                "msg": "metadata json too long (65536 > 65535)",
                "type": "value_error.metadata_json_too_long",
            }
        ]
    }
    mock_rpc.assert_not_called()


def test_add_image_small_dimensions(mock_rpc):
    mock_rpc.side_effect = ValueError("Image size too small")
    response = client.post(
        "/models/vit_b32/add",
        json={"url": "http://example.com/image.jpg"},
    )
    assert response.status_code == 422
    assert response.json() == {
        "detail": [{"msg": "Image size too small", "type": "parameter_error"}]
    }
    mock_rpc.assert_called_once_with(
        "insert_images",
        ["vit_b32", ["http://example.com/image.jpg"], [None]],
    )


def test_add_image_server_error(mock_rpc):
    mock_rpc.side_effect = RuntimeError("Server error")
    response = client.post(
        "/models/vit_b32/add",
        json={"url": "http://example.com/image.jpg"},
    )
    assert response.status_code == 500
    assert response.json() == {
        "detail": [{"msg": "Server error", "type": "server_error"}]
    }
    mock_rpc.assert_called_once_with(
        "insert_images",
        ["vit_b32", ["http://example.com/image.jpg"], [None]],
    )


def test_search_success(mock_rpc):
    mock_rpc.return_value = [
        {
            "url": "http://example.com/image1.jpg",
            "metadata": {"tags": ["cat", "cute"]},
            "similarity": 10,
        },
        {
            "url": "http://example.com/image2.jpg",
            "metadata": None,
            "similarity": 20,
        },
    ]
    response = client.post(
        "/models/vit_b32/search",
        json={"url": "http://example.com/query.jpg"},
    )
    assert response.status_code == 200
    assert response.json() == [
        {
            "url": "http://example.com/image1.jpg",
            "metadata": {"tags": ["cat", "cute"]},
            "similarity": 10,
        },
        {
            "url": "http://example.com/image2.jpg",
            "metadata": None,
            "similarity": 20,
        },
    ]
    mock_rpc.assert_called_once_with(
        "search", ["vit_b32", "http://example.com/query.jpg"]
    )


def test_search_empty(mock_rpc):
    mock_rpc.return_value = []
    response = client.post(
        "/models/vit_b32/search",
        json={"url": "http://example.com/query.jpg"},
    )
    assert response.status_code == 200
    assert response.json() == []
    mock_rpc.assert_called_once_with(
        "search", ["vit_b32", "http://example.com/query.jpg"]
    )


def test_search_returns_422_when_invalid_url(mock_rpc):
    response = client.post(
        "/models/vit_b32/search",
        json={"url": "not_a_url"},
    )
    assert response.status_code == 422
    mock_rpc.assert_not_called()


def test_search_returns_500_when_rpc_error(mock_rpc):
    mock_rpc.side_effect = RuntimeError("Invalid argument")
    response = client.post(
        "/models/vit_b32/search",
        json={"url": "http://example.com/query.jpg"},
    )
    assert response.status_code == 500
    mock_rpc.assert_called_once_with(
        "search", ["vit_b32", "http://example.com/query.jpg"]
    )


def test_compare_returns_similarity(mock_rpc):
    mock_rpc.return_value = 0.42
    response = client.post(
        "/models/vit_b32/compare",
        json={"url_left": "http://left.org", "url_right": "http://right.org"},
    )
    assert response.status_code == 200
    assert pytest.approx(0.42) == response.json()
    mock_rpc.assert_called_once_with(
        "compare", ["vit_b32", "http://left.org", "http://right.org"]
    )


def test_compare_returns_422_when_invalid_url(mock_rpc):
    response = client.post(
        "/models/vit_b32/compare",
        json={"url_left": "not_a_url", "url_right": "http://right.org"},
    )
    assert response.status_code == 422
    mock_rpc.assert_not_called()


def test_compare_returns_500_when_rpc_error(mock_rpc):
    mock_rpc.side_effect = RuntimeError("Internal server error")
    response = client.post(
        "/models/vit_b32/compare",
        json={"url_left": "http://left.org", "url_right": "http://right.org"},
    )
    assert response.status_code == 500
    mock_rpc.assert_called_once_with(
        "compare", ["vit_b32", "http://left.org", "http://right.org"]
    )


def test_remove_image_returns_204(mock_rpc):
    mock_rpc.return_value = None
    response = client.post(
        "/models/vit_b32/remove",
        json={"url": "http://example.com/image.jpg"},
    )
    assert response.status_code == 204
    mock_rpc.assert_called_once_with(
        "remove_image", ["vit_b32", "http://example.com/image.jpg"]
    )


def test_remove_image_returns_422_when_invalid_url(mock_rpc):
    response = client.post(
        "/models/vit_b32/remove",
        json={"url": "not_a_url"},
    )
    assert response.status_code == 422
    mock_rpc.assert_not_called()


def test_remove_image_returns_500_when_rpc_error(mock_rpc):
    mock_rpc.side_effect = RuntimeError("Invalid argument")
    response = client.post(
        "/models/vit_b32/remove",
        json={"url": "http://example.com/image.jpg"},
    )
    assert response.status_code == 500
    mock_rpc.assert_called_once_with(
        "remove_image", ["vit_b32", "http://example.com/image.jpg"]
    )


def test_list_urls_returns_urls(mock_rpc):
    mock_rpc.return_value = [
        "http://example.com/image1.jpg",
        "http://example.com/image2.jpg",
    ]
    response = client.post("/models/vit_b32/urls")
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0] == "http://example.com/image1.jpg"
    assert response.json()[1] == "http://example.com/image2.jpg"
    mock_rpc.assert_called_once_with("list_urls", ["vit_b32", None, None])


def test_list_urls_with_cursor_returns_urls(mock_rpc):
    mock_rpc.return_value = [
        "http://example.com/image1.jpg",
        "http://example.com/image2.jpg",
    ]
    response = client.post("/models/vit_b32/urls", json={"cursor": "some url"})
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0] == "http://example.com/image1.jpg"
    assert response.json()[1] == "http://example.com/image2.jpg"
    mock_rpc.assert_called_once_with("list_urls", ["vit_b32", "some url", None])


def test_list_urls_with_limit_returns_urls(mock_rpc):
    mock_rpc.return_value = [
        "http://example.com/image1.jpg",
        "http://example.com/image2.jpg",
    ]
    response = client.post("/models/vit_b32/urls", json={"limit": 10})
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0] == "http://example.com/image1.jpg"
    assert response.json()[1] == "http://example.com/image2.jpg"
    mock_rpc.assert_called_once_with("list_urls", ["vit_b32", None, 10])


def test_list_urls_with_cursor_and_limit_returns_urls(mock_rpc):
    mock_rpc.return_value = [
        "http://example.com/image1.jpg",
        "http://example.com/image2.jpg",
    ]
    response = client.post(
        "/models/vit_b32/urls", json={"cursor": "some url", "limit": 10}
    )
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0] == "http://example.com/image1.jpg"
    assert response.json()[1] == "http://example.com/image2.jpg"
    mock_rpc.assert_called_once_with("list_urls", ["vit_b32", "some url", 10])


def test_list_urls_returns_500_when_rpc_error(mock_rpc):
    mock_rpc.side_effect = RuntimeError("Internal server error")
    response = client.post("/models/vit_b32/urls")
    assert response.status_code == 500
    mock_rpc.assert_called_once_with("list_urls", ["vit_b32", None, None])


def test_count_success(mock_rpc):
    mock_rpc.return_value = 42
    response = client.get("/models/vit_b32/count")
    assert response.status_code == 200
    assert response.json() == 42
    mock_rpc.assert_called_once_with("count", ["vit_b32"])


def test_list_urls_returns_500_when_rpc_error(mock_rpc):
    mock_rpc.side_effect = RuntimeError("Internal server error")
    response = client.get("/models/vit_b32/count")
    assert response.status_code == 500
    mock_rpc.assert_called_once_with("count", ["vit_b32"])


def test_export_success(mock_rpc):
    mock_rpc.return_value = [
        {
            "url": "http://example.com/image.jpg",
            "metadata": {"tags": ["cat", "cute"]},
            "embedding": [1.0],
        },
    ]
    response = client.post(
        "/models/vit_b32/export",
    )
    assert response.status_code == 200
    assert response.json() == [
        {
            "url": "http://example.com/image.jpg",
            "metadata": {"tags": ["cat", "cute"]},
            "embedding": [1.0],
        },
    ]
    mock_rpc.assert_called_once_with(
        "list_urls",
        [
            "vit_b32",
            None,
            None,
            ["url", "embedding", "metadata"],
        ],
    )
