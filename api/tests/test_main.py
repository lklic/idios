import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

from ..main import app

client = TestClient(app)


@pytest.fixture
def mock_rpc():
    with patch("rpc_client.RpcClient.__call__") as mock_rpc:
        yield mock_rpc


def test_ping(mock_rpc):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == "pong"
    assert response.text == '"pong"'
    mock_rpc.assert_not_called()


def test_ping_with_rpc(mock_rpc):
    mock_rpc.return_value = "pong"
    response = client.get("/ping?rpc=1")
    assert response.status_code == 200
    assert response.json() == "pong"


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


def test_search_add_image_success(mock_rpc):
    mock_rpc.return_value = {"added": ["http://example.com/image.jpg"], "found": []}
    response = client.post(
        "/models/vit_b32/search_add",
        json={
            "url": "http://example.com/image.jpg",
            "metadata": {"tags": ["cat", "cute"]},
        },
    )
    assert response.status_code == 204
    mock_rpc.assert_called_once_with(
        "insert_images",
        [
            "vit_b32",
            ["http://example.com/image.jpg"],
            [{"tags": ["cat", "cute"]}],
            None,
            False,
        ],
    )


def test_search_add_image_conflict(mock_rpc):
    mock_rpc.return_value = {"added": [], "found": ["http://example.com/image.jpg"]}
    response = client.post(
        "/models/vit_b32/search_add",
        json={
            "url": "http://example.com/image.jpg",
            "metadata": {"tags": ["cat", "cute"]},
        },
    )
    assert response.status_code == 409
    assert response.json() == {"detail": "Image already inserted"}
    mock_rpc.assert_called_once_with(
        "insert_images",
        [
            "vit_b32",
            ["http://example.com/image.jpg"],
            [{"tags": ["cat", "cute"]}],
            None,
            False,
        ],
    )


def test_search_by_url_success(mock_rpc):
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
        "search_by_url", ["vit_b32", "http://example.com/query.jpg", 10]
    )


def test_search_by_url_success_with_limit(mock_rpc):
    mock_rpc.return_value = [
        {
            "url": "http://example.com/image1.jpg",
            "metadata": {"tags": ["cat", "cute"]},
            "similarity": 10,
        }
    ] * 100
    response = client.post(
        "/models/vit_b32/search",
        json={"url": "http://example.com/query.jpg", "limit": 100},
    )
    assert response.status_code == 200
    assert len(response.json()) == 100
    mock_rpc.assert_called_once_with(
        "search_by_url", ["vit_b32", "http://example.com/query.jpg", 100]
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
        "search_by_url", ["vit_b32", "http://example.com/query.jpg", 10]
    )


def test_search_by_url_returns_422_when_invalid_url(mock_rpc):
    response = client.post(
        "/models/vit_b32/search",
        json={"url": "not_a_url"},
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


def test_search_by_text_success(mock_rpc):
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
        json={"text": "cute cat"},
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
    mock_rpc.assert_called_once_with("search_by_text", ["vit_b32", "cute cat", 10])


def test_search_by_text_success_with_limit(mock_rpc):
    mock_rpc.return_value = [
        {
            "url": "http://example.com/image1.jpg",
            "metadata": {"tags": ["cat", "cute"]},
            "similarity": 10,
        }
    ] * 100
    response = client.post(
        "/models/vit_b32/search",
        json={"text": "some text", "limit": 100},
    )
    assert response.status_code == 200
    assert len(response.json()) == 100
    mock_rpc.assert_called_once_with("search_by_text", ["vit_b32", "some text", 100])


def test_search_returns_422_whithout_query(mock_rpc):
    response = client.post(
        "/models/vit_b32/search",
        json={},
    )
    assert response.status_code == 422
    assert response.json() == {"detail": "Either 'text' or 'url' must be provided."}
    mock_rpc.assert_not_called()


def test_search_returns_500_when_rpc_error(mock_rpc):
    mock_rpc.side_effect = RuntimeError("Invalid argument")
    response = client.post(
        "/models/vit_b32/search",
        json={"url": "http://example.com/query.jpg"},
    )
    assert response.status_code == 500
    mock_rpc.assert_called_once_with(
        "search_by_url", ["vit_b32", "http://example.com/query.jpg", 10]
    )


def test_compare_returns_similarity(mock_rpc):
    mock_rpc.return_value = 0.42
    response = client.post(
        "/models/vit_b32/compare",
        json={"url": "http://left.org", "other": "http://right.org"},
    )
    assert response.status_code == 200
    assert pytest.approx(0.42) == response.json()
    mock_rpc.assert_called_once_with(
        "compare", ["vit_b32", "http://left.org", "http://right.org"]
    )


def test_compare_returns_422_when_invalid_url(mock_rpc):
    response = client.post(
        "/models/vit_b32/compare",
        json={"url": "not_a_url", "other": "http://right.org"},
    )
    assert response.status_code == 422
    mock_rpc.assert_not_called()


def test_compare_returns_500_when_rpc_error(mock_rpc):
    mock_rpc.side_effect = RuntimeError("Internal server error")
    response = client.post(
        "/models/vit_b32/compare",
        json={"url": "http://left.org", "other": "http://right.org"},
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
        "remove_images", ["vit_b32", ["http://example.com/image.jpg"]]
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
        "remove_images", ["vit_b32", ["http://example.com/image.jpg"]]
    )


def test_list_images_returns_urls(mock_rpc):
    mock_rpc.return_value = [
        "http://example.com/image1.jpg",
        "http://example.com/image2.jpg",
    ]
    response = client.post("/models/vit_b32/urls")
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0] == "http://example.com/image1.jpg"
    assert response.json()[1] == "http://example.com/image2.jpg"
    mock_rpc.assert_called_once_with("list_images", ["vit_b32", None, None])


def test_list_images_with_cursor_returns_urls(mock_rpc):
    mock_rpc.return_value = [
        "http://example.com/image1.jpg",
        "http://example.com/image2.jpg",
    ]
    response = client.post("/models/vit_b32/urls", json={"cursor": "some url"})
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0] == "http://example.com/image1.jpg"
    assert response.json()[1] == "http://example.com/image2.jpg"
    mock_rpc.assert_called_once_with("list_images", ["vit_b32", "some url", None])


def test_list_images_with_limit_returns_urls(mock_rpc):
    mock_rpc.return_value = [
        "http://example.com/image1.jpg",
        "http://example.com/image2.jpg",
    ]
    response = client.post("/models/vit_b32/urls", json={"limit": 10})
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0] == "http://example.com/image1.jpg"
    assert response.json()[1] == "http://example.com/image2.jpg"
    mock_rpc.assert_called_once_with("list_images", ["vit_b32", None, 10])


def test_list_images_with_cursor_and_limit_returns_urls(mock_rpc):
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
    mock_rpc.assert_called_once_with("list_images", ["vit_b32", "some url", 10])


def test_list_images_returns_500_when_rpc_error(mock_rpc):
    mock_rpc.side_effect = RuntimeError("Internal server error")
    response = client.post("/models/vit_b32/urls")
    assert response.status_code == 500
    mock_rpc.assert_called_once_with("list_images", ["vit_b32", None, None])


def test_count_success(mock_rpc):
    mock_rpc.return_value = 42
    response = client.get("/models/vit_b32/count")
    assert response.status_code == 200
    assert response.json() == 42
    mock_rpc.assert_called_once_with("count", ["vit_b32"])


def test_list_images_returns_500_when_rpc_error(mock_rpc):
    mock_rpc.side_effect = RuntimeError("Internal server error")
    response = client.get("/models/vit_b32/count")
    assert response.status_code == 500
    mock_rpc.assert_called_once_with("count", ["vit_b32"])


def test_dump_success(mock_rpc):
    mock_rpc.return_value = [
        {
            "url": "http://example.com/image.jpg",
            "metadata": {"tags": ["cat", "cute"]},
            "embedding": [1.0],
        },
    ]
    response = client.post(
        "/models/vit_b32/dump",
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
        "list_images",
        [
            "vit_b32",
            None,
            None,
            ["url", "embedding", "metadata"],
        ],
    )


def test_restore_success(mock_rpc):
    mock_rpc.return_value = None
    response = client.post(
        "/models/vit_b32/restore",
        json=[
            {
                "url": "http://example.com/image.jpg",
                "metadata": {"tags": ["cat", "cute"]},
                "embedding": [1.0, 2.0, 3.0],
            },
            {
                "url": "http://example.com/image2.jpg",
                "metadata": {"tags": ["dog", "cuter"]},
                "embedding": [],
            },
        ],
    )
    assert response.status_code == 204
    mock_rpc.assert_called_once_with(
        "insert_images",
        [
            "vit_b32",
            ["http://example.com/image.jpg", "http://example.com/image2.jpg"],
            [{"tags": ["cat", "cute"]}, {"tags": ["dog", "cuter"]}],
            [[1.0, 2.0, 3.0], []],
        ],
    )
