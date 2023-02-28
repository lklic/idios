from ..milvus import (
    ensure_connection,
    get_collection,
)

from pymilvus import (
    connections,
    utility,
)

import pytest


def test_ensure_connection():
    # depending on the order of test, the connection might be already created
    if None != dict(connections.list_connections())["default"]:
        pytest.skip("ensure_connection called by another test")

    assert 1 == len(connections.list_connections())
    ensure_connection()
    print(connections.list_connections())
    assert None != dict(connections.list_connections())["default"]
    ensure_connection()
    assert 1 == len(connections.list_connections())
    assert None != dict(connections.list_connections())["default"]


def test_get_collection():
    collection_name = "test_collection_to_remove"
    ensure_connection()  # for utilities
    if collection_name in utility.list_collections():
        utility.drop_collection(collection_name)

    assert collection_name not in utility.list_collections()
    collection = get_collection(collection_name, 42)
    assert collection_name in utility.list_collections()
    assert {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 2048},
    } == collection.index()._index_params
    utility.drop_collection(collection_name)
    assert collection_name not in utility.list_collections()
