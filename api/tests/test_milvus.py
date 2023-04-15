from ..milvus import (
    ensure_connection,
    get_collection,
)

from pymilvus import (
    connections,
    utility,
)

import pytest
import uuid


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


def test_query_results_are_sorted_by_pk():
    collection_name = "test_collection_to_remove"
    if collection_name in utility.list_collections():
        utility.drop_collection(collection_name)
    assert collection_name not in utility.list_collections()
    collection = get_collection(collection_name, 42)

    urls = [str(uuid.uuid4()) for _ in range(10)]
    if sorted(urls) == urls:
        urls[0], urls[1] = urls[1], urls[0]

    for url in urls:
        collection.insert([[url], [[0] * 42], ["null"]])

    assert sorted(urls) == [
        search_result["url"]
        for search_result in collection.query(
            f'url > ""',
            consistency_level="Strong",
        )
    ]

    utility.drop_collection(collection_name)
