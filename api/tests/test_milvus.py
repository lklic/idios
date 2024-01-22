from ..milvus import (
    ensure_connection,
    get_collection,
    INDEX_PARAMS,
)

from pymilvus import (
    connections,
    utility,
)

import pytest
import uuid
from unittest.mock import patch


@pytest.fixture
def mock_index_params():
    collection_name = "test_collection_to_remove"
    with patch.dict(
        INDEX_PARAMS,
        {
            collection_name: {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 2048},
            }
        },
    ):
        yield collection_name


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


def test_get_collection(mock_index_params):
    ensure_connection()  # for utilities
    if mock_index_params in utility.list_collections():
        utility.drop_collection(mock_index_params)

    assert mock_index_params not in utility.list_collections()
    collection = get_collection(mock_index_params, 42)
    assert mock_index_params in utility.list_collections()
    assert {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 2048},
    } == collection.index()._index_params
    utility.drop_collection(mock_index_params)
    assert mock_index_params not in utility.list_collections()


def test_query_results_are_sorted_by_pk(mock_index_params):
    if mock_index_params in utility.list_collections():
        utility.drop_collection(mock_index_params)
    assert mock_index_params not in utility.list_collections()
    collection = get_collection(mock_index_params, 42)

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

    utility.drop_collection(mock_index_params)


N_ENTITIES = 1000
DIM = 512
embeddings = [[0] * DIM for _ in range(N_ENTITIES)]
urls = [str(uuid.uuid4()) for _ in range(N_ENTITIES)]
metadatas = ["null"] * N_ENTITIES


@pytest.mark.benchmark
def test_batch_insert(benchmark, mock_index_params):
    ensure_connection()  # for utilities
    utility.drop_collection(mock_index_params)
    collection = get_collection(mock_index_params, DIM)
    benchmark(collection.insert, [urls, embeddings, metadatas])


@pytest.mark.benchmark
def test_individual_insert(benchmark, mock_index_params):
    ensure_connection()  # for utilities
    utility.drop_collection(mock_index_params)
    collection = get_collection(mock_index_params, DIM)

    def insert_each():
        for params in zip(urls, embeddings, metadatas):
            collection.insert([[param] for param in params])

    benchmark(insert_each)
