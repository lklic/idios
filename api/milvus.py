# https://github.com/milvus-io/pymilvus/blob/43c26e8bc983752d21895a51b8d41dc41af5370d/examples/example.py

from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
    MilvusException,
)

import os

from common import embedding_dimensions, MAX_METADATA_LENGTH

DEFAULT_ROOT_PASSWORD = "Milvus"


def ensure_connection():
    if dict(connections.list_connections())["default"] is not None:
        return

    host, _, port = os.environ.get("MILVUS_URL", "").partition(":")
    host = host or "127.0.0.1"
    port = port or "19530"
    password = os.environ.get("MILVUS_PASSWORD", DEFAULT_ROOT_PASSWORD)
    if password == DEFAULT_ROOT_PASSWORD:
        print("!!! Set the MILVUS_PASSWORD environment variable !!!")
        return

    # Change the root password to the specified if the default one is in use
    try:
        connections.connect(host=host, port=port, user="root", password=password)
        utility.list_collections()  # Actually triggers the auth check failure
    except MilvusException as e:
        if "auth check failure" in str(e):
            connections.connect(
                host=host, port=port, user="root", password=DEFAULT_ROOT_PASSWORD
            )
            utility.reset_password("root", DEFAULT_ROOT_PASSWORD, password)
            print("Root password set")
        else:
            raise e
            # raise MilvusException("Failed to connect to Milvus server") from e


# https://github.com/towhee-io/examples/blob/9d199df094e3ec96a0764485ef48285b70be4193/image/reverse_image_search/1_build_image_search_engine.ipynb
def get_collection(collection_name, dim):
    ensure_connection()

    if utility.has_collection(collection_name):
        collection = Collection(name=collection_name)
        collection.load()
        return collection

    embedding_field_name = "embedding"
    fields = [
        FieldSchema(
            name="url",
            dtype=DataType.VARCHAR,
            description="url to image",
            max_length=2083,  # https://docs.pydantic.dev/usage/types/#urls
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(
            name=embedding_field_name,
            dtype=DataType.FLOAT_VECTOR,
            description="image embedding vectors",
            dim=dim,
        ),
        FieldSchema(
            name="metadata",
            dtype=DataType.VARCHAR,
            description="metadata",
            max_length=MAX_METADATA_LENGTH,
        ),
    ]
    schema = CollectionSchema(fields=fields, description="reverse image search")
    collection = Collection(name=collection_name, schema=schema)

    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 2048},
    }
    collection.create_index(field_name=embedding_field_name, index_params=index_params)

    collection.load()
    return collection


def destroy_all_data_from_all_collections_in_the_whole_database():
    ensure_connection()
    for c in utility.list_collections():
        utility.drop_collection(c)


collections = {
    name: get_collection(name, dim) for name, dim in embedding_dimensions.items()
}
metrics = {
    name: collection.index()._index_params["metric_type"]
    for name, collection in collections.items()
}
