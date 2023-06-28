import json
import numpy as np

from embeddings import load_image_from_url, embeddings
from milvus import collections, metrics


def format_url_list(urls):
    quoted_urls = [f'"{url}"' for url in urls]
    return f'[{",".join(quoted_urls)}]'


def insert_images(
    model_name, urls, metadatas, image_embeddings=None, replace_existing=True
):
    existing_urls = []
    if not replace_existing:
        existing_urls = [
            search_result["url"]
            for search_result in collections[model_name].query(
                f"url in {format_url_list(urls)}",
                consistency_level="Strong",  # https://milvus.io/docs/consistency.md
            )
        ]

    new_urls = [url for url in urls if url not in existing_urls]
    image_embeddings = (
        [embeddings[model_name].extract(load_image_from_url(url)) for url in new_urls]
        if image_embeddings is None
        else [
            embedding
            for url, embedding in zip(urls, image_embeddings)
            if url not in existing_urls
        ]
    )
    metadatas = [
        json.dumps(metadata)
        for url, metadata in zip(urls, metadatas)
        if url not in existing_urls
    ]

    if len(new_urls) > 0:
        collections[model_name].insert([new_urls, image_embeddings, metadatas])

    return {
        "added": new_urls,
        "found": existing_urls,
    }


def search(model_name, url):
    embedding = embeddings[model_name].extract(load_image_from_url(url))
    search_results = collections[model_name].search(
        data=[embedding],
        anns_field="embedding",
        param={
            "metric_type": metrics[model_name],
            "params": {"nprobe": 10},
        },
        output_fields=["metadata"],
        limit=10,
        expr=None,
        consistency_level="Strong",  # https://milvus.io/docs/consistency.md
    )
    return [
        {
            "url": hit.id,
            "metadata": json.loads(hit.entity.get("metadata")),
            "similarity": 100 * (1 - hit.distance),
        }
        for hit in search_results[0]
    ]


def compare(model_name, url_left, url_right):
    # alternatively, we could first try to fetch the embeddings from milvus in
    # case their computation is significantly more expensive than a query
    left = embeddings[model_name].extract(load_image_from_url(url_left))
    right = embeddings[model_name].extract(load_image_from_url(url_right))

    # calc_distance() has been removed from milvus
    # it's a bit overkill anyway if we don't compare with vectors from the db
    if metrics[model_name] == "L2":
        # _squared_ L2, to be consistent with the distances in milvus' search
        return 100 * (1 - np.sum(np.square(np.array(left) - np.array(right))))

    raise RuntimeError(
        "Distance calculation has not been implemented in the API. "
        "Please contact the administrator."
    )


def list_images(model_name, cursor="", limit=None, output_fields=None):
    def prepare(entry):
        if "embedding" in entry:
            entry["embedding"] = [float(x) for x in entry["embedding"]]
        if "metadata" in entry:
            entry["metadata"] = json.loads(entry["metadata"])
        return entry

    return [
        search_result["url"] if output_fields is None else prepare(search_result)
        for search_result in collections[model_name].query(
            f'url > "{cursor}"',
            consistency_level="Strong",  # https://milvus.io/docs/consistency.md
            limit=limit,
            output_fields=output_fields,
        )
    ]


def ping():
    return "pong"


def count(model_name):
    urls = list_images(model_name)
    result = len(urls)
    while urls:
        urls = list_images(model_name, urls[-1])
        result += len(urls)
    return result


def remove_images(model_name, urls):
    # Milvus only supports deleting entities with clearly specified primary
    # keys, which can be achieved merely with the term expression in. Other
    # operators can be used only in query or scalar filtering in vector search.
    # See Boolean Expression Rules for more information.
    # https://milvus.io/docs/v2.2.x/delete_data.md?shell#Delete-Entities
    collections[model_name].delete(f"url in {format_url_list(urls)}")


commands = dict(
    insert_images=insert_images,
    search=search,
    compare=compare,
    list_images=list_images,
    count=count,
    remove_images=remove_images,
    ping=ping,
)
