import json
import numpy as np
import time

from common import MAX_MILVUS_PAGINATION
from features import load_image_from_url, features
from milvus import collections, metrics


def insert_image(model_name, url, metadata):
    times = [time.time()]

    if collections[model_name].query(
        f'url in ["{url}"]',
        consistency_level="Strong",  # https://milvus.io/docs/consistency.md
    ):
        raise ValueError("Url already in collection.")

    times.append(time.time())

    embedding = features[model_name].extract(load_image_from_url(url))

    times.append(time.time())

    collections[model_name].insert([[url], [embedding], [json.dumps(metadata)]])

    times.append(time.time())
    durations = [times[i + 1] - times[i] for i in range(len(times) - 1)]
    print(f"query/extract/insert: {durations}")


def search(model_name, url):
    embedding = features[model_name].extract(load_image_from_url(url))
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
    left = features[model_name].extract(load_image_from_url(url_left))
    right = features[model_name].extract(load_image_from_url(url_right))

    # calc_distance() has been removed from milvus
    # it's a bit overkill anyway if we don't compare with vectors from the db
    if metrics[model_name] == "L2":
        # _squared_ L2, to be consistent with the distances in milvus' search
        return 100 * (1 - np.sum(np.square(np.array(left) - np.array(right))))

    raise RuntimeError(
        "Distance calculation has not been implemented in the API. "
        "Please contact the administrator."
    )


def list_urls(model_name, cursor="", limit=MAX_MILVUS_PAGINATION):
    return [
        search_result["url"]
        for search_result in collections[model_name].query(
            f'url > "{cursor}"',
            consistency_level="Strong",  # https://milvus.io/docs/consistency.md
            limit=limit,
        )
    ]


def count(model_name):
    urls = list_urls(model_name)
    result = len(urls)
    while urls:
        urls = list_urls(model_name, urls[-1])
        result += len(urls)
    return result


def remove_image(model_name, url):
    # Milvus only supports deleting entities with clearly specified primary
    # keys, which can be achieved merely with the term expression in. Other
    # operators can be used only in query or scalar filtering in vector search.
    # See Boolean Expression Rules for more information.
    # https://milvus.io/docs/v2.2.x/delete_data.md?shell#Delete-Entities
    collections[model_name].delete(f'url in ["{url}"]')


commands = dict(
    insert_image=insert_image,
    search=search,
    compare=compare,
    list_urls=list_urls,
    count=count,
    remove_image=remove_image,
)
