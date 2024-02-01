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

    if image_embeddings is None and hasattr(embeddings[model_name], "cardinality"):
        local_urls = []
        local_embeddings = []
        local_metadatas = []
        for url, metadata in zip(urls, metadatas):
            if url not in existing_urls:
                for descriptor, location in embeddings[model_name].get_image_embedding(
                    load_image_from_url(url)
                ):
                    local_urls.append(f"{url}#{location}")
                    local_embeddings.append(descriptor)
                    local_metadatas.append(metadata)  # FIXME? wasteful
        urls = local_urls
        image_embeddings = local_embeddings
        metadatas = local_metadatas

    new_urls = [url for url in urls if url not in existing_urls]
    image_embeddings = (
        [
            embeddings[model_name].get_image_embedding(load_image_from_url(url))
            for url in new_urls
        ]
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


def similarity_score(distance):
    # 2 is the maximum distance between normalised vectors
    return 100 * (1 - distance / 2)


def search_by_embeddings(model_name, embeddings, limit=10):
    search_results = collections[model_name].search(
        data=embeddings,
        anns_field="embedding",
        param={
            "metric_type": metrics[model_name],
            "params": {
                "nprobe": 64
            },  # https://milvus.io/docs/v1.1.1/performance_faq.md
        },
        output_fields=["metadata"],
        limit=limit,
        expr=None,
        consistency_level="Strong",  # https://milvus.io/docs/consistency.md
    )
    return [
        {
            "url": hit.id,
            "metadata": json.loads(hit.entity.get("metadata")),
            "similarity": similarity_score(hit.distance),
        }
        for hit in search_results[0]
    ]


def search_by_local_features(model_name, url, limit):
    search_results = collections[model_name].search(
        data=[
            feature[0]
            for feature in embeddings[model_name].get_image_embedding(
                load_image_from_url(url)
            )
        ],
        anns_field="embedding",
        param={
            "metric_type": metrics[model_name],
            "params": {
                "params": {"ef": limit},  # TODO ?
            },  # https://milvus.io/docs/v1.1.1/performance_faq.md
        },
        output_fields=["metadata"],
        limit=limit,  # TODO ?
        expr=None,
        consistency_level="Strong",  # https://milvus.io/docs/consistency.md
    )

    scores = {}
    metadatas = {}
    for search_result in search_results:
        for hit in search_result:
            url = hit.id.split("#")[0]
            scores[url] = scores.get(url, 0) + 1
            if url not in metadatas:
                metadatas[url] = json.loads(hit.entity.get("metadata"))
    return [
        {
            "url": url,
            "metadata": metadatas[url],
            "similarity": min(score * 100 / embeddings[model_name].cardinality, 100),
        }
        for url, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]
        if score > 0  # TODO
    ]


def search_by_url(model_name, url, limit=10):
    if hasattr(embeddings[model_name], "cardinality"):
        return search_by_local_features(model_name, url, limit)
    embedding = embeddings[model_name].get_image_embedding(load_image_from_url(url))
    return search_by_embeddings(model_name, [embedding], limit)


def search_by_text(model_name, text, limit=10):
    embedding = embeddings[model_name].get_text_embedding(text)
    return search_by_embeddings(model_name, [embedding], limit)


def compare(model_name, url_left, url_right):
    # alternatively, we could first try to fetch the embeddings from milvus in
    # case their computation is significantly more expensive than a query
    left = embeddings[model_name].get_image_embedding(load_image_from_url(url_left))
    right = embeddings[model_name].get_image_embedding(load_image_from_url(url_right))

    # calc_distance() has been removed from milvus
    # it's a bit overkill anyway if we don't compare with vectors from the db
    if metrics[model_name] == "L2":
        # _squared_ L2, to be consistent with the distances in milvus' search
        return similarity_score(np.sum(np.square(np.array(left) - np.array(right))))

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
    if hasattr(embeddings[model_name], "cardinality"):
        for url in urls:
            full_urls = [
                search_result["url"]
                for search_result in collections[model_name].query(
                    f'url like "{url}%"',
                    consistency_level="Strong",  # https://milvus.io/docs/consistency.md
                )
            ]
            collections[model_name].delete(f"url in {format_url_list(full_urls)}")
    else:
        collections[model_name].delete(f"url in {format_url_list(urls)}")


commands = dict(
    insert_images=insert_images,
    search_by_url=search_by_url,
    search_by_text=search_by_text,
    compare=compare,
    list_images=list_images,
    count=count,
    remove_images=remove_images,
    ping=ping,
)
