import json
import numpy as np
import cv2

from embeddings import load_image_from_url, embeddings
from milvus import collections
from common import INDEX_PARAMS, SEARCH_PARAMS, CARDINALITIES, FEATURE_ID_SEPARATOR


def format_url_list(urls):
    quoted_urls = [f'"{url}"' for url in urls]
    return f'[{",".join(quoted_urls)}]'


def insert_images(
    model_name, urls, metadatas, image_embeddings=None, replace_existing=True
):
    if any("%" in url for url in urls):
        # % in a url prevents prefix queries in milvus 2.3 which are used for local features
        # there are several solution, depending on when and if this causes a problem
        # - wait for milvus to fix the issue
        # - decode/encode urls or replace % with a characters that is not allowed in urls
        # - do not use the url as an id and store the local feature position in an extra column
        raise RuntimeError(
            "Urls containing % are not supported yet. "
            "Please contact the administrator."
        )

    existing_urls = []
    if not replace_existing:
        existing_urls = [
            search_result["url"]
            for search_result in collections[model_name].query(
                f"url in {format_url_list(urls)}",
                consistency_level="Strong",  # https://milvus.io/docs/consistency.md
            )
        ]

    if image_embeddings is None and CARDINALITIES[model_name] > 1:
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
            "metric_type": INDEX_PARAMS[model_name]["metric_type"],
            "params": SEARCH_PARAMS[model_name],
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
    # 0.2s
    results = collections[model_name].query(
        f'url like "{url}#%"',
        consistency_level="Strong",
        output_fields=["embedding"],
    )
    descriptors = []
    positions = []
    for result in results:
        descriptors.append(result["embedding"])
        positions.append(
            [
                float(c)
                for c in result["url"].split("#")[1].split(FEATURE_ID_SEPARATOR)[:2]
            ]
        )

    # 1.4s
    if descriptors == []:
        for feature in embeddings[model_name].get_image_embedding(
            load_image_from_url(url)
        ):
            descriptors.append(feature[0])
            positions.append(
                [float(c) for c in feature[1].split(FEATURE_ID_SEPARATOR)[:2]]
            )

    search_results = collections[model_name].search(
        data=descriptors,
        anns_field="embedding",
        param={
            "metric_type": INDEX_PARAMS[model_name]["metric_type"],
            "params": SEARCH_PARAMS[model_name],
        },
        output_fields=["metadata"],
        limit=limit,
        expr=None,
        consistency_level="Strong",  # https://milvus.io/docs/consistency.md
    )

    matchings = {}
    metadatas = {}
    for i, search_result in enumerate(search_results):
        # Results are sorted by increasing distance, so we only need to take the first
        # appearance of a url for this feature.
        # Using a list would allow ratio tests, à la SIFT
        already_matched = set()
        for hit in search_result:
            url_part, location = hit.id.split("#")
            if url_part not in metadatas:
                metadatas[url_part] = json.loads(hit.entity.get("metadata"))
            if url_part not in already_matched:
                matchings[url_part] = matchings.get(url_part, []) + [
                    [
                        positions[i],
                        [float(c) for c in location.split(FEATURE_ID_SEPARATOR)[:2]],
                    ]
                ]
                already_matched.add(url_part)

    results = []
    for url_part, matching in matchings.items():
        n_matching = len(matching)

        if n_matching < 4:
            continue

        M, mask = cv2.findHomography(
            np.array([m[0] for m in matching]),
            np.array([m[1] for m in matching]),
            cv2.RANSAC,
            5.0,
        )

        n_inliers = np.count_nonzero(mask)

        inliers_ratio = n_inliers / n_matching
        if inliers_ratio < 0.50:
            # print("Too many outliers")
            continue

        if np.linalg.det(M) == 0:
            # print("homography is singular")
            continue

        if abs(1 - np.linalg.cond(M[0:2, 0:2])) > 0.1:
            # print(
            #     "condition number of the top-left 2x2 sub-matrix"
            #     "is too far from a pure rotation"
            # )
            continue

        if np.any(np.abs(M[2, 0:2]) > 0.1):
            # print("perspective parameters are too high")
            continue

        # Other possible checks:
        # whether the translation part is within than a threshold
        # t = np.linalg.norm(M[0:2, 2])
        # whether the scale factor is reasonnable
        # s = M[2, 2]

        results.append(
            {
                "url": url_part,
                "metadata": metadatas[url],
                "similarity": 100 * inliers_ratio,
            }
        )

    # In theory we could have more than limit results,
    # in practice filtering will avoid that
    # results = sorted(results.items(), key=lambda x: x["similarity"], reverse=True)[
    #     :limit
    # ]
    return results


def search_by_url(model_name, url, limit=10):
    if CARDINALITIES[model_name] > 1:
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
    if INDEX_PARAMS[model_name]["metric_type"] == "L2":
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

    # list urls only on local features
    if output_fields is None and CARDINALITIES[model_name] > 1:
        if cursor is not None:
            cursor += "Z"

    results = [
        search_result["url"] if output_fields is None else prepare(search_result)
        for search_result in collections[model_name].query(
            f'url > "{cursor}"',
            consistency_level="Strong",  # https://milvus.io/docs/consistency.md
            limit=limit,
            output_fields=output_fields,
        )
    ]

    # list urls only on local features
    if output_fields is None and CARDINALITIES[model_name] > 1:
        return list(set([result.split("#")[0] for result in results]))

    # full dump or global features
    return results


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
    if CARDINALITIES[model_name] > 1:
        for url in urls:
            full_urls = [
                search_result["url"]
                for search_result in collections[model_name].query(
                    f'url like "{url}#%"',
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
