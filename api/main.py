from fastapi import FastAPI, status, Response, HTTPException
from pydantic import BaseModel, Field, HttpUrl
from enum import Enum
from typing import Literal

import json
import numpy as np
from pymilvus import utility

from milvus import get_collection
from features_rpc import FeaturesRpc

collections = {
    name: get_collection(name, dim)
    for name, dim in {
        "vit_b32": 512,
        # "vit_l14" : 4096,
    }.items()
}

for collection in collections.values():
    collection.load()

metrics = {
    name: collection.index()._index_params["metric_type"]
    for name, collection in collections.items()
}


features_rpc = FeaturesRpc()


def request_features(model_name, url, error_location=["body", "url"]):
    try:
        return features_rpc(model_name, url)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[
                {
                    "loc": error_location,
                    "msg": str(e),
                    "type": "parameter_error",
                }
            ],
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_501_INTERNAL_SERVER_ERROR,
            detail=[
                {
                    "loc": error_location,
                    "msg": str(e),
                    "type": "server_error",
                }
            ],
        )


app = FastAPI(
    title="Idios",
    description="Idios is a reverse image search application with an HTTP API that allows you to search for images that are visually similar.",
    version="0.0.1",
)


ModelName = Enum("ModelName", {k: k for k in collections.keys()})
ModelName.__doc__ = (
    "The enumeration of supported models to extract features/embeddings from images"
)


class ImageUrl(HttpUrl):
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            title="URL of the image",
            description="""
The url serves as the primary key in the Milvus collection. For consistancy, image URLs should not have escape characters. Optionally, additional metadata can be associated with the image as an arbitrary JSON object.

Assumptions
- Images should have persitant URLs that will not change over time.
- Images are accessible from the server where Idios is hosted and do not require authentication.
- Only JPEG images are supported
- Images must have their dimensions above 150 x 150 pixels, otherwise the API will return an error `IMAGE_SIZE_TOO_SMALL`
- If one of the image dimension exceeds 1000 pixels, Idios will resize the image so that the maximum dimension is set to 1000 pixels and the original aspect ratio is kept.
- Some image links may be permalinks from library or museum image collections or be hosted on IIIF servers that only accept certain request headers, additionally, the URL may return a 303 redirect to point you to the actual image. The API should be able to handle the redirect silently and still use the original URL as the primary key.
""",
            example="https://iiif.itatti.harvard.edu/iiif/2/yashiro!letters-jp!letter_001.pdf/full/full/0/default.jpg",
        )


class ImageMetadata(dict):
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            title="Arbitrary metadata to attach to the image",
            example={"tags": ["text"], "language": "ja"},
        )


class ImageAndMetada(BaseModel):
    url: ImageUrl
    metadata: ImageMetadata | None


class SingleImage(BaseModel):
    url: ImageUrl


class ImagePair(BaseModel):
    url_left: ImageUrl
    url_right: ImageUrl


class Distance(BaseModel):
    __root__: float


class SearchResult(BaseModel):
    url: ImageUrl
    metadata: ImageMetadata | None
    distance: Distance


class SearchResults(BaseModel):
    __root__: list[SearchResult]


@app.post(
    "/models/{model_name}/add",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["model"],
    summary="Adds an image embedding to the index.",
)
async def insert_image(model_name: ModelName, image: ImageAndMetada):
    embedding = request_features(model_name.value, image.url)
    collections[model_name.value].insert(
        [[image.url], [embedding], [json.dumps(image.metadata)]]
    )


@app.post(
    "/models/{model_name}/search",
    tags=["model"],
    summary="Search images with similar embeddings in the index",
    response_model=SearchResults,
)
async def search(model_name: ModelName, image: SingleImage):
    embedding = request_features(model_name.value, image.url)
    search_results = collections[model_name.value].search(
        data=[embedding],
        anns_field="embedding",
        param={
            "metric_type": metrics[model_name.value],
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
            "distance": hit.distance,
        }
        for hit in search_results[0]
    ]


@app.post(
    "/models/{model_name}/compare",
    tags=["model"],
    summary="Compute the distance between two images",
    response_model=Distance,
)
async def compare(model_name: ModelName, images: ImagePair):
    # alternatively, we could first try to fetch the embeddings from milvus in
    # case their computation is significantly more expensive than a query
    embedding_left, embedding_right = [
        request_features(model_name.value, images.url_left, ["body", "url_left"]),
        request_features(model_name.value, images.url_right, ["body", "url_right"]),
    ]

    # calc_distance() has been removed from milvus
    # it's a bit overkill anyway if we don't compare with vectors from the db
    if metrics[model_name.value] == "L2":
        # squared L2, to be consistent with the distances in milvus' search
        return np.sum(np.square(np.array(embedding_left) - np.array(embedding_right)))

    raise HTTPException(
        status_code=status.HTTP_501_INTERNAL_SERVER_ERROR,
        detail=[
            {
                "loc": [metrics[model_name.value]],
                "msg": (
                    "Distance calculation has not been implemented in the API. "
                    "Please contact the administrator."
                ),
                "type": "not_implemented.distance",
            }
        ],
    )


@app.get(
    "/models/{model_name}/urls",
    tags=["model"],
    summary="List all images",
    response_model=list[ImageUrl],
)
async def list_urls(model_name: ModelName):
    return [
        result["url"]
        for result in collections[model_name.value].query(
            'url > ""',
            consistency_level="Strong",  # https://milvus.io/docs/consistency.md
        )
    ]


# @app.get(
#    "/models/{model_name}/count",
#    tags=["model"],
#    summary="Number of images",
# )
# async def count(model_name: ModelName):
#    return collections[model_name.value].num_entities


@app.post(
    "/models/{model_name}/remove",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["model"],
    summary="Remove an image embedding from the index.",
)
async def remove_image(model_name: ModelName, image: SingleImage):
    # Milvus only supports deleting entities with clearly specified primary
    # keys, which can be achieved merely with the term expression in. Other
    # operators can be used only in query or scalar filtering in vector search.
    # See Boolean Expression Rules for more information.
    # https://milvus.io/docs/v2.2.x/delete_data.md?shell#Delete-Entities
    collections[model_name.value].delete(f'url in ["{image.url}"]')


@app.get("/ping", tags=["misc"], response_model=Literal["pong"])
async def ping():
    return "pong"
