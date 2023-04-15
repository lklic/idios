from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel, HttpUrl, confloat
from enum import Enum
from typing import Literal

import json

from common import feature_dimensions, MAX_METADATA_LENGTH
from rpc_client import RpcClient


rpc = RpcClient()


def try_rpc(command, args):
    try:
        return rpc(command, args)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[
                {
                    "msg": str(e),
                    "type": "parameter_error",
                }
            ],
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=[
                {
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


ModelName = Enum("ModelName", {k: k for k in feature_dimensions.keys()})
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


class Cursor(str):
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            title="Enumeration cursor",
            description="""
The cursor parameter is used for pagination purposes to allow clients to retrieve a large set of results in smaller, more manageable chunks. The cursor is a string that represents the position of the last item returned in the previous API call.

When making a request to retrieve a set of images, the client can include the cursor parameter in the request to specify where in the set of results they want to start from. The API will return a subset of the results starting after the url specified by the cursor.
""",
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


class SimilarityScore(BaseModel):
    __root__: confloat(ge=0, le=100)


class Pagination(BaseModel):
    cursor: Cursor


class SearchResult(BaseModel):
    url: ImageUrl
    metadata: ImageMetadata | None
    similarity_score: SimilarityScore


class SearchResults(BaseModel):
    __root__: list[SearchResult]


@app.post(
    "/models/{model_name}/add",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["model"],
    summary="Adds an image embedding to the index.",
)
async def insert_image(model_name: ModelName, image: ImageAndMetada):
    metadata_string = json.dumps(image.metadata)
    if len(metadata_string) > MAX_METADATA_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[
                {
                    "loc": ["body", "metadata"],
                    "msg": f"metadata json too long ({len(metadata_string)} > {MAX_METADATA_LENGTH})",
                    "type": "value_error.metadata_json_too_long",
                }
            ],
        )
    try_rpc("insert_image", [model_name.value, image.url, metadata_string])


@app.post(
    "/models/{model_name}/search",
    tags=["model"],
    summary="Search images with similar embeddings in the index",
    response_model=SearchResults,
)
async def search(model_name: ModelName, image: SingleImage):
    return try_rpc("search", [model_name.value, image.url])


@app.post(
    "/models/{model_name}/compare",
    tags=["model"],
    summary="Compute the similarity score between two images",
    response_model=SimilarityScore,
)
async def compare(model_name: ModelName, images: ImagePair):
    # alternatively, we could first try to fetch the embeddings from milvus in
    # case their computation is significantly more expensive than a query
    return try_rpc("compare", [model_name.value, images.url_left, images.url_right])


@app.post(
    "/models/{model_name}/urls",
    tags=["model"],
    summary="List all images",
    response_model=list[ImageUrl],
)
async def list_urls(model_name: ModelName, pagination: Pagination | None = None):
    return try_rpc(
        "list_urls", [model_name.value, pagination.cursor if pagination else None]
    )


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
    return try_rpc("remove_image", [model_name.value, image.url])


@app.get("/ping", tags=["misc"], response_model=Literal["pong"])
async def ping():
    return "pong"
