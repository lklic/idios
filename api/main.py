from fastapi import FastAPI, status, HTTPException, Query
from pydantic import BaseModel, HttpUrl, confloat, conint
from enum import Enum
from typing import Literal, Optional

import json

from common import embedding_dimensions, MAX_METADATA_LENGTH, MAX_MILVUS_PAGINATION
from rpc_client import RpcClient


app = FastAPI(
    title="Idios",
    description="""
Idios is a reverse image search application with an HTTP API that allows you to search for images that are visually similar.

It supports several embedding models. Images can be added to each of these models separately. To do so, their embedding will be combuted and indexed in a milvus collection. This will later allow to search, or compare other images in this collection. Images of each model can also be deleted, listed and counted.
""".strip(),
    version="0.1.0",
    # terms_of_service, contact, licence_info:
    # https://fastapi.tiangolo.com/tutorial/metadata/#metadata-for-api
    openapi_tags=[
        {
            "name": "model",
            "description": """
Endpoints to manipulate the index of images for each embedding model
            """.strip(),
        },
        {"name": "misc", "description": "Extra"},
    ],
)


ModelName = Enum("ModelName", {k: k for k in embedding_dimensions.keys()})
ModelName.__doc__ = (
    "The enumeration of supported models to extract embeddings from images"
)


class ImageUrl(HttpUrl):
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            title="URL of the image",
            description="""
The url serves as the primary key in the Milvus collection. For consistancy, image URLs should not have escape characters. Optionally, additional metadata can be associated with the image as an arbitrary JSON object.

Assumptions
- Url follows pydantics' [HttpUrl](https://docs.pydantic.dev/usage/types/#urls): scheme http or https, TLD required, host required, max length 2083.
- Images should have persitant URLs that will not change over time.
- Images are accessible from the server where Idios is hosted and do not require authentication.
- Only JPEG images are supported
- Images must have their dimensions above 150 x 150 pixels, otherwise the API will return an error `IMAGE_SIZE_TOO_SMALL`
- If one of the image dimension exceeds 1000 pixels, Idios will resize the image so that the maximum dimension is set to 1000 pixels and the original aspect ratio is kept.
- Some image links may be permalinks from library or museum image collections or be hosted on IIIF servers that only accept certain request headers, additionally, the URL may return a 303 redirect to point you to the actual image. The API should be able to handle the redirect silently and still use the original URL as the primary key.
""".strip(),
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
""".strip(),
        )


class ImageMetadata(dict):
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(
            title="Optional, arbitrary JSON object to attach to the image",
            description="""
The metadata is parsed as JSON to ensure its validity and regenerated to possibly minimise it (e.g. removing unnecessary white spaces). The regenerated JSON should at most 65535 characters long.
""".strip(),
            example={"tags": ["text"], "language": "ja"},
        )


class SingleImage(BaseModel):
    url: ImageUrl


class ImageAndMetada(SingleImage):
    metadata: ImageMetadata | None


class SearchParameters(BaseModel):
    text: Optional[str]
    url: Optional[ImageUrl]
    limit: conint(ge=1) = 10


class ImagePair(SingleImage):
    other: ImageUrl


class SimilarityScore(BaseModel):
    "Scores range from 0 to 100, with 100 being a perfect match."

    __root__: confloat(ge=0, le=100)


class Pagination(BaseModel):
    cursor: Cursor | None
    limit: conint(ge=1, le=MAX_MILVUS_PAGINATION) | None


class SearchResult(ImageAndMetada):
    similarity: SimilarityScore


class SearchResults(BaseModel):
    __root__: list[SearchResult]


class DatabaseEntry(ImageAndMetada):
    embedding: list[float]


def try_rpc(command, args):
    # Create a new client each time, i.e. reconnect and set up the channel
    # This costs < 10ms, but fixes disconnections and parallelisation issues
    rpc_client = RpcClient()
    try:
        return rpc_client(command, args)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[{"msg": str(e), "type": "parameter_error"}],
        )
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=[{"msg": str(e), "type": "server_error"}],
        )
    finally:
        rpc_client.connection.close()


@app.post(
    "/models/{model_name}/add",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["model"],
    summary="""
Adds a new image embedding to the index.
Adding an existing url will replace the metadata with the provided one.
    """.strip(),
)
async def upsert_image(model_name: ModelName, image: ImageAndMetada):
    try_rpc(
        "insert_images",
        [model_name.value, [image.url], [check_json_string_length(image.metadata)]],
    )


@app.post(
    "/models/{model_name}/search_add",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["model"],
    summary="""
Adds a new image embedding to the index.
Adding an existing url will trigger a 409 HTTP error.
    """.strip(),
    responses={409: {"description": "Image already inserted"}},
)
async def insert_image(model_name: ModelName, image: ImageAndMetada):
    result = try_rpc(
        "insert_images",
        [
            model_name.value,
            [image.url],
            [check_json_string_length(image.metadata)],
            None,
            False,
        ],
    )

    if (
        "found" in result
        and isinstance(result["found"], list)
        and len(result["found"]) > 0
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Image already inserted"
        )


def check_json_string_length(metadata):
    metadata_string = json.dumps(metadata)
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
    return json.loads(metadata_string)


@app.post(
    "/models/{model_name}/restore",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["model"],
    summary="""
Adds multiple image embeddings to the index.
Existing urls will have their metadata replaced with the provided one.
    """.strip(),
)
async def restore(model_name: ModelName, images: list[DatabaseEntry]):
    # The type declaration generates the openapi documentation as expected
    # but results in this rather ugly line. There may be a better use of pydantic
    # images = images.__root__
    try_rpc(
        "insert_images",
        [
            model_name.value,
            [image.url for image in images],
            [check_json_string_length(image.metadata) for image in images],
            [image.embedding for image in images],
        ],
    )


class BulkInsertResult(BaseModel):
    added: list[ImageUrl]
    found: list[ImageUrl]
    failed: list[dict]


@app.post(
    "/models/{model_name}/add_bulk",
    status_code=status.HTTP_200_OK,
    tags=["model"],
    summary="""
Adds multiple images to the index at once.
Existing urls will have their metadata replaced with the provided ones.
Returns lists of successfully added URLs, existing URLs, and failed URLs with error messages.
    """.strip(),
    response_model=BulkInsertResult
)
async def add_bulk(model_name: ModelName, images: list[ImageAndMetada]):
    result = try_rpc(
        "insert_images",
        [
            model_name.value,
            [image.url for image in images],
            [check_json_string_length(image.metadata) for image in images],
            None,  # No pre-computed embeddings
            True,   # Replace existing
            False,  # Don't fail on error, continue processing
        ],
    )
    return result


@app.post(
    "/models/{model_name}/search",
    tags=["model"],
    summary="Search images with similar embeddings in the index",
    response_model=SearchResults,
)
async def search(model_name: ModelName, params: SearchParameters):
    if params.url:
        return try_rpc("search_by_url", [model_name.value, params.url, params.limit])
    elif params.text:
        return try_rpc("search_by_text", [model_name.value, params.text, params.limit])
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Either 'text' or 'url' must be provided.",
        )


@app.post(
    "/models/{model_name}/compare",
    tags=["model"],
    summary="Compute the similarity score between two images",
    response_model=SimilarityScore,
)
async def compare(model_name: ModelName, images: ImagePair):
    return try_rpc("compare", [model_name.value, images.url, images.other])


@app.post(
    "/models/{model_name}/urls",
    tags=["model"],
    summary="""
List the urls of all images.
Use the pagination cursor to make sure that the enumeration reached the end.
""".strip(),
    response_model=list[ImageUrl],
)
async def list_images(
    model_name: ModelName, pagination: Pagination = Pagination(cursor=None, limit=None)
):
    return try_rpc(
        "list_images", [model_name.value, pagination.cursor, pagination.limit]
    )


@app.post(
    "/models/{model_name}/dump",
    tags=["model"],
    summary="""
Dump the urls, the embedding and the metadata of all images.
Use the pagination cursor to make sure that the enumeration reached the end.
""".strip(),
    response_model=list[DatabaseEntry],
)
async def dump(
    model_name: ModelName, pagination: Pagination = Pagination(cursor=None, limit=None)
):
    return try_rpc(
        "list_images",
        [
            model_name.value,
            pagination.cursor,
            pagination.limit,
            ["url", "embedding", "metadata"],
        ],
    )


@app.get(
    "/models/{model_name}/count",
    tags=["model"],
    summary="Count the number of images in in a given index",
)
async def count(model_name: ModelName):
    return try_rpc("count", [model_name.value])


@app.post(
    "/models/{model_name}/remove",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["model"],
    summary="Remove an image embedding from the index.",
)
async def remove_image(model_name: ModelName, image: SingleImage):
    return try_rpc("remove_images", [model_name.value, [image.url]])


@app.get(
    "/ping",
    tags=["misc"],
    response_model=Literal["pong"],
    summary="Check for the health of the server.",
)
async def ping(
    rpc: bool = Query(
        False,
        description="Also send an RPC ping to check a round-trip with a worker via the job queue",
    )
):
    if rpc:
        return try_rpc("ping", [])
    return "pong"
