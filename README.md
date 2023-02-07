# Idios

## Idios is a reverse image search application with an HTTP API that allows you to search for images that are visually similar.

## Features
 - Fully Dockerized for quick deployment
 - Simple GUI to upload images and test
 - HTTP API for sending requests programmatically


Embeddings are stored in the vector database Milvus, with different Milvus "collections" of embeddings that are mapped to a specific model in Idios. The architecture is extensible in a way to allows new models to be added in the future. For Version 1.0 OpenAI's CLIP model will be used as the primary model. Images have their embedding extracted based on a specific model, and then are added to a collection for storage and search. Each model has a corresponding Milvus collection to which it is mapped. The "index" corresponds to a mapping between a specific model (for extracting embeddings) and a Milvus collection


### References

- Milvus Image Similarity Search example: https://milvus.io/docs/image_similarity_search.md

 Note: as of version 2.2 there is no need to use a MySQL database to search for similar images because the field schema now supoorts the VARCHAR data type as primary key. Please see the Python API reference: https://milvus.io/api-reference/pymilvus/v2.2.0/Schema/FieldSchema.md. 

- Image Deduplication: https://milvus.io/docs/image_deduplication_system.md This can be used for near-exact image matching 

## Docker configuration

- For extracting the embedding from large amounts of images, docker swarm can scale the number of containers that extract the embeddings.

## Assumptions

- Images should have persitant URLs that will not change over time.
- Images are accessible from the server where Idios is hosted and do not require authentication.
- Only JPEG images are supported
- Images must have their dimensions above 150 x 150 pixels, otherwise the API will return an error `IMAGE_SIZE_TOO_SMALL`
- If one of the image dimension exceeds 1000 pixels, Idios will resize the image so that the maximum dimension is set to 1000 pixels and the original aspect ratio is kept.
- Some image links may be permalinks from library or image collections 


# HTTP API

Idios can be controlled using a simple HTTP API that listens on port 4213, responses are formatted in JSON.

Each image has an associated ID (`VARCHAR`) which is the URL of the image, and serves as the primary key in Milvus collection. For consistancy, image URLs should not have escape characters.


Each image has an associated ID which is the url of the image (string) that serves as the primary key in a Mulvus collection. Optionally, additional metadata can be associated with the image as an arbitrary JSON object.

# API Reference

Idios has a simple HTTP API. All request parameters are specified via `application/x-www-form-urlencoded`.

* [POST `/add`](#post-add)
* [DELETE `/delete`](#delete-delete)
* [POST `/search`](#post-search)
* [POST `/compare`](#post-compare)
* [GET `/count`](#get-count)
* [GET `/list`](#get-list)
* [GET `/ping`](#get-ping)

---

### POST `/add`

Adds an image embedding to the index.

#### Parameters

* **url** *(required)*

  The image to add to the database. It may be provided as a URL via `url` .

* **metadata** *(default: None)*

  An arbitrary JSON object featuring meta data to attach to the image.


* Answer type: "IMAGE_ADDED"
* Possible error types: "IMAGE_NOT_DECODED", "IMAGE_SIZE_TOO_SMALL", "IMAGE_DOWNLOADER_HTTP_ERROR" with the HTTP status code in the "image_downloader_http_response_code" field.

#### Example Response

```json
{
  "status": "ok",
  "error": [],
  "method": "add",
  "result": []
}
```

---

### DELETE `/delete`

Deletes an image from the index.

#### Parameters

* **url** *(required)*

  The URL of the image signature in the index.

#### Example Response

```json
{
  "status": "ok",
  "error": [],
  "method": "delete",
  "result": []
}
```

---

### POST `/search`

Searches for a similar image in the database. Scores range from 0 to 100, with 100 being a perfect match.

#### Parameters

* **url** *(required)*

  The image URL to search for in the index. It may be provided as a URL via `url`.


#### Example Response

```json
{
  "status": "ok",
  "error": [],
  "method": "search",
  "result": [
    {
      "score": 99.0,
      "ulr": "http://example.com/image.jpg"
    }
  ]
}
```

---

### POST `/compare`

Compares two images, returning a score for their similarity. Scores range from 0 to 100, with 100 being a perfect match.

#### Parameters

* **url1** , **url2** *(required)*

  The images to compare. They may be provided as a URL via `url1`/`url2` .

#### Example Response

```json
{
  "status": "ok",
  "error": [],
  "method": "compare",
  "result": [
    {
      "score": 99.0
    }
  ]
}
```

---

### GET `/count`

Count the number of images in in a given index.

#### Example Response

```json
{
  "status": "ok",
  "error": [],
  "method": "list",
  "result": [420]
}
```

---

### GET `/list`

Lists the URLs for the image signatures in the database.

#### Parameters

* **offset** *(default: 0)*

  The location in the database to begin listing image paths.

* **limit** *(default: 20)*

  The number of image paths to retrieve.

#### Example Response

```json
{
  "status": "ok",
  "error": [],
  "method": "list",
  "result": [
    "http://img.youtube.com/vi/iqPqylKy-bY/0.jpg",
    "https://i.ytimg.com/vi/zbjIwBggt2k/hqdefault.jpg",
    "https://s-media-cache-ak0.pinimg.com/736x/3d/67/6d/3d676d3f7f3031c9fd91c10b17d56afe.jpg"
  ]
}
```

---

### GET `/ping`

Check for the health of the server.

#### Example Response

```json
{
  "status": "ok",
  "error": [],
  "method": "ping",
  "result": []
}
```






Idios can have any number of indexes, each one is stored in a Mulvus “collection” and can be specified specified through the http API as such:

“http://localhost:4212/index/“

Where “index” is the name of miles collection.


Adding an image to the index
This call allows to add the embedding of an image in the index to make it available for searching. You need to provide the compressed binary data of the image and an id to identify it.
* Path: /index/image
* HTTP method: PUT
* Data: JSON containing the URL of the image in the "url" field.
* Answer type: "IMAGE_ADDED"
* Possible error types: "IMAGE_NOT_DECODED", "IMAGE_SIZE_TOO_SMALL", "IMAGE_DOWNLOADER_HTTP_ERROR" with the HTTP status code in the "image_downloader_http_response_code" field.

Removing an image from the index
This call removes the signature of an image in the index thanks to its id. Be careful to not call often this method if your index is big because it is currently very slow.
* Path: /index/image
* HTTP method: DELETE
* Answer type: "IMAGE_REMOVED"

Search request
This call performs a search in the index thanks to a request image. It returns the id of the matched images from the most to the least relevant ones.

* Path: /index/searcher
* HTTP method: POST
* Data: a JSON containing the URL of the image in the "url" field.
* Answer: "SEARCH_RESULTS" as type field and a list of the the matched image ids from the most to the least relevant one in the "image_ids" field

Ping
This call sends a simple PING command that answers with a PONG.
* Path: /
* HTTP method: POST
* Data: a json with a "type" field of value "PONG"
* Answer type: "PONG"
* Example:Command line:curl -X POST -d '{"type":"PING"}' http://localhost:4213/Answer:{ "type" : "PONG"}


