# Idios

Idios is a multi-model reverse image search application with an HTTP API that
allows you to store and search for images that are visually similar.

### Features

 - Fully Dockerized for quick deployment and development set up
 - Simple GUI to upload images and test functionality
 - HTTP API for sending requests programmatically
 - Easy scaling by spawning multiple workers
 - Further indexing scaling with Milvus cloud native architecture

### Components

 - Standalone Milvus vector database and dependencies (minio and etcd)
 - Python worker(s) to compute image image embeddings and interact with Milvus
 - FastAPI based, stateless HTTP API. FastAPI leverages python typing to process requests parameters, responses, and documentation.
 - RabbitMQ to manage the job queue between the API and the worker(s)

Users submit image urls to the API via HTTP. The API posts a job in the
RabbitMQ managed message queue and waits for its result. A worker picks the
job, computes the embedding, stores it in the vector database Milvus and
returns a successful response to the API.

Another possible job is to measure the similarity between two images. Images
added to Milvus can also be ranked by similarity with a query image. They can
also be removed from the index, counted, and listed by url.

Embeddings are stored in different Milvus "collections" of embeddings that are
mapped 1:1 to a specific model in Idios. The model has a corresponding
collection name in Milvus, and is specified in the API call. The architecture
is extensible in a way to allows new models to be added in the future. For
Version 1.0 OpenAI's CLIP ViT-b32 model will be used as the primary model.
Further models include CLIP ViT-l14 or ResNet. Images have their embedding
extracted based on a specific model, and then are added to a collection for
storage and search.

Each image has an associated ID (`VARCHAR`) which is the URL of the image, and
serves as the primary key in the Milvus collection. Optionally, additional
metadata can be associated with the image as an arbitrary JSON object.

### HTTP API Reference

Idios can be controlled using a simple HTTP API that listens on port 4213.
Requests parameters and responses are formatted in JSON.

The API has an interactive documentation at the /doc path of a live setup. This
[interactive documentation](https://qbonnard.github.io/idios/) is also hosted
on the repository. It is regenerated on each push to master by GitHub's CI.

Each endpoint has a "Try it out" button which reaveals an interface to build a
curl request.

## Development

### Quickstart

Install the docker engine (or docker desktop) and the compose plugin:
https://docs.docker.com/compose/install/

Run the development environment:
```sh
docker compose -p idios -f docker/docker-compose.yml up --build --remove-orphans -d
```

This command can be invoked by calling `make up` at the repository root,
assuming make is available on you system. [`Makefile`](./Makefile) contains many
useful commands for development, mostly using make as a command launcher rather
than a build system. The commands are documented in the Makefile itself.

### Architecture

The [development compose file](./docker/docker-compose.yml) contains the core
components of idios: a standalone milvus deployment with the required etcd and
minio services, the api service, the worker service, and the rabbitmq service.
It also includes attu, a web UI for milvus.

The api and worker service share a common image, including the
[dependencies](./api/requirements.txt) of
[both](./api/requirements-worker.txt), as well as extra [development
requirements](./api/requirements-dev.txt) (mostly testing instrumentation).

### Repository structure

[`docker`](./docker) and [`doc`](./doc) are folders that respectively contain
docker (compose) related configuration files and a static export of the
interactive API reference.

The `[api](./api)` folder contains two groups of source files:

- the api code :
  - [main.py](./api/main.py) specifies the HTTP API with FastAPI
  - [rpc_client.py](./api/rpc_client.py) handles remote procedure call generation
    and response parsing
  - [openapi.py](./api/openapi.py) is a script to export the static API reference

- the worker code :
  - [features.py](./api/features.py) leverages models to generate embeddings
    from images at given urls
  - [milvus.py](./api/milvus.py) wraps calls to the milvus API
  - [commands.py](./api/commands.py) integrates the  together
  - [worker.py](./api/worker.py) wraps the commands

- [common.py](./api/common.api) includes constants shared between the two groups

## Deployment

### Set up

### Scaling

### Further scaling of milvus

https://milvus.io/tools/sizing

## References

docker
docker compose
FastAPI
pymilvus
rabbitmq/RPC sample
vit sample
