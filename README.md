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
 - FastAPI based, stateless HTTP API. FastAPI leverages python typing to
   process requests parameters, responses, and documentation.
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

The API has an interactive documentation at the /docs path of a live setup. This
[interactive documentation](https://lklic.github.io/idios/) is also hosted
on the repository. It is regenerated on each push to master by [GitHub's
CI](./.github/workflows). Each endpoint has a "Try it out" button which reveals
an interface to build a curl request.

## Development

### Quickstart

Install the docker engine (or docker desktop) and the compose plugin:
https://docs.docker.com/compose/install/

Run the development environment:
```sh
docker compose -p idios -f docker/docker-compose.dev.yml up --build --remove-orphans -d
```

This command can be invoked by calling `make up` at the repository root,
assuming make is available on you system. [`Makefile`](./Makefile) contains many
useful commands for development, mostly using make as a command launcher rather
than a build system. The commands are documented in the Makefile itself.

### Architecture

The [development compose file](./docker/docker-compose.dev.yml) contains the core
components of idios: a standalone Milvus deployment with the required etcd and
minio services, the api service, the worker service, and the rabbitmq service.
rabbitmq is configured to serve its management web UI. The development compose
file also includes attu, a web UI for Milvus.

The api and worker service share a common image, including the
[dependencies](./api/requirements.txt) of
[both](./api/requirements-worker.txt), as well as extra [development
requirements](./api/requirements-dev.txt) (mostly testing instrumentation).

### Repository structure

[`docker`](./docker) and [`doc`](./doc) are folders that respectively contain
docker (compose) related configuration files and a static export of the
interactive API reference.

The [`api`](./api) folder contains two groups of source files:

- the api code :
  - [main.py](./api/main.py) specifies the HTTP API with FastAPI
  - [rpc_client.py](./api/rpc_client.py) handles remote procedure call generation
    and response parsing
  - [openapi.py](./api/openapi.py) is a script to export the static API reference

- the worker code :
  - [embeddings.py](./api/embeddings.py) leverages models to generate embeddings
    from images at given urls
  - [milvus.py](./api/milvus.py) wraps calls to the Milvus API
  - [commands.py](./api/commands.py) integrates the  together
  - [worker.py](./api/worker.py) wraps the commands

- [common.py](./api/common.api) includes constants shared between the two groups

## Deployment

### Scaling

There are many ways to deploy Idios, depending on how we plan on scaling. Each
core component (Milvus, RabbitMQ, the worker and the api) are designed to be
scaled horizontally. See the relevant documentation for
[Milvus](https://milvus.io/docs/scaleout.md) (and [siziing
estimations](https://milvus.io/tools/sizing)) and
[RabbitMQ](https://www.rabbitmq.com/clustering.html). The workers are
stateless; replicas can be spawned by giving them access to Milvus and the job
queue in RabbitMQ. The API is stateless too; replicas only need access to the
job queue and can be spawned behind a load balancer.

Additional components can be added, like a load balancer, TLS-terminating
reverse proxy like traefik, administration interfaces, web server for static
files...

Most likely, the limiting factors will be Milvus and the embedding computation
in the worker.

### Deployment design

In the deployment described here, we make the following decisions :
- Deploy Milvus standalone.
- Make RabbitMQ and Milvus publicly accessible (but password protected), so
  that we can easily add workers. The alternative would be to setup a private
  network, e.g. managed by docker swarm, but that usually assume that all
  machines are on the same local network (or at least the same data center).
- We will setup the api and RabbitMQ on the same node (machine) as Milvus,
  because their resource usage will be negligible, and the three of them need
  to be publicly accessible.
- We will add a traefik instance to manage TLS certificates and reverse
  proxying domains to the three components above.
- The worker node will be deployed separately, so that we can deploy it on the
  frontend node, one or more other nodes, or both.
- Use a domain name for each of the three publicly available service. One
  domain would be enough by configuring path prefixes in traefik. None at all
  (i.e. a simple IP adress and ports) would work too, at the expense of TLS
  (i.e. no https).

### Prerequisites

To setup the described deployment, you will need:
- A machine with [docker compose set up](https://docs.docker.com/compose/install)
  to serve as frontend node,
- 3 domain names pointing to the frontend node. In our example we take:
  api.idios.org, queue.idios.org, milvus.idios.org
- Ports 80, 443 (for traefik), 5672 (for RabbitMQ), 19530 (for Milvus) open:
- Zero or more machines with docker compose
- A passwordless ssh access to the nodes. They are not required but make life a
  lot easier. There are many tutorial, often by the hosting providers. Make
  sure to have an ssh-agent running so that docker build works without problem.

### Docker contexts

[Docker contexts](https://docs.docker.com/engine/context/working-with-contexts/)
simplify docker-based orchestration by handling remote docker instances with
the same commands as if they were local. For example:
```
docker --context idios-worker0 ps -a
```
will show all containers (`ps -a`) as if we were running the command on the
docker daemon referred to as idios-worker0.

There is already a `default` context referencing the local machine. An
`idios-frontend` can be created with:
```
docker context create idios-frontend --docker 'host=ssh://username@host.or.ip'
```
and inspected with:
```
docker context ls
```

A docker engine can be referred by multiple contexts, so we can use the
following command to make the location of the first worker transparent:
```
# to run the first worker on the same node as the frontend:
docker context create idios-worker0 --docker 'host=ssh://username@host.or.ip'
# to run the first worker on the local machine:
docker context create idios-worker0 --docker 'host=unix:///var/run/docker.sock'
```

### Set up

Edit the lines marked with a `#TOEDIT` comment in the
[docker-compose.frontend.yml](./docker/docker-compose.frontend.yml),
[docker-compose.milvus.yml](./docker/docker-compose.milvus.yml) and
[docker-compose.worker.yml](./docker/docker-compose.worker.yml). They are the
uri and credentials needed to connect to the Milvus and RabbitMQ services.

Then start the components one by one:
```
docker --context idios-frontend compose -p idios-frontend -f docker/docker-compose.frontend.yml up -d --build --remove-orphans
docker --context idios-milvus compose -p idios-milvus -f docker/docker-compose.milvus.yml up -d --build --remove-orphans
docker --context idios-worker compose -p idios-worker -f docker/docker-compose.worker.yml up -d --build --remove-orphans
```

The `Makefile` includes the corresponding shortcuts:
```
make deploy-frontend
make deploy-milvus
make deploy-worker
```

Should you change anything (in the code or the docker (compose) configuration),
the same commands can be used to update the deployment.

### Performance tuning

The API can handle a specified number or request concurrently, as specified by the `WEB_CONCURRENCY` environment variable of uvicorn in the api container. This can be adjusted in the docker-compose yaml file.

Several workers can be spawned on a single machine, depending on the available CPU and memory. The following command allows to specify the number of workers replicas:
```
docker --context idios-worker compose -p idios-worker -f docker/docker-compose.worker.yml up -d --build --remove-orphans --scale worker=4
```

### Simpler deployment

If scaling isn't an issue,
[docker-compose.minimal.yml](./docker/docker-compose.minimal.yml) shows the
minimal setup to run Idios on a single server, without TLS certificates. In
this scenario, we assume that this repository is directly on the production
machine, and issue the commands in a remote SSH shell. All volumes are in same
folder as the compose file.

Edit the lines marked with a `#TOEDIT` in the
[docker-compose.minimal.yml](./docker/docker-compose.minimal.yml) and run
```
make up COMPONENT=minimal
```

This should build and start the idios simple deployment (it takes about 20
minutes on a residential connection to download the pip packages and the docker
images).

Alternatively, a docker context on a development machine can be used to build
the images of the api and worker containers on the production server without
having the code on it. Only the
[docker-compose.minimal.yml](./docker/docker-compose.minimal.yml) and
[milvus.yaml](./docker/milvus.yaml) need to be copied on the server. The images
can be built from the development machine with:
```
docker --context idios-minimal compose -p idios -f docker/docker-compose.minimal.yml build api
docker --context idios-minimal compose -p idios -f docker/docker-compose.minimal.yml build worker
```
The -p parameter should match the folder containing the two yaml files on the
server.


## Dump/restore collections

The idios API includes dump and restore endpoints to allow for the
backup/restoration, migration or out of band analysis of the collections. The
[dump.py](./api/dump.py) script iteratively dumps the entities of the
collection by batch to avoid time outs. Its usage is described with the `-h`
command line option. The resulting json files can be loaded into another Idios
instance with a command like:
```
for f in dump-dir/*; do curl -H 'Content-Type: application/json' -d "@$f" http://localhost:4213/models/vit_b32/restore; done
```

## Orders of magnitudes

Following are some empirical measures :

### Memory usage

For 168803 images stored in a 512-dimensional vector embedding (vit-b32):
- etcd     379MB
- milvus   2GB
- minio    160MB
- rabbitmq 130MB
- worker   830MB
- api      24MB

For 471838 images stored in a 512-dimensional vector embedding (vit-b32):
- etcd      126M
- milvus    3G
- minio     135M
- rabbitmq  131M
- worker    889MB
- api       71MB

### Disk usage

For 168803 images stored in a 512-dimensional vector embedding (vit-b32):
- etcd       8 KiB
- milvus   680 MiB
- minio    1.6 GiB
- rabbitmq 472 KiB

For 471838 images stored in a 512-dimensional vector embedding (vit-b32):
- etcd      411M
- milvus    2.5G
- minio     1.8G
- rabbitmq  448K

A dump of the urls and embeddings (no metadata) takes 5.1GB.

### Timings

On an server with an Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz 32 GB, no GPU:
- using the API to add images (including embedding computation) can be done at
  2.5 images/s. 4 workers bring this rate to 10 images/s, but more seem to be
  counterproductive.
- similarity search takes ~0.5 seconds, ~1.5 if milvus is busy adding images.
- dumpinging 471838 entities takes about an hour

On a laptop with an i7-4720HQ CPU, 16GB, no GPU:
- Restoring entities takes about 2s per 1000-entities batch. It can double (if
  milvus does some housekeeping background task ?)
- Batching insertions in milvus has a very significant impact : 1000 individual
  insertions take ~7 s, but inserting a 1000-batch takes ~100ms
- Batching embedding computations has a smaller impact : 100 individual
  computations take ~9.3s, but 100-batch computation takes ~6.7s
- Querying by url takes around 400ms
- Downloading an image and computing its vit_b32 embedding takes around 200ms
- Inserting an image takes around 4ms

## References

- [FastAPI](https://fastapi.tiangolo.com/tutorial/)
- [Milvus API](https://milvus.io/docs/check_collection.md)
- [Docker](https://docs.docker.com/reference/)
