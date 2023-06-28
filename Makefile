# What runs when `make` is called without argument (i.e. nothing)
default:

################################################################################
# Development commands
################################################################################

# The default docker compose configuration file is the development one
COMPONENT=dev

# Avoid the repetition of docker compose parameter
COMPOSE=docker compose -p idios -f docker/docker-compose.${COMPONENT}.yml

# Avoid the repetition of to start something in the dev container.
# The dev container has all the dependencies for both the api and the worker,
# plus development specific tools.
RUN=${COMPOSE} run --remove-orphans --build --rm dev

# Watch for file changes and runs the `watched` command.
# This requires entr to be installed on the development system
WATCH_CMD=find -not -path '*/.*' -not -path '*__pycache__*' -o -name '.dockerignore' | entr -cd make watched
watch:
	${WATCH_CMD} ; while [ $$? -ne 0 ]; do ${WATCH_CMD}; done

# Command to run on a file change.
# By default, it lints, test and regenerate the API reference,
# but other commands can be specified, either as dependencies
# or as build command
watched: black test doc/openapi.yaml

# Start a shell in the dev container
shell:
	${RUN} bash

# Generate the SwaggerUI interactive API reference
doc/openapi.yaml: api/openapi.py api/main.py
	${RUN} ./openapi.py -o ./openapi.yaml
	mv api/openapi.yaml $@

# Runs the black linter/formatter on python files
black:
	docker run --rm --volume $$(pwd)/api:/src --workdir /src pyfound/black:latest_release black .

# Build all the containers
build:
	${COMPOSE} build --progress=plain

# Runs the tests that need to be rerun, as defined by pytest-testmon
test:
	${RUN} python -m pytest --testmon --tb=short

# Runs an end to end test, calling the API in a way that it interacts with all other components
# The echo commands informally compare the actual to the expected result.
API_URL=localhost:4213
integration-test:
	curl -X POST ${API_URL}/models/vit_b32/urls
	@echo =?=[]
	curl -H "Content-Type: application/json" -d '{"url": "https://iiif.itatti.harvard.edu/iiif/2/yashiro!letters-jp!letter_001.pdf/full/full/0/default.jpg"}' ${API_URL}/models/vit_b32/add
	@echo =?=
	curl -X POST ${API_URL}/models/vit_b32/urls
	@echo =?=[url]
	curl -H "Content-Type: application/json" -d '{"url": "https://iiif.itatti.harvard.edu/iiif/2/yashiro!letters-jp!letter_001.pdf/full/full/0/default.jpg"}' ${API_URL}/models/vit_b32/search
	@echo =?=[result]
	curl -H "Content-Type: application/json" -d '{"url_left": "https://iiif.itatti.harvard.edu/iiif/2/yashiro!letters-jp!letter_001.pdf/full/full/0/default.jpg", "url_right": "https://iiif.itatti.harvard.edu/iiif/2/yashiro!letters-jp!letter_001.pdf/full/full/0/default.jpg"}' ${API_URL}/models/vit_b32/compare
	@echo =?=100
	curl ${API_URL}/models/vit_b32/count
	@echo =?=1
	curl -H "Content-Type: application/json" -d '{"url": "https://iiif.itatti.harvard.edu/iiif/2/yashiro!letters-jp!letter_001.pdf/full/full/0/default.jpg"}' ${API_URL}/models/vit_b32/search_add
	@echo =?={"detail":"Image already inserted"}
	curl -H "Content-Type: application/json" -d '{"url": "https://iiif.itatti.harvard.edu/iiif/2/yashiro!letters-jp!letter_001.pdf/full/full/0/default.jpg"}' ${API_URL}/models/vit_b32/remove
	@echo =?=
	curl ${API_URL}/models/vit_b32/count
	@echo =?=0

# Reset the milvus database content
destroy-milvus:
	${RUN} python -c 'import milvus; milvus.destroy_all_data_from_all_collections_in_the_whole_database()'

# Starts the whole development environment
up:
	${COMPOSE} up --build --remove-orphans -d

# Stops and tear down the development environment
down:
	${COMPOSE} down

# Shows the logs of the various components of the development environment
# The c variable is a hack to pass parameters on the make command line.
# For example, to see only the logs of the api and worker container, use:
# make logs c="api worker"
# Use CTRL+C to stop the logs
logs:
	${COMPOSE} logs -f ${c}

# Destroys volumes storing the state of the development environment
rm-volumes:
	docker volume rm idios_etcd idios_milvus idios_minio idios_rabbitmq

# A short cut to a generic compose command
# The c variable is a hack to pass parameters on the make command line.
# For example, to start the milvus container (only) in the background, use:
# make compose c="up -d milvus"
compose:
	${COMPOSE} ${c}



################################################################################
# Deployment commands
################################################################################

# A compose command prefix that that takes one of components  of the deployment
# illustration from the README (frontend, milvus and worker) as a parameter
DEPLOY_COMPOSE=docker --context idios-${COMPONENT} compose -p idios-${COMPONENT} -f docker/docker-compose.${COMPONENT}.yml

# A generic deploy command that takes the COMPONENT variable as a "make parameter"
# followed by shortcuts for each component
deploy:
	${DEPLOY_COMPOSE} up --build --remove-orphans -d

deploy-frontend:
	make deploy COMPONENT=frontend

deploy-milvus:
	make deploy COMPONENT=milvus

deploy-worker:
	make deploy COMPONENT=worker

# A generic log command that takes the COMPONENT variable as a "make parameter"
# followed by shortcuts for each component
deploy-logs:
	${DEPLOY_COMPOSE} logs -f

deploy-logs-frontend:
	make deploy-logs COMPONENT=frontend

deploy-logs-milvus:
	make deploy-logs COMPONENT=milvus

deploy-logs-worker:
	make deploy-logs COMPONENT=worker

# A quick test that each component is alive
deployment-test:
	curl https://api.idios.org/ping #TOEDIT
	curl https://queue.idios.org | grep 'RabbitMQ Management' #TOEDIT
	curl https://milvus.idios.org/api/v1/health #TOEDIT
