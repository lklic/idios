default:

COMPOSE=docker compose -p idios -f docker/docker-compose.yml
RUN=${COMPOSE} run --remove-orphans --build --rm dev

WATCH_CMD=find -not -path '*/.*' -not -path '*__pycache__*' -o -name '.dockerignore' | entr -cd make watched
watch:
	${WATCH_CMD} ; while [ $$? -ne 0 ]; do ${WATCH_CMD}; done

watched: black test doc/openapi.yaml

shell:
	${RUN} bash

doc/openapi.yaml: api/openapi.py api/main.py
	${RUN} ./openapi.py -o doc/openapi.py

black:
	docker run --rm --volume $$(pwd)/api:/src --workdir /src pyfound/black:latest_release black .

build:
	${COMPOSE} build --progress=plain

test:
	${RUN} python -m pytest --testmon --tb=short
	
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
	curl -H "Content-Type: application/json" -d '{"url": "https://iiif.itatti.harvard.edu/iiif/2/yashiro!letters-jp!letter_001.pdf/full/full/0/default.jpg"}' ${API_URL}/models/vit_b32/remove
	@echo =?=
	curl ${API_URL}/models/vit_b32/count
	@echo =?=0

destroy-milvus:
	${RUN} python -c 'import milvus; milvus.destroy_all_data_from_all_collections_in_the_whole_database()'

up:
	${COMPOSE} up --build --remove-orphans -d

down:
	${COMPOSE} down

logs:
	${COMPOSE} logs -f ${c}

rm-volumes:
	docker volume rm idios_etcd idios_milvus idios_minio idios_rabbitmq

compose:
	${COMPOSE} ${c}

DEPLOY_COMPOSE=docker --context idios-${COMPONENT} compose -p idios-${COMPONENT} -f docker/docker-compose.${COMPONENT}.yml

deploy:
	${DEPLOY_COMPOSE} up --build --remove-orphans -d

deploy-frontend:
	make deploy COMPONENT=frontend

deploy-milvus:
	make deploy COMPONENT=milvus

deploy-worker:
	make deploy COMPONENT=worker

deploy-logs:
	${DEPLOY_COMPOSE} logs -f

deploy-logs-frontend:
	make deploy-logs COMPONENT=frontend

deploy-logs-milvus:
	make deploy-logs COMPONENT=milvus

deploy-logs-worker:
	make deploy-logs COMPONENT=worker

deployment-test:
	curl https://api.idios.org/ping #TOEDIT
	curl https://queue.idios.org | grep 'RabbitMQ Management' #TOEDIT
	curl https://milvus.idios.org/api/v1/health #TOEDIT
