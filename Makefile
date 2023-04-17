default:

COMPOSE=docker compose -p idios -f docker/docker-compose.yml
RUN=${COMPOSE} run --remove-orphans --build --rm worker

WATCH_CMD=find -not -path '*/.*' -not -path '*__pycache__*' | entr -cd make watched
watch:
	${WATCH_CMD} ; while [ $$? -ne 0 ]; do ${WATCH_CMD}; done

watched: black test doc/openapi.yaml

shell:
	${RUN} bash

doc/openapi.yaml: api/main.py
	${RUN} ./openapi.py
	mv api/openapi.yaml doc

black:
	docker run --rm --volume $$(pwd)/api:/src --workdir /src pyfound/black:latest_release black .

build:
	${COMPOSE} build --progress=plain

test:
	${RUN} python -m pytest --testmon --tb=short

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
