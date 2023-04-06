default:

COMPOSE=docker compose -p idios -f docker/docker-compose.yml

WATCH_CMD=find -not -path '*/.*' -not -path '*__pycache__*' | entr -cd make watched
watch:
	${WATCH_CMD} ; while [ $$? -ne 0 ]; do ${WATCH_CMD}; done

watched: black up test


shell:
	${COMPOSE} run --rm worker bash

black:
	docker run --rm --volume $$(pwd)/api:/src --workdir /src pyfound/black:latest_release black .

build:
	${COMPOSE} build --progress=plain

test: build
	${COMPOSE} run --rm worker python -m pytest --testmon --tb=short

up: build
	${COMPOSE} up --remove-orphans -d

down:
	${COMPOSE} down

logs:
	${COMPOSE} logs -f ${c}

rm-volumes:
	docker volume rm idios_etcd idios_milvus idios_minio idios_rabbitmq

compose:
	${COMPOSE} ${c}
