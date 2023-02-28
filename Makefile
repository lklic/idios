default:

WATCH_CMD=find -not -path '*/.*' -not -path '*__pycache__*' | entr -cd make watched
watch:
	${WATCH_CMD} ; while [ $$? -ne 0 ]; do ${WATCH_CMD}; done

watched: black
	docker compose exec api python -m pytest --testmon --tb=short

shell:
	docker compose run --rm api bash

black:
	docker run --rm --volume $$(pwd):/src --workdir /src pyfound/black:latest_release black .

build:
	docker compose build

test: build
	docker compose run --rm api python -m pytest

up: build
	docker compose up --remove-orphans -d

down:
	docker compose down

logs:
	docker compose logs -f

rm-volumes:
	docker volume rm idios_etcd idios_milvus idios_minio
