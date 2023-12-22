FROM python:3.10 AS base

ARG UID=1000
RUN groupadd -r -g $UID user && \
    useradd --create-home --shell /bin/bash --no-log-init -r -g user -u $UID user

USER user
WORKDIR /app
ENV PATH="/home/user/.local/bin:${PATH}"


FROM base AS dev

USER root
RUN --mount=type=cache,target=/home/user/.cache/pip chown --recursive user:user /home/user/.cache
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

USER user
RUN mkdir -p /home/user/.cache/huggingface
COPY requirements*.txt /tmp
RUN --mount=type=cache,target=/home/user/.cache/pip pip install -r /tmp/requirements-dev.txt



FROM base AS api

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt


# If running behind a proxy like Nginx or Traefik add --proxy-headers
CMD ["uvicorn", "main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "4213"]
HEALTHCHECK --start-period=1s --interval=10s --timeout=1s --retries=5 \
  CMD curl -f http://localhost:4213/ping || exit 1

COPY . /app



FROM base AS worker

USER root
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

USER user
COPY ./requirements-worker.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Download models
COPY ./embeddings.py /app/
RUN python embeddings.py

ENV PYTHONUNBUFFERED=1
STOPSIGNAL SIGINT
CMD ["python", "worker.py"]
HEALTHCHECK --start-period=5s --interval=5s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:8000/health || exit 1

COPY . /app



FROM milvusdb/milvus:v2.2.2 AS milvus

CMD ["milvus", "run", "standalone"]
HEALTHCHECK --start-period=1s --interval=10s --timeout=1s --retries=5 \
  CMD curl -f http://localhost:9091/api/v1/health || exit 1

COPY ./milvus.yaml /milvus/configs/milvus.yaml
