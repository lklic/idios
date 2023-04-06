FROM python:3.10

ARG UID=1000
RUN groupadd -r -g $UID user && \
    useradd --create-home --shell /bin/bash --no-log-init -r -g user -u $UID user

RUN --mount=type=cache,target=/home/user/.cache/pip chown --recursive user:user /home/user/.cache

USER user

RUN mkdir -p /home/user/.cache/huggingface

ENV PATH="/home/user/.local/bin:${PATH}"
COPY requirements*.txt /tmp

RUN --mount=type=cache,target=/home/user/.cache/pip pip install -r /tmp/requirements-dev.txt
