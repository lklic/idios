DIMENSIONS = {
    "vit_b32": 512,
    # "vit_l14" : 4096,
    "sift20": 128,
    "sift100": 128,
}

INDEX_PARAMS = {
    "vit_b32": {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 2048},
    },
    "sift20": {
        "metric_type": "L2",
        # https://milvus.io/docs/benchmark.md#Test-pipeline
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 200},
    },
    "sift100": {
        "metric_type": "L2",
        # https://milvus.io/docs/benchmark.md#Test-pipeline
        "index_type": "HNSW",
        "params": {"M": 8, "efConstruction": 200},
    },
}

# https://milvus.io/docs/v1.1.1/performance_faq.md
SEARCH_PARAMS = {
    "vit_b32": {
        "nprobe": 64,
    },
    "sift20": {"ef": 100},
    "sift100": {"ef": 100},
}

CARDINALITIES = {
    "vit_b32": 1,
    "sift20": 20,
    "sift100": 100,
}

MAX_METADATA_LENGTH = 65535

# As of 2.3, the maximum number of items returned by pymilvus is:
MAX_MILVUS_PAGINATION = 16384

JOB_QUEUE_NAME = "idios_rpc_queue"
