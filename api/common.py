embedding_dimensions = {
    "vit_b32": 512,
    # "vit_l14" : 4096,
    "resnet50": 2048,  # Embedding dimension for ResNet50

}

MAX_METADATA_LENGTH = 65535

# As of 2.3, the maximum number of items returned by pymilvus is:
MAX_MILVUS_PAGINATION = 16384

JOB_QUEUE_NAME = "idios_rpc_queue"
