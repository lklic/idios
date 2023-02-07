import json
from flask import Flask, request
from PIL import Image
from clip import CLIP
import requests
import numpy as np
from milvus import Milvus, IndexType, MetricType

app = Flask(__name__)
milvus = Milvus()

# Connect to Milvus server
milvus.connect(host='127.0.0.1', port='19530')

# Instantiate CLIP model
clip = CLIP()

# Default index
index_name = "default"

@app.route('/index/image', methods=['PUT'])
def add_image():
    data = request.get_json()
    url = data["url"]
    
    # Open image and resize if needed
    img = Image.open(requests.get(url, stream=True).raw)
    if img.width > 1000 or img.height > 1000:
        img.thumbnail((1000, 1000))
    img = np.array(img)
    
    # Get image embedding
    embedding = clip.get_embedding(img)
    
    # Insert embedding into Milvus
    status, ids = milvus.insert(collection_name=index_name, records=embedding)
    
    return json.dumps({"type": "IMAGE_ADDED"})

@app.route('/index/image', methods=['DELETE'])
def remove_image():
    data = request.get_json()
    url = data["url"]
    
    # Remove embedding from Milvus
    status = milvus.delete_by_vector(collection_name=index_name, vector=embedding)
    
    return json.dumps({"type": "IMAGE_REMOVED"})

@app.route('/index/searcher', methods=['POST'])
def search():
    data = request.get_json()
    url = data["url"]
    
    # Open image and resize if needed
    img = Image.open(requests.get(url, stream=True).raw)
    if img.width > 1000 or img.height > 1000:
        img.thumbnail((1000, 1000))
    img = np.array(img)
    
    # Get image embedding
    embedding = clip.get_embedding(img)
    
    # Search for similar images in Milvus
    status, results = milvus.search(collection_name=index_name, query_records=embedding, top_k=10)
    
    image_ids = [result.id for result in results]
    
    return json.dumps({"type": "SEARCH_RESULTS", "image_ids": image_ids})

@app.route('/', methods=['POST'])
def ping():
    data = request.get_json()
    if data["type"] == "PING":
        return json.dumps({"type": "PONG"})

if __name__ == '__main__':
    app.run(port=4213)
