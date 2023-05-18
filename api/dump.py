#!/bin/env python3

import requests
import argparse
import json
import os
import sys

parser = argparse.ArgumentParser(description="Dump the content of an idios collection")
parser.add_argument(
    "model_endpoint",
    help='The URL prefix to the collection endpoint (e.g. "https://api.idios.it/models/vit_b32")',
)
parser.add_argument("output_prefix", help="The prefix for the output file name")
parser.add_argument(
    "batch_size",
    type=int,
    nargs="?",
    default=1000,
    help="The batch size for each request (default 1000) to avoid timeouts",
)

args = parser.parse_args()

output_dir = os.path.dirname(args.output_prefix)
os.makedirs(output_dir, exist_ok=True)

cursor = ""
batch_count = 1

while True:
    if cursor:
        print(f"Requesting entries after {cursor}")
    else:
        print(f"Requesting entries")

    try:
        response = requests.post(
            f"{args.model_endpoint}/export",
            json={"limit": args.batch_size, "cursor": cursor},
        )
        response.raise_for_status()
        entries = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: {e} decoding `{response.text}`")
        sys.exit(1)

    if not entries:
        break

    output_file = f"{args.output_prefix}_{str(batch_count).zfill(8)}.json"
    print(f"Writing {len(entries)} entries to {output_file}")
    with open(output_file, "w") as f:
        json.dump(entries, f)

    cursor = entries[-1]["url"]
    batch_count += 1
