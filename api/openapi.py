#!/usr/bin/env python3

from fastapi.openapi.utils import get_openapi
from main import app
import yaml
import argparse


specs = get_openapi(
    title=app.title if app.title else None,
    version=app.version if app.version else None,
    openapi_version=app.openapi_version if app.openapi_version else None,
    description=app.description if app.description else None,
    routes=app.routes if app.routes else None,
)


# Use > style only for long strings
def string_representer(dumper, data):
    if len(data) > 60 and len(data.split(" ")[0]) < 60:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=">")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


yaml.add_representer(str, string_representer)

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", default="openapi.yaml")
args = parser.parse_args()
with open(args.output, "w") as f:
    yaml.dump(specs, f, width=80)
