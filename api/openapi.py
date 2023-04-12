#!/usr/bin/env python3

from fastapi.openapi.utils import get_openapi
from main import app
import yaml

specs = get_openapi(
    title=app.title if app.title else None,
    version=app.version if app.version else None,
    openapi_version=app.openapi_version if app.openapi_version else None,
    description=app.description if app.description else None,
    routes=app.routes if app.routes else None,
)

with open(f"openapi.yaml", "w") as f:
    yaml.dump(specs, f)
