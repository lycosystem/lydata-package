"""Submodule to collect data interactively using a simple web interface.

The idea is to dynamically serve a JSON schema from the `lydata.schema` module. This
can then be loaded into the `index.html` file where the `JSON Editor`_ is used to
create a form for the user to fill in. This collects a list of patient records that
will then be transformed into a :py:class:`lydata.accessor.LyDataFrame`, which the user
can then download as a CSV file.
"""

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import RootModel
from starlette.responses import FileResponse, HTMLResponse

import lydata

app = FastAPI(
    title="LyData Collector",
    description=(
        "A simple web interface to collect data for the LyData datasets. "
        "This is a prototype and not intended for production use."
    ),
    version=lydata.__version__,
)

BASE_DIR = Path(__file__).parent


@app.get("/")
def serve_index() -> HTMLResponse:
    """Serve the index.html file."""
    with open(BASE_DIR / "index.html") as file:
        content = file.read()
    return HTMLResponse(content=content)


@app.get("/schema")
def serve_schema() -> dict[str, Any]:
    """Serve the JSON schema for the patient and tumor records."""
    from lydata.schema import create_full_record_model, get_default_modalities

    modalities = get_default_modalities()
    schema = create_full_record_model(modalities, title="Record")

    root_schema = RootModel[list[schema]]
    return root_schema.model_json_schema()


@app.get("/collector.js")
def serve_collector_js() -> FileResponse:
    """Serve the collector.js file."""
    return FileResponse(BASE_DIR / "collector.js")


def main() -> None:
    """Run the FastAPI app using Uvicorn."""
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
