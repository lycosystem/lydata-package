"""Submodule to collect data interactively using a simple web interface.

With the simply command

.. code-block:: bash

    uvx --from lydata lycollect

One can start a very basic web server that serves an interactive UI at
``http://localhost:8000/``. There, one can enter patient, tumor, and lymphatic
involvement data one by one. When completed, the "submit" button will parse, validate,
and convert the data to serve a downloadable CSV file.

This resulting CSV file is in the correct format to be used in `LyProX`_ and for
inference using our `lymph-model`_ library.

.. _LyProX: https://lyprox.org
.. _lymph-model: https://lymph-model.readthedocs.io
"""

import io
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import RootModel
from starlette.responses import FileResponse, HTMLResponse

import lydata
import lydata.validator

app = FastAPI(
    title="LyData Collector",
    description=(
        "A simple web interface to collect data for the LyData datasets. "
        "This is a prototype and not intended for production use."
    ),
    version=lydata.__version__,
)

BASE_DIR = Path(__file__).parent
modalities = lydata.schema.get_default_modalities()
RecordModel = lydata.schema.create_full_record_model(modalities, title="Record")
ROOT_MODEL = RootModel[list[RecordModel]]


@app.get("/")
def serve_index() -> HTMLResponse:
    """Serve the index.html file."""
    with open(BASE_DIR / "index.html") as file:
        content = file.read()
    return HTMLResponse(content=content)


@app.get("/schema")
def serve_schema() -> dict[str, Any]:
    """Serve the JSON schema for the patient and tumor records."""
    return ROOT_MODEL.model_json_schema()


@app.get("/collector.js")
def serve_collector_js() -> FileResponse:
    """Serve the collector.js file."""
    return FileResponse(BASE_DIR / "collector.js")


@app.post("/submit")
async def process(data: RootModel) -> StreamingResponse:
    """Convert the submitted data to a DataFrame."""
    logger.info(f"Received data: {data.root}")
    flattened_records = []

    for record in data.root:
        flattened_record = lydata.validator.flatten(record)
        logger.debug(f"Flattened record: {flattened_record}")
        flattened_records.append(flattened_record)

    df = pd.DataFrame(flattened_records)
    df.columns = pd.MultiIndex.from_tuples(flattened_record.keys())
    logger.info(df.patient.core.head())

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    logger.success("Data prepared for download")
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=lydata_records.csv"},
    )


def main() -> None:
    """Run the FastAPI app using Uvicorn."""
    import uvicorn

    logger.enable("lydata")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
