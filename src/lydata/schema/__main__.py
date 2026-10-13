"""Write the JSON schema of a patient record to a file."""

import json
import sys
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, RootModel

from lydata.schema import create_full_record_model, get_default_modalities


def _write_schema_to_file(
    schema: type[BaseModel] | None = None,
    file_path: Path = Path("schema.json"),
) -> None:
    """Write the Pydantic schema to a file."""
    if schema is None:
        modalities = get_default_modalities()
        schema = create_full_record_model(modalities, model_name="Record")

    root_schema = RootModel[list[schema]]

    with open(file_path, "w") as f:
        json_schema = root_schema.model_json_schema()
        f.write(json.dumps(json_schema, indent=2))

    logger.success(f"Schema written to {file_path}")


if __name__ == "__main__":
    logger.enable("lydata")
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    _write_schema_to_file()
