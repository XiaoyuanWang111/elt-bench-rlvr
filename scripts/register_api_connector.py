"""Register ELT-Bench's REST API connector in Airbyte without the web UI.

ELT-Bench's README registers the connector by hand (Builder > Import a YAML, then publish). This script makes the
same two API calls, so the step also works on a machine without a browser, and prints the definition id that
belongs in ``setup/airbyte/airbyte_credential.json`` as ``api_definition_id``.

    export ELT_RLVR_AIRBYTE_CLIENT_ID=... ELT_RLVR_AIRBYTE_CLIENT_SECRET=...
    python scripts/register_api_connector.py --elt-bench-root /path/to/ELT-Bench --single-page

``--single-page`` registers the connector without its paginator. ELT-Bench's API server ignores ``limit`` and
``offset`` and answers every request with the whole table, while the manifest's paginator asks for the next page
until one comes back with fewer than 9,973 rows. A table with at least that many rows (40 of the 193 API tables, in
30 tasks) is therefore read again and again and the sync never ends. One request already returns the table.
"""

import argparse
import json
import os
from pathlib import Path

import httpx
import yaml

MANIFESTS = {"snowflake": "elt_snowflake.yaml", "databricks": "elt_databricks.yaml", "redshift": "elt_redshift.yaml"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Register ELT-Bench's REST API connector through Airbyte's API.")
    parser.add_argument("--elt-bench-root", type=Path, required=True)
    parser.add_argument("--destination", choices=sorted(MANIFESTS), default="snowflake")
    parser.add_argument("--name", default="ELT Bench")
    parser.add_argument("--single-page", action="store_true", help="drop the paginator of every stream")
    parser.add_argument(
        "--api-url",
        default=os.environ.get("ELT_RLVR_AIRBYTE_API_URL", "http://localhost:8000/api/public/v1"),
        help="Airbyte's public API",
    )
    arguments = parser.parse_args()

    setup = arguments.elt_bench_root / "setup"
    workspace = json.loads((setup / "airbyte" / "airbyte_credential.json").read_text(encoding="utf-8"))["workspace_id"]
    manifest = yaml.safe_load((setup / MANIFESTS[arguments.destination]).read_text(encoding="utf-8"))
    if arguments.single_page:
        for stream in manifest["definitions"]["streams"].values():
            stream["retriever"].pop("paginator", None)

    public_api = arguments.api_url.rstrip("/")
    internal_api = public_api.removesuffix("/public/v1") + "/v1"
    token = httpx.post(
        f"{public_api}/applications/token",
        json={
            "client_id": os.environ["ELT_RLVR_AIRBYTE_CLIENT_ID"],
            "client_secret": os.environ["ELT_RLVR_AIRBYTE_CLIENT_SECRET"],
            "grant-type": "client_credentials",
        },
        timeout=30,
    )
    token.raise_for_status()
    headers = {"authorization": f"Bearer {token.json()['access_token']}", "accept": "application/json"}

    def call(path: str, body: dict) -> dict:
        response = httpx.post(f"{internal_api}/{path}", headers=headers, json=body, timeout=300)
        if response.status_code >= 400:
            raise SystemExit(f"{path} failed with status {response.status_code}: {response.text[:600]}")
        return response.json()

    project = call(
        "connector_builder_projects/create",
        {"workspaceId": workspace, "builderProject": {"name": arguments.name, "draftManifest": manifest}},
    )
    published = call(
        "connector_builder_projects/publish",
        {
            "workspaceId": workspace,
            "builderProjectId": project["builderProjectId"],
            "name": arguments.name,
            "initialDeclarativeManifest": {
                "manifest": manifest,
                "spec": {"connectionSpecification": manifest["spec"]["connection_specification"]},
                "description": "REST API source of ELT-Bench",
                "version": 1,
            },
        },
    )
    print(f"api_definition_id: {published['sourceDefinitionId']}")


if __name__ == "__main__":
    main()
