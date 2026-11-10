"""Set the connector version of an Airbyte source definition, or register a connector image as a new one.

ELT-Bench documents Terraform provider 0.6.5. Two sources that ship with Airbyte 2.5 no longer accept the fields
that provider writes (see the setup notes in the README). Both commands below restore a version that does.

    export ELT_RLVR_AIRBYTE_CLIENT_ID=... ELT_RLVR_AIRBYTE_CLIENT_SECRET=...
    python scripts/airbyte_source_definition.py --elt-bench-root /path/to/ELT-Bench pin S3 4.12.2
    python scripts/airbyte_source_definition.py --elt-bench-root /path/to/ELT-Bench \
        register "MongoDb 1.5.19" airbyte/source-mongodb-v2 1.5.19

``pin`` changes the definition Airbyte ships under that name. ``register`` adds a definition and prints its id,
which belongs in ``ELT_RLVR_AIRBYTE_DEFINITION_IDS``, for example ``{"mongodb": "<id>"}``.
"""

import argparse
import json
import os
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description="Pin or register an Airbyte source definition.")
    parser.add_argument("--elt-bench-root", type=Path, required=True)
    parser.add_argument(
        "--api-url",
        default=os.environ.get("ELT_RLVR_AIRBYTE_API_URL", "http://localhost:8000/api/public/v1"),
        help="Airbyte's public API",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    pin = commands.add_parser("pin", help="set the connector version of a definition that Airbyte ships")
    pin.add_argument("name", help="name of the source definition as Airbyte lists it, for example S3")
    pin.add_argument("version", help="connector image tag, for example 4.12.2")
    register = commands.add_parser("register", help="add a connector image as a custom source definition")
    register.add_argument("name", help="name of the new definition")
    register.add_argument("image", help="connector image, for example airbyte/source-mongodb-v2")
    register.add_argument("version", help="connector image tag")
    arguments = parser.parse_args()

    credential_file = arguments.elt_bench_root / "setup" / "airbyte" / "airbyte_credential.json"
    workspace = json.loads(credential_file.read_text(encoding="utf-8"))["workspace_id"]
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
        # Airbyte pulls the connector image and reads its specification before it answers.
        response = httpx.post(f"{internal_api}/{path}", headers=headers, json=body, timeout=900)
        if response.status_code >= 400:
            raise SystemExit(f"{path} failed with status {response.status_code}: {response.text[:600]}")
        return response.json()

    listed = call("source_definitions/list_for_workspace", {"workspaceId": workspace})["sourceDefinitions"]
    matches = [definition for definition in listed if definition["name"] == arguments.name]
    if arguments.command == "pin":
        if len(matches) != 1:
            raise SystemExit(f"{len(matches)} source definitions are named {arguments.name!r}")
        definition = call(
            "source_definitions/update",
            {
                "sourceDefinitionId": matches[0]["sourceDefinitionId"],
                "dockerImageTag": arguments.version,
                "workspaceId": workspace,
            },
        )
    elif matches:
        definition = matches[0]
    else:
        definition = call(
            "source_definitions/create_custom",
            {
                "workspaceId": workspace,
                "sourceDefinition": {
                    "name": arguments.name,
                    "dockerRepository": arguments.image,
                    "dockerImageTag": arguments.version,
                    "documentationUrl": "https://docs.airbyte.com/integrations/sources",
                },
            },
        )
    print(f"{definition['name']}: {definition['dockerRepository']}:{definition['dockerImageTag']}")
    print(f"definition id: {definition['sourceDefinitionId']}")


if __name__ == "__main__":
    main()
