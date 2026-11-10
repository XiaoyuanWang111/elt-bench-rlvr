"""Extract-load through a live Airbyte deployment, the engine of the official ELT-Bench tasks.

A connection from ``pipeline/extract_load.yaml`` is rendered as Terraform resources of the Airbyte provider, the
same resources an ELT-Bench agent writes by hand, so the benchmark's connector documentation applies unchanged and
Terraform validates the configuration. The engine then applies the resources, triggers one sync job through
Airbyte's API, waits for it, and destroys the resources.

Three things stay on the engine's side of the trust boundary:

- The destination. The policy configures sources and streams; the engine adds the destination resource with the
  warehouse credentials and points it at the rollout's namespace.
- Secrets. ``config.yaml`` shows the policy a placeholder such as ``<secret:postgres.config.password>`` in place
  of each secret. The engine passes every secret, and its own API token, to Terraform as a sensitive variable
  through the process environment. No secret is written to the rendered file, Terraform does not print sensitive
  values, and the engine also removes them from any message it returns.
- The set of sources. A host, URL, bucket or connection string in a configuration must occur in the task's own
  ``config.yaml``, so a connection cannot be pointed at a file that is not one of the task's sources, such as a
  published ground-truth table.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import yaml

from elt_rlvr.extract_load.base import ConnectionRejectedError, ExtractLoadEngine, StreamResult, table_name_for_stream
from elt_rlvr.pipeline import ConnectionConfig
from elt_rlvr.settings import destination_database
from elt_rlvr.warehouse.base import Warehouse

SOURCE_RESOURCES = {
    "postgres": "airbyte_source_postgres",
    "mongodb": "airbyte_source_mongodb_v2",
    "file": "airbyte_source_file",
    "s3": "airbyte_source_s3",
    "custom_api": "airbyte_source_custom",
}
DESTINATION_RESOURCES = {"snowflake": "airbyte_destination_snowflake"}
INPUT_DIRECTORIES = {"snowflake": "inputs", "databricks": "inputs_databricks", "redshift": "inputs_redshift"}
SECRET_KEYS = frozenset({"password", "aws_secret_access_key", "secret_access_key", "secret", "private_key", "token"})
LOCATOR_KEYS = frozenset({"host", "url", "endpoint", "bucket", "connection_string", "database"})
SECRET_PREFIX, SECRET_SUFFIX = "<secret:", ">"
PROVIDER_VERSION = "0.6.5"
SUCCEEDED, FAILED_STATES = "succeeded", ("failed", "cancelled", "incomplete")
TOKEN_VARIABLE = "airbyte_token"
MIN_SCRUBBED_LENGTH = 6
"""Secrets shorter than this (ELT-Bench's ``test`` keys) are not searched for in messages: removing every
occurrence of such a string would garble the message without protecting anything."""
MAX_ERROR_LINES = 30


def secret_placeholder(path: str) -> str:
    return f"{SECRET_PREFIX}{path}{SECRET_SUFFIX}"


def _is_secret(key: Any, value: Any) -> bool:
    return isinstance(value, str | int) and not isinstance(value, bool) and str(key).lower() in SECRET_KEYS


def collect_secrets(config: Any, path: str = "") -> dict[str, str]:
    """Map the placeholder of every secret in a task's ``config.yaml`` to its value."""
    secrets: dict[str, str] = {}
    if isinstance(config, Mapping):
        for key, value in config.items():
            child = f"{path}.{key}" if path else str(key)
            if _is_secret(key, value):
                secrets[secret_placeholder(child)] = str(value)
            else:
                secrets.update(collect_secrets(value, child))
    elif isinstance(config, list):
        for index, value in enumerate(config):
            secrets.update(collect_secrets(value, f"{path}[{index}]"))
    return secrets


def redact_secrets(config: Any, path: str = "") -> Any:
    """A copy of a task's ``config.yaml`` with every secret replaced by its placeholder."""
    if isinstance(config, Mapping):
        return {
            key: (
                secret_placeholder(f"{path}.{key}" if path else str(key))
                if _is_secret(key, value)
                else redact_secrets(value, f"{path}.{key}" if path else str(key))
            )
            for key, value in config.items()
        }
    if isinstance(config, list):
        return [redact_secrets(value, f"{path}[{index}]") for index, value in enumerate(config)]
    return config


def _leaves(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        return [leaf for child in value.values() for leaf in _leaves(child)]
    if isinstance(value, list):
        return [leaf for child in value for leaf in _leaves(child)]
    return [str(value)] if isinstance(value, str | int | float) and not isinstance(value, bool) else []


def _escape_interpolation(value: Any) -> Any:
    """Escape Terraform template sequences so that a configuration string is always taken literally."""
    if isinstance(value, str):
        return value.replace("${", "$${").replace("%{", "%%{")
    if isinstance(value, Mapping):
        return {key: _escape_interpolation(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_escape_interpolation(child) for child in value]
    return value


@dataclass(frozen=True)
class AirbyteSettings:
    """Connection values of one Airbyte deployment and one task's sources. Holds secrets; never logged."""

    api_url: str
    """Airbyte's public API as reachable from this process, e.g. ``http://localhost:8000/api/public/v1``."""
    provider_server_url: str
    """The same API as reachable from where Terraform runs."""
    workspace_id: str
    username: str | None = field(default=None, repr=False)
    password: str | None = field(default=None, repr=False)
    client_id: str | None = field(default=None, repr=False)
    client_secret: str | None = field(default=None, repr=False)
    custom_definition_id: str | None = None
    destination_resource: str = "airbyte_destination_snowflake"
    destination_configuration: Mapping[str, Any] = field(default_factory=dict, repr=False)
    uppercase_namespace: bool = True
    """Whether the destination stores an unquoted schema name in upper case, as Snowflake does."""
    secrets: Mapping[str, str] = field(default_factory=dict, repr=False)
    allowed_locators: tuple[str, ...] = ()
    terraform_command: tuple[str, ...] = ("terraform",)
    sync_timeout_seconds: float = 3600.0
    poll_seconds: float = 5.0


def load_elt_bench_settings(elt_bench_root: Path, task_id: str, destination: str = "snowflake") -> AirbyteSettings:
    """Read the Airbyte and destination values that ELT-Bench's ``write_config.py`` generated for a task."""
    if destination not in DESTINATION_RESOURCES:
        raise ValueError(f"the Airbyte engine has no destination resource for {destination!r} yet")
    config = yaml.safe_load((elt_bench_root / INPUT_DIRECTORIES[destination] / task_id / "config.yaml").read_text())
    airbyte, warehouse = config["Airbyte"]["config"], config[destination]["config"]
    sources = {key: value for key, value in config.items() if key not in ("Airbyte", destination)}
    destination_configuration = {
        "host": f"{warehouse['account']}.snowflakecomputing.com",
        "role": warehouse["role"],
        "warehouse": warehouse["warehouse"],
        "database": destination_database(),
        "username": warehouse["username"],
        "credentials": {"username_and_password": {"password": warehouse["password"]}},
    }
    return AirbyteSettings(
        api_url=os.environ.get("ELT_RLVR_AIRBYTE_API_URL", "http://localhost:8000/api/public/v1"),
        provider_server_url=os.environ.get("ELT_RLVR_AIRBYTE_PROVIDER_URL", airbyte["server_url"]),
        workspace_id=airbyte["workspace_id"],
        username=airbyte.get("username"),
        password=airbyte.get("password"),
        client_id=os.environ.get("ELT_RLVR_AIRBYTE_CLIENT_ID"),
        client_secret=os.environ.get("ELT_RLVR_AIRBYTE_CLIENT_SECRET"),
        custom_definition_id=airbyte.get("custom_api_definition_id"),
        destination_resource=DESTINATION_RESOURCES[destination],
        destination_configuration=destination_configuration,
        secrets=collect_secrets(sources),
        allowed_locators=tuple(_leaves(sources)),
        terraform_command=tuple(os.environ.get("ELT_RLVR_TERRAFORM", "terraform").split()),
    )


@dataclass
class RenderedConnection:
    """One connection as Terraform configuration plus the sensitive values it refers to."""

    resources: dict[str, Any]
    """Terraform configuration in JSON syntax. It names secrets only as ``${var.<name>}``."""
    variables: dict[str, str] = field(default_factory=dict, repr=False)
    """Values of the sensitive variables, handed to Terraform through ``TF_VAR_<name>``."""

    def reference(self, value: str) -> str:
        """Register a sensitive value and return the expression that refers to it."""
        name = f"secret_{len(self.variables)}"
        self.variables[name] = value
        return f"${{var.{name}}}"


class AirbyteEngine(ExtractLoadEngine):
    """Runs each connection as Terraform-managed Airbyte resources plus one sync job."""

    def __init__(
        self,
        *,
        settings: AirbyteSettings | None = None,
        elt_bench_root: str | None = None,
        task_id: str | None = None,
        destination: str = "snowflake",
    ) -> None:
        if settings is None:
            if elt_bench_root is None or task_id is None:
                raise ValueError("AirbyteEngine needs either settings or an ELT-Bench root and task")
            settings = load_elt_bench_settings(Path(elt_bench_root), task_id, destination)
        self._settings = settings
        self._plugin_cache = Path(tempfile.gettempdir()) / "elt_rlvr_terraform_plugins"
        self._plugin_cache.mkdir(parents=True, exist_ok=True)

    def source_types(self) -> tuple[str, ...]:
        return tuple(SOURCE_RESOURCES)

    def _scrub(self, text: str, rendered: RenderedConnection) -> str:
        """Remove every sensitive value of ``rendered`` from a message before it is returned to the policy."""
        for value in sorted(rendered.variables.values(), key=len, reverse=True):
            if len(value) >= MIN_SCRUBBED_LENGTH:
                text = text.replace(value, "<secret>")
        return text

    def _check_locators(self, value: Any, key: str = "") -> None:
        """Reject a configuration that points at something that is not in the task's ``config.yaml``."""
        if isinstance(value, Mapping):
            for child_key, child in value.items():
                self._check_locators(child, str(child_key))
        elif isinstance(value, list):
            for child in value:
                self._check_locators(child, key)
        elif isinstance(value, str) and (key.lower() in LOCATOR_KEYS or "://" in value):
            if value.startswith(SECRET_PREFIX):
                return
            if len(value) < 3 or not any(value in allowed for allowed in self._settings.allowed_locators):
                raise ConnectionRejectedError(f"{key} {value!r} is not a source of this task")

    def _resolve_placeholders(self, value: Any, rendered: RenderedConnection) -> Any:
        """Replace every secret placeholder the policy wrote with a reference to a sensitive variable."""
        if isinstance(value, str) and value.startswith(SECRET_PREFIX):
            if value not in self._settings.secrets:
                raise ConnectionRejectedError(f"unknown secret placeholder {value!r}")
            return rendered.reference(self._settings.secrets[value])
        if isinstance(value, Mapping):
            return {key: self._resolve_placeholders(child, rendered) for key, child in value.items()}
        if isinstance(value, list):
            return [self._resolve_placeholders(child, rendered) for child in value]
        return value

    def _externalize_secrets(self, value: Any, rendered: RenderedConnection) -> Any:
        """Replace the secrets of an engine-owned configuration with references to sensitive variables."""
        if isinstance(value, Mapping):
            return {
                key: (
                    rendered.reference(str(child))
                    if _is_secret(key, child)
                    else self._externalize_secrets(child, rendered)
                )
                for key, child in value.items()
            }
        if isinstance(value, list):
            return [self._externalize_secrets(child, rendered) for child in value]
        return value

    def _access_token(self) -> str | None:
        """A fresh application token, or ``None`` when the deployment is used with a username and password.

        Tokens live for a few minutes, so one is requested for every Terraform run and every API call.
        """
        settings = self._settings
        if not settings.client_id:
            return None
        response = httpx.post(
            f"{settings.api_url}/applications/token",
            json={
                "client_id": settings.client_id,
                "client_secret": settings.client_secret,
                "grant-type": "client_credentials",
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["access_token"]

    def _provider(self, rendered: RenderedConnection) -> dict[str, Any]:
        """The provider block: a bearer token when application credentials are configured, else basic auth."""
        settings = self._settings
        if settings.client_id:
            return {"server_url": settings.provider_server_url, "bearer_auth": f"${{var.{TOKEN_VARIABLE}}}"}
        return {
            "server_url": settings.provider_server_url,
            "username": settings.username,
            "password": rendered.reference(settings.password or ""),
        }

    def render_resources(self, connection: ConnectionConfig, namespace: str) -> RenderedConnection:
        """Render one connection into ``namespace`` as Terraform configuration."""
        if connection.source_type not in SOURCE_RESOURCES:
            raise ConnectionRejectedError(
                f"unknown source type {connection.source_type!r}; choose one of: {', '.join(SOURCE_RESOURCES)}"
            )
        self._check_locators(connection.configuration)
        settings = self._settings
        rendered = RenderedConnection(resources={})
        configuration = self._resolve_placeholders(_escape_interpolation(connection.configuration), rendered)
        source_resource = SOURCE_RESOURCES[connection.source_type]
        suffix = uuid.uuid4().hex[:8]
        source: dict[str, Any] = {
            "name": f"elt_rlvr_{connection.name}_{suffix}",
            "workspace_id": settings.workspace_id,
            "configuration": configuration,
        }
        if connection.source_type == "custom_api":
            source["configuration"] = json.dumps(configuration)
            source["definition_id"] = settings.custom_definition_id
        destination = self._externalize_secrets(settings.destination_configuration, rendered)
        destination["schema"] = namespace.upper() if settings.uppercase_namespace else namespace.lower()
        provider = self._provider(rendered)
        if settings.client_id:
            rendered.variables[TOKEN_VARIABLE] = ""
        rendered.resources = {
            "terraform": {
                "required_providers": {"airbyte": {"source": "airbytehq/airbyte", "version": PROVIDER_VERSION}}
            },
            "variable": {name: {"type": "string", "sensitive": True} for name in rendered.variables},
            "provider": {"airbyte": provider},
            "resource": {
                source_resource: {"source": source},
                settings.destination_resource: {
                    "destination": {
                        "name": f"elt_rlvr_destination_{suffix}",
                        "workspace_id": settings.workspace_id,
                        "configuration": destination,
                    }
                },
                "airbyte_connection": {
                    "connection": {
                        "name": f"elt_rlvr_{connection.name}_{suffix}",
                        "source_id": f"${{{source_resource}.source.source_id}}",
                        "destination_id": f"${{{settings.destination_resource}.destination.destination_id}}",
                        "namespace_definition": "destination",
                        "configurations": {
                            "streams": [
                                {"name": _escape_interpolation(stream.name), "sync_mode": stream.sync_mode}
                                for stream in connection.streams
                            ]
                        },
                    }
                },
            },
            "output": {"connection_id": {"value": "${airbyte_connection.connection.connection_id}"}},
        }
        return rendered

    def _terraform(self, directory: Path, rendered: RenderedConnection, *arguments: str) -> subprocess.CompletedProcess:
        if TOKEN_VARIABLE in rendered.variables:
            rendered.variables[TOKEN_VARIABLE] = self._access_token() or ""
        environment = {**os.environ, "TF_PLUGIN_CACHE_DIR": str(self._plugin_cache), "TF_IN_AUTOMATION": "1"}
        environment.update({f"TF_VAR_{name}": value for name, value in rendered.variables.items()})
        return subprocess.run(
            [*self._settings.terraform_command, *arguments],
            cwd=directory,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def _terraform_error(self, result: subprocess.CompletedProcess, rendered: RenderedConnection) -> str:
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        start = next((index for index, line in enumerate(lines) if line.lstrip().startswith("Error:")), 0)
        return self._scrub("\n".join(lines[start : start + MAX_ERROR_LINES]), rendered)

    def _request(self, method: str, path: str, **arguments: Any) -> dict[str, Any]:
        settings = self._settings
        headers = {"accept": "application/json"}
        auth = None
        token = self._access_token()
        if token:
            headers["authorization"] = f"Bearer {token}"
        else:
            auth = (settings.username or "", settings.password or "")
        response = httpx.request(
            method, f"{settings.api_url}{path}", headers=headers, auth=auth, timeout=60, **arguments
        )
        response.raise_for_status()
        return response.json()

    def _run_sync_job(self, connection_id: str) -> str | None:
        """Trigger one sync and wait for it. Returns ``None`` on success, else why the job did not succeed."""
        job = self._request("POST", "/jobs", json={"jobType": "sync", "connectionId": connection_id})
        deadline = time.monotonic() + self._settings.sync_timeout_seconds
        while time.monotonic() < deadline:
            status = self._request("GET", f"/jobs/{job['jobId']}")["status"]
            if status == SUCCEEDED:
                return None
            if status in FAILED_STATES:
                return f"sync job ended with status {status}"
            time.sleep(self._settings.poll_seconds)
        return f"sync job did not finish within {self._settings.sync_timeout_seconds:g} seconds"

    def sync(self, connection: ConnectionConfig, *, warehouse: Warehouse, namespace: str) -> list[StreamResult]:
        rendered = self.render_resources(connection, namespace)
        directory = Path(tempfile.mkdtemp(prefix="elt_rlvr_terraform_"))
        try:
            # One value per line, so that Terraform's error excerpt shows the offending field and nothing else.
            (directory / "main.tf.json").write_text(json.dumps(rendered.resources, indent=2), encoding="utf-8")
            for arguments in (
                ("init", "-input=false", "-no-color"),
                ("apply", "-auto-approve", "-input=false", "-no-color"),
            ):
                result = self._terraform(directory, rendered, *arguments)
                if result.returncode != 0:
                    raise ConnectionRejectedError(self._terraform_error(result, rendered))
            output = self._terraform(directory, rendered, "output", "-json")
            connection_id = json.loads(output.stdout)["connection_id"]["value"]
            failure = self._run_sync_job(connection_id)

            stored = {name.lower(): name for name in warehouse.list_tables(namespace)}
            results = []
            for stream in connection.streams:
                table = stored.get(table_name_for_stream(stream.name).lower())
                if failure is None and table is not None:
                    rows = warehouse.row_count(namespace, table)
                    results.append(StreamResult(connection.name, stream.name, table, succeeded=True, rows_loaded=rows))
                else:
                    error = failure or "the sync job succeeded but did not create the table"
                    results.append(
                        StreamResult(
                            connection.name, stream.name, table_name_for_stream(stream.name), False, error=error
                        )
                    )
            return results
        finally:
            if (directory / "terraform.tfstate").exists():
                self._terraform(directory, rendered, "destroy", "-auto-approve", "-input=false", "-no-color")
            shutil.rmtree(directory, ignore_errors=True)
