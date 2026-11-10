"""Official tasks: what the policy may see of an ELT-Bench input bundle, and what the Airbyte engine refuses.

The tests build a small checkout in the layout of ELT-Bench and render Airbyte resources from it. They need neither
Airbyte nor a warehouse.
"""

import dataclasses
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from elt_rlvr.extract_load.airbyte import AirbyteEngine, load_elt_bench_settings
from elt_rlvr.extract_load.base import ConnectionRejectedError
from elt_rlvr.pipeline import ConnectionConfig, StreamConfig
from elt_rlvr.tasks.elt_bench import elt_bench_backend, load_elt_bench_task

POSTGRES_PASSWORD = "pg-secret-4711"
S3_SECRET = "s3-secret-0815"
WAREHOUSE_PASSWORD = "warehouse-secret-2718"
AIRBYTE_PASSWORD = "airbyte-secret-1618"
FILE_URL = "https://drive.google.com/uc?export=download&id=example"
SECRETS = (POSTGRES_PASSWORD, S3_SECRET, WAREHOUSE_PASSWORD, AIRBYTE_PASSWORD)

CONFIG = {
    "Airbyte": {
        "config": {
            "server_url": "http://airbyte-abctl-control-plane:80/api/public/v1/",
            "username": "airbyte",
            "password": AIRBYTE_PASSWORD,
            "workspace_id": "00000000-0000-0000-0000-000000000001",
        }
    },
    "aws_s3": {
        "AWS_ACCESS_KEY_ID": "test",
        "AWS_DEFAULT_REGION": "us-west-2",
        "AWS_ENDPOINT_URL": "http://elt-localstack:4566",
        "AWS_SECRET_ACCESS_KEY": S3_SECRET,
        "data": [
            {"path": "s3://human-resources-bucket/position.csv", "sync_mode": "full_refresh_append", "table": "position"}
        ],
    },
    "flat_files": [{"format": "csv", "path": FILE_URL, "sync_mode": "full_refresh_append", "table": "location"}],
    "postgres": {
        "config": {
            "database": "human_resources",
            "host": "elt-postgres",
            "password": POSTGRES_PASSWORD,
            "port": 5432,
            "schema": "public",
            "sync_mode": "full_refresh_append",
            "tables": ["employee"],
            "user": "postgres",
        }
    },
    "snowflake": {
        "config": {
            "account": "example-account",
            "database": "human_resources",
            "password": WAREHOUSE_PASSWORD,
            "role": "AIRBYTE_ROLE",
            "schema": "AIRBYTE_SCHEMA",
            "username": "AIRBYTE_USER",
            "warehouse": "AIRBYTE_WAREHOUSE",
        }
    },
}
DATA_MODEL = {
    "models": [
        {
            "name": "positions",
            "description": "Each record represents a position",
            "columns": [
                {"name": "position_id", "description": "Unique identifier for the position"},
                {"name": "avg_salary", "description": "Average salary of employees in the position"},
            ],
        }
    ]
}


@pytest.fixture
def elt_bench_root(tmp_path: Path) -> Path:
    root = tmp_path / "ELT-Bench"
    inputs = root / "inputs" / "human_resources"
    (inputs / "schemas").mkdir(parents=True)
    (inputs / "documentation").mkdir()
    (inputs / "config.yaml").write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    (inputs / "data_model.yaml").write_text(yaml.safe_dump(DATA_MODEL), encoding="utf-8")
    (inputs / "snowflake_credential.json").write_text(json.dumps({"password": WAREHOUSE_PASSWORD}), encoding="utf-8")
    (inputs / "schemas" / "employee.csv").write_text(
        "column_name,column_description\nssn,employee id\n", encoding="utf-8"
    )
    for name in ("source_postgres.md", "source_file.md", "connection.md", "destination_snowflake.md", "trigger_job.md"):
        (inputs / "documentation" / name).write_text(f"# {name}\n", encoding="utf-8")
    (root / "evaluation").mkdir()
    (root / "evaluation" / "table.json").write_text(
        json.dumps({"human_resources": {"employee": 25, "location": 8, "position": 4}}), encoding="utf-8"
    )
    (root / "evaluation" / "sort_key.json").write_text(
        json.dumps({"human_resources": {"positions": ["position_id"]}}), encoding="utf-8"
    )
    (root / "setup" / "destination").mkdir(parents=True)
    (root / "setup" / "destination" / "snowflake_credential.json").write_text(
        json.dumps({"account": "example-account", "user": "admin", "password": WAREHOUSE_PASSWORD}), encoding="utf-8"
    )
    return root


@pytest.fixture
def engine(elt_bench_root: Path) -> AirbyteEngine:
    return AirbyteEngine(settings=load_elt_bench_settings(elt_bench_root, "human_resources"))


def _connection(source_type: str, configuration: dict, stream: str = "employee") -> ConnectionConfig:
    return ConnectionConfig("source", source_type, configuration, (StreamConfig(stream, "full_refresh_append"),))


POSTGRES_CONFIGURATION = {
    "host": "elt-postgres",
    "port": 5432,
    "database": "human_resources",
    "username": "postgres",
    "password": "<secret:postgres.config.password>",
    "schemas": ["public"],
}


def test_workspace_of_an_official_task_holds_no_secret(elt_bench_root, tmp_path):
    task = load_elt_bench_task(elt_bench_root, "human_resources", workspace_directory=tmp_path / "workspaces")

    visible = "\n".join(path.read_text(encoding="utf-8") for path in task.workspace_dir.rglob("*") if path.is_file())

    assert all(secret not in visible for secret in SECRETS)
    assert "<secret:postgres.config.password>" in visible
    assert "<secret:aws_s3.AWS_SECRET_ACCESS_KEY>" in visible


def test_workspace_of_an_official_task_leaves_out_airbyte_and_destination_settings(elt_bench_root, tmp_path):
    task = load_elt_bench_task(elt_bench_root, "human_resources", workspace_directory=tmp_path / "workspaces")

    config = yaml.safe_load((task.workspace_dir / "config.yaml").read_text(encoding="utf-8"))
    documents = sorted(path.name for path in (task.workspace_dir / "documentation").iterdir())

    assert set(config) == {"destination", "aws_s3", "flat_files", "postgres"}
    assert config["destination"] == {"warehouse": "snowflake"}
    assert documents == ["connection.md", "pipeline.md", "source_file.md", "source_postgres.md"]
    assert not (task.workspace_dir / "snowflake_credential.json").exists()


def test_official_task_takes_tables_keys_and_row_counts_from_the_benchmark(elt_bench_root, tmp_path):
    task = load_elt_bench_task(elt_bench_root, "human_resources", workspace_directory=tmp_path / "workspaces")

    assert task.source_tables == ("employee", "location", "position")
    assert task.expected_row_counts == {"employee": 25, "location": 8, "position": 4}
    assert task.data_models[0].key == ("position_id",)
    assert task.data_models[0].columns == ("position_id", "avg_salary")


def test_logged_configuration_of_an_official_task_holds_no_secret(elt_bench_root, tmp_path):
    task = load_elt_bench_task(elt_bench_root, "human_resources", workspace_directory=tmp_path / "workspaces")
    backend = elt_bench_backend(elt_bench_root)

    logged = repr(task) + repr(backend)

    assert all(secret not in logged for secret in SECRETS)


def test_rendered_file_names_secrets_only_as_sensitive_variables(engine):
    rendered = engine.render_resources(_connection("postgres", POSTGRES_CONFIGURATION), "rollout_abc")

    resources = rendered.resources
    source = resources["resource"]["airbyte_source_postgres"]["source"]
    destination = resources["resource"]["airbyte_destination_snowflake"]["destination"]["configuration"]
    connection = resources["resource"]["airbyte_connection"]["connection"]
    assert all(secret not in json.dumps(resources) for secret in SECRETS)
    assert sorted(rendered.variables.values()) == sorted([POSTGRES_PASSWORD, WAREHOUSE_PASSWORD, AIRBYTE_PASSWORD])
    assert source["configuration"]["password"] in {f"${{var.{name}}}" for name in rendered.variables}
    assert all(resources["variable"][name] == {"type": "string", "sensitive": True} for name in rendered.variables)
    assert destination["schema"] == "ROLLOUT_ABC"
    assert connection["source_id"] == "${airbyte_source_postgres.source.source_id}"
    assert connection["configurations"]["streams"] == [{"name": "employee", "sync_mode": "full_refresh_append"}]


def test_key_pair_credential_replaces_the_password_user_of_the_destination(elt_bench_root, tmp_path):
    private_key = "-----BEGIN PRIVATE KEY-----\nexample-key-material\n-----END PRIVATE KEY-----\n"
    key_file = tmp_path / "key.p8"
    key_file.write_text(private_key, encoding="utf-8")
    credential = {
        "account": "example-account",
        "user": "SERVICE_USER",
        "role": "SYSADMIN",
        "private_key_file": str(key_file),
    }
    (elt_bench_root / "setup" / "destination" / "snowflake_credential.json").write_text(
        json.dumps(credential), encoding="utf-8"
    )
    engine = AirbyteEngine(settings=load_elt_bench_settings(elt_bench_root, "human_resources"))

    rendered = engine.render_resources(_connection("postgres", POSTGRES_CONFIGURATION), "rollout_abc")

    destination = rendered.resources["resource"]["airbyte_destination_snowflake"]["destination"]["configuration"]
    assert destination["username"] == "SERVICE_USER"
    assert destination["role"] == "SYSADMIN"
    assert set(destination["credentials"]) == {"key_pair_authentication"}
    assert private_key in rendered.variables.values()
    assert WAREHOUSE_PASSWORD not in rendered.variables.values()
    assert "example-key-material" not in json.dumps(rendered.resources)


def test_unknown_secret_placeholder_is_rejected(engine):
    configuration = {**POSTGRES_CONFIGURATION, "password": "<secret:snowflake.config.password>"}

    with pytest.raises(ConnectionRejectedError, match="unknown secret placeholder"):
        engine.render_resources(_connection("postgres", configuration), "rollout_abc")


@pytest.mark.parametrize(
    "url",
    [
        "https://huggingface.co/datasets/tttjjj/elt_bench/resolve/main/gt_snowflake/human_resources/positions.csv",
        "https://example.com/positions.csv",
        "file:///ground_truth/positions.csv",
    ],
)
def test_file_source_cannot_point_at_a_url_outside_the_task(url, engine):
    configuration = {"dataset_name": "positions", "format": "csv", "url": url, "provider": {"https_public_web": {}}}

    with pytest.raises(ConnectionRejectedError, match="not a source of this task"):
        engine.render_resources(_connection("file", configuration, "positions"), "rollout_abc")


def test_database_source_cannot_point_at_another_host(engine):
    configuration = {**POSTGRES_CONFIGURATION, "host": "grader-postgres"}

    with pytest.raises(ConnectionRejectedError, match="not a source of this task"):
        engine.render_resources(_connection("postgres", configuration), "rollout_abc")


def test_sources_named_in_the_task_config_are_accepted(engine):
    file_configuration = {
        "dataset_name": "location",
        "format": "csv",
        "url": FILE_URL,
        "provider": {"https_public_web": {}},
    }
    s3_configuration = {
        "bucket": "human-resources-bucket",
        "endpoint": "http://elt-localstack:4566",
        "aws_access_key_id": "test",
        "aws_secret_access_key": "<secret:aws_s3.AWS_SECRET_ACCESS_KEY>",
        "streams": [{"name": "position", "globs": ["position.csv"], "format": {"csv_format": {}}}],
    }

    engine.render_resources(_connection("file", file_configuration, "location"), "rollout_abc")
    rendered = engine.render_resources(_connection("s3", s3_configuration, "position"), "rollout_abc")

    assert S3_SECRET in rendered.variables.values()


def test_terraform_template_in_a_configuration_string_is_rendered_literally(engine):
    configuration = {**POSTGRES_CONFIGURATION, "jdbc_url_params": '${file("/etc/passwd")}%{if true}x%{endif}'}

    rendered = engine.render_resources(_connection("postgres", configuration), "rollout_abc")

    source = rendered.resources["resource"]["airbyte_source_postgres"]["source"]
    assert source["configuration"]["jdbc_url_params"] == '$${file("/etc/passwd")}%%{if true}x%%{endif}'


def test_error_text_returned_to_the_policy_has_secrets_removed(engine):
    stderr = f"Error: authentication failed for postgres with password {POSTGRES_PASSWORD} and {WAREHOUSE_PASSWORD}\n"
    result = subprocess.CompletedProcess(args=["terraform"], returncode=1, stdout="", stderr=stderr)
    rendered = engine.render_resources(_connection("postgres", POSTGRES_CONFIGURATION), "rollout_abc")

    message = engine._terraform_error(result, rendered)

    assert message == "Error: authentication failed for postgres with password <secret> and <secret>"


def test_unknown_source_type_is_rejected(engine):
    with pytest.raises(ConnectionRejectedError, match="unknown source type"):
        engine.render_resources(_connection("mysql", {"host": "elt-postgres"}), "rollout_abc")


def test_sync_job_that_runs_past_the_deadline_is_cancelled(engine):
    settings = dataclasses.replace(engine._settings, sync_timeout_seconds=0.05, poll_seconds=0.01)
    slow_engine = AirbyteEngine(settings=settings)
    calls = []

    def request(method, path, **arguments):
        calls.append((method, path))
        return {"jobId": 7, "status": "running"}

    slow_engine._request = request

    failure = slow_engine._run_sync_job("connection-id")

    assert failure == "sync job did not finish within 0.05 seconds and was cancelled"
    assert calls[0] == ("POST", "/jobs")
    assert calls[-1] == ("DELETE", "/jobs/7")


def test_sync_job_that_succeeds_is_not_cancelled(engine):
    calls = []

    def request(method, path, **arguments):
        calls.append((method, path))
        return {"jobId": 7, "status": "succeeded"}

    engine._request = request

    assert engine._run_sync_job("connection-id") is None
    assert ("DELETE", "/jobs/7") not in calls
