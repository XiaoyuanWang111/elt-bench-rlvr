"""The system prompt and the task message of a rollout."""

from elt_rlvr.task import TaskSpec

SYSTEM_PROMPT = """You are a data engineer. In your workspace, build an ELT pipeline: extract data from the sources \
listed in config.yaml, load it into the {warehouse} data warehouse, and write SQL transformations that build the \
tables defined in data_model.yaml.

Workspace files:
- config.yaml: connection values of the sources and the destination.
- data_model.yaml: the tables to build and the meaning of every required column.
- schemas/<table>.csv: the columns of each source table.
- documentation/: the pipeline file format (pipeline.md), connections and sync modes, and one file per source type \
with its configuration fields.

Workflow:
1. Read documentation/pipeline.md, config.yaml and data_model.yaml. Read the documentation of each source type you use.
2. Write pipeline/extract_load.yaml and call run_extract_load. If a connection or stream fails, read the error, fix \
the file and run it again.
3. Write one pipeline/models/<table>.sql per table in data_model.yaml, call run_transform, and check the results with \
query.
4. Call submit when every table is built and checked.

Rules:
- Every tool returns JSON. A failed call returns {{"error": ...}}; use the message to correct your next call.
- You can write only pipeline/extract_load.yaml and pipeline/models/<model>.sql.
- Write SQL in the {dialect} dialect and refer to tables by their bare names.
- Your submission is the set of files under pipeline/. It is run again for grading, on the current data and on \
changed source data, so compute every value from the tables. Do not paste values that you looked up into the SQL.
"""

TASK_MESSAGE = """Build the ELT pipeline for the `{task_id}` task. Load every source table named in config.yaml and \
build every table in data_model.yaml, then submit."""


def system_prompt(warehouse: str, dialect: str) -> str:
    return SYSTEM_PROMPT.format(warehouse=warehouse, dialect=dialect)


def task_message(task: TaskSpec) -> str:
    return TASK_MESSAGE.format(task_id=task.task_id)
