# Pipeline files

You build the pipeline by writing files under `pipeline/` and running them with tools.

## pipeline/extract_load.yaml

One entry per connection. A connection copies streams (tables) from one source into the destination. Each stream
becomes a raw table with the same name.

```yaml
connections:
  - name: my_connection            # letters, digits, underscore
    source:
      type: <source type>
      configuration:               # the `configuration` block of that source's resource
        <field>: <value>
    streams:
      - name: <stream name>
        sync_mode: full_refresh_append
```

Source types and where their `configuration` fields are documented:

| `type` | documentation |
| --- | --- |
| `postgres` | `documentation/source_postgres.md` |
| `mongodb` | `documentation/source_mongodb_v2.md` |
| `file` | `documentation/source_file.md` (one connection per file) |
| `s3` | `documentation/source_s3.md` |
| `custom_api` | `documentation/source_custom_api.md` |

Those files describe Terraform resources of the Airbyte provider. Write the fields of the resource's
`configuration` block as YAML under `configuration`. Do not write `name`, `workspace_id` or `definition_id`: the
environment fills them in. The destination and its credentials are already configured; you configure only sources
and streams. `documentation/connection.md` describes streams and sync modes.

Take the real values from `config.yaml`. Its key names are not always the connector's field names. The values in
the documentation are examples; do not copy them. A secret appears in `config.yaml` as a placeholder such as
`<secret:postgres.config.password>`: write the placeholder as the value of the matching field and the environment
substitutes the secret when it runs the connection.

`run_extract_load` applies the file and reports, per stream, the rows loaded or the error. A sync takes minutes.
A stream that has already loaded from its connection with the same source configuration and sync mode is left
alone, so after fixing one connection you can call it again. Pass `connections` to sync the named connections
again in full; in an append mode that loads their rows a second time. An episode can run four connection syncs
per source table of the task; `source_tables` in the result is the number of rows each loaded table holds now.

## pipeline/models/<model>.sql

One file per table to build. The file holds a single `SELECT`; the environment materializes its result as a table
named after the file. A model can read the raw tables and other models by their bare names. dbt's
`{{ ref('name') }}` and `{{ source('any', 'name') }}` are accepted and mean the same as the bare name.

`run_transform` runs the model files in dependency order and reports, per model, its row count and columns or the
error. `data_model.yaml` lists the tables you must build and the meaning of every required column. Extra columns
are allowed. Column names must match `data_model.yaml`. `schemas/<table>.csv` describes the columns of each source
table.

## Checking and finishing

`query` runs one read-only `SELECT` over your raw tables and models and returns the first rows.

Call `submit` when the pipeline is complete. The files under `pipeline/` are your submission. The model files are
run again from the raw tables when the pipeline is graded, and they must also give correct results when the source
data changes: with fewer rows, with a row whose related rows in another table are gone (a customer without
orders), and with other key values. A model that fails or returns another number of rows on changed data earns
nothing for its table.

A raw table counts as loaded only if `extract_load.yaml` still describes its stream as it was loaded: after you
rename a connection, change its source configuration or a stream's sync mode, call `run_extract_load` again
before you submit. That syncs the changed streams again, and in an append mode it adds their rows a second time.
