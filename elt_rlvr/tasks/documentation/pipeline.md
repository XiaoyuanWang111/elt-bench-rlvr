# Pipeline files

You build the pipeline by writing files under `pipeline/` and running them with tools.

## pipeline/extract_load.yaml

One entry per connection. A connection copies streams (tables) from one source into the destination. Each stream
becomes a raw table with the same name.

```yaml
connections:
  - name: my_connection            # letters, digits, underscore
    source:
      type: <source type>          # see documentation/source_<type>.md
      configuration:               # fields of that source type
        <field>: <value>
    streams:
      - name: <stream name>
        sync_mode: full_refresh_overwrite
```

Take the real values from `config.yaml`. Its key names are not always the connector's field names; the
`documentation/source_<type>.md` files list the fields each source type accepts. See `documentation/connection.md`
for streams and sync modes.

`run_extract_load` applies the file and reports, per stream, the rows loaded or the error. A stream that has
already loaded from its connection with the same source configuration and sync mode is left alone, so after
fixing one connection you can call it again. Pass `connections` to sync the named connections again in full; in
an append mode that loads their rows a second time. An episode can run four connection syncs per source table of
the task; `source_tables` in the result is the number of rows each loaded table holds now.

## pipeline/models/<model>.sql

One file per table to build. The file holds a single `SELECT`; the environment materializes its result as a table
named after the file. A model can read the raw tables and other models by their bare names. dbt's
`{{ ref('name') }}` and `{{ source('any', 'name') }}` are accepted and mean the same as the bare name.

```sql
-- pipeline/models/customers.sql
select c.customer_id, count(o.order_id) as num_orders
from customers_raw c
left join orders o on o.customer_id = c.customer_id
group by c.customer_id
```

`run_transform` runs the model files in dependency order and reports, per model, its row count and columns or the
error. `data_model.yaml` lists the tables you must build and the meaning of every required column. Extra columns
are allowed. Column names must match `data_model.yaml`.

## Checking and finishing

`query` runs one read-only `SELECT` over your raw tables and models and returns the first rows.

Call `submit` when the pipeline is complete. The files under `pipeline/` are your submission. The model files are
run again from the raw tables when the pipeline is graded, and they must also give correct results when the source
data changes. A raw table counts as loaded only if `extract_load.yaml` still describes its stream as it was loaded:
after you rename a connection, change its source configuration or a stream's sync mode, call `run_extract_load`
again before you submit.
