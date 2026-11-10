# Connections, streams and sync modes

A connection has one source and a list of streams. A stream is one table, collection or file of the source.

```yaml
streams:
  - name: orders
    sync_mode: full_refresh_append
```

## Fields

- `name` (String, required): the stream to copy. It must exist in the source.
- `sync_mode` (String, required): one of
  - `full_refresh_overwrite`: replace the destination table with the current contents of the stream.
  - `full_refresh_append`: add the current contents of the stream to the destination table. Running the same
    connection twice in this mode loads every row twice.
  - `incremental_append`, `incremental_deduped_history`: need a cursor field; the sources in this workspace do not
    support them.

## Result

Each stream is loaded into the destination table of the same name. The loader adds the columns `_airbyte_raw_id`
and `_airbyte_extracted_at` to every raw table.

List only the streams named in `config.yaml`.
