# documents source

Reads collections from a document store.

```yaml
source:
  type: documents
  configuration:
    connection_string: docstore://cluster0
    database: app
```

## Schema

### Required

- `connection_string` (String): address of the document store, starting with `docstore://`.
- `database` (String): the database that holds the collections.

Streams are the collections of the database.

## Loaded shape

Every top-level field of a document becomes a column. A document that lacks a field has NULL in that column. A
field whose value is an object or an array is loaded as JSON text; read inside it with the warehouse's JSON
functions, for example `json_extract_string(address, '$.city')` in DuckDB.
