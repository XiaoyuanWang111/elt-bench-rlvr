# sqlite source

Reads tables from a relational database.

```yaml
source:
  type: sqlite
  configuration:
    database: sales.sqlite
```

## Schema

### Required

- `database` (String): name of the database to read.

Streams are the tables of the database.
