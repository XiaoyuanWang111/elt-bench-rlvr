# file source

Reads one flat file. Each file needs its own connection.

```yaml
source:
  type: file
  configuration:
    dataset_name: invoices
    format: csv
    url: exports/invoices.csv
    reader_options: '{"sep": "|"}'
```

## Schema

### Required

- `dataset_name` (String): name of the stream this file is exposed as, which is also the destination table.
- `format` (String): must be one of ["csv", "jsonl"].
- `url` (String): location of the file.

### Optional

- `reader_options` (String): a JSON object with options for the reader. Supported key: `sep`, the column separator
  of a csv file. Default: `{"sep": ","}`.

The connection has exactly one stream, named `dataset_name`.
