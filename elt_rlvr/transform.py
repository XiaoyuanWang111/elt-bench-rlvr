"""Run transformation models: vet each SELECT, order the models by dependency, materialize them as tables."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from elt_rlvr.sql_guard import SqlRejectedError, vet_query
from elt_rlvr.warehouse.base import Warehouse, WarehouseError


@dataclass(frozen=True)
class ModelResult:
    """Outcome of materializing one model."""

    name: str
    succeeded: bool
    rows: int = 0
    columns: tuple[str, ...] = ()
    error: str | None = None


def run_models(
    warehouse: Warehouse,
    *,
    models: Mapping[str, str],
    raw_table_namespaces: Mapping[str, str],
    target_namespace: str,
    timeout_seconds: float,
    only: Collection[str] | None = None,
) -> list[ModelResult]:
    """Materialize ``models`` into ``target_namespace``.

    A model may read the raw tables in ``raw_table_namespaces`` and other models. A model that fails, or that
    depends on a failed model, is reported and the rest still run.

    Args:
        warehouse: Destination to run in.
        models: ``{model name: SELECT text}``.
        raw_table_namespaces: Raw tables the models may read, mapped to the namespace each one lives in.
        target_namespace: Namespace the model tables are created in.
        timeout_seconds: Limit for each model's statement.
        only: Model names to run; their dependencies run too. ``None`` runs every model.
    """
    raw_names = {name.lower() for name in raw_table_namespaces}
    table_namespaces = dict(raw_table_namespaces)
    table_namespaces.update({name: target_namespace for name in models if name.lower() not in raw_names})
    model_names = {name.lower(): name for name in models}

    vetted: dict[str, str] = {}
    dependencies: dict[str, list[str]] = {}
    failures: dict[str, str] = {}
    for name, sql in models.items():
        if name.lower() in raw_names:
            failures[name] = f"model name {name!r} is already the name of a source table"
            continue
        readable = {table: namespace for table, namespace in table_namespaces.items() if table.lower() != name.lower()}
        try:
            query = vet_query(sql, dialect=warehouse.dialect, table_namespaces=readable)
        except SqlRejectedError as error:
            failures[name] = str(error)
            continue
        vetted[name] = query.sql
        dependencies[name] = [model_names[table.lower()] for table in query.tables if table.lower() in model_names]

    wanted = set(models) if only is None else {model_names[name.lower()] for name in only if name.lower() in model_names}
    pending = list(wanted)
    while pending:
        for dependency in dependencies.get(pending.pop(), []):
            if dependency not in wanted:
                wanted.add(dependency)
                pending.append(dependency)

    results: dict[str, ModelResult] = {}

    def materialize(name: str, ancestors: tuple[str, ...]) -> ModelResult:
        if name in results:
            return results[name]
        if name in failures:
            results[name] = ModelResult(name, succeeded=False, error=failures[name])
            return results[name]
        if name in ancestors:
            cycle = " -> ".join((*ancestors[ancestors.index(name) :], name))
            return ModelResult(name, succeeded=False, error=f"models depend on each other in a cycle: {cycle}")
        for dependency in dependencies[name]:
            outcome = materialize(dependency, (*ancestors, name))
            if not outcome.succeeded:
                results[name] = ModelResult(name, succeeded=False, error=f"depends on failed model {dependency!r}")
                return results[name]
        try:
            rows = warehouse.create_table_from_query(
                target_namespace, name, vetted[name], timeout_seconds=timeout_seconds
            )
            columns = tuple(column.name for column in warehouse.describe(target_namespace, name))
            results[name] = ModelResult(name, succeeded=True, rows=rows, columns=columns)
        except WarehouseError as error:
            results[name] = ModelResult(name, succeeded=False, error=str(error))
        return results[name]

    return [materialize(name, ()) for name in models if name in wanted]
