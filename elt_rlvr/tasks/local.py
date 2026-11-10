"""Three credential-free tasks in the layout of an ELT-Bench task, from easy to hard.

Each task has file-backed sources of several kinds, a ``config.yaml`` with their connection values, a
``data_model.yaml`` that describes the tables to build, per-table column descriptions, connector documentation, and
a grader-side reference pipeline. The source data is synthetic and generated from a seed, so the tasks need no
download and can be made larger with ``scale``.

- ``library``: two CSV files, one model with counts and sums.
- ``storefront``: a relational database, a semicolon-separated CSV file and a document collection with nested
  fields; two models with conditional aggregates, JSON access and a tie-broken ranking.
- ``clinic``: a relational database, a JSON Lines file, a CSV file and a document collection; two models with
  percentages, a busiest-month ranking and a comparison with a group average.
"""

import csv
import json
import random
import shutil
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from elt_rlvr.task import TaskSpec, load_local_task

FIRST_NAMES = ("Ada", "Ben", "Chen", "Dara", "Eli", "Fay", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lea", "Mo", "Nia")
LAST_NAMES = ("Abe", "Brook", "Cruz", "Diaz", "Engel", "Faro", "Gray", "Hale", "Imai", "Jones", "Klein", "Lund")
CITIES = ("Austin", "Boise", "Chicago", "Denver", "Eugene")
BOOKS = ("Dune", "Emma", "Ulysses", "Beloved", "Walden", "Hamlet", "Solaris", "Persuasion")
CATEGORIES = ("audio", "garden", "kitchen", "office", "outdoor")
COUNTRIES = ("CA", "DE", "JP", "MX", "US")
DEPARTMENTS = ("Cardiology", "Dermatology", "Neurology", "Oncology", "Pediatrics", "Radiology")

SourceBuilder = Callable[[Path, random.Random, int], None]


@dataclass(frozen=True)
class LocalTask:
    """Everything needed to write one task to disk."""

    name: str
    build_sources: SourceBuilder
    config: dict[str, Any]
    data_models: list[dict[str, Any]]
    schemas: dict[str, list[tuple[str, str]]]
    extract_load: dict[str, Any]
    models: dict[str, str]
    model_keys: dict[str, list[str]]
    foreign_keys: list[dict[str, str]]


def _write_csv(path: Path, header: list[str], rows: list[list[Any]], separator: str = ",") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=separator, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _write_json_lines(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def _write_sqlite(path: Path, tables: dict[str, tuple[str, list[tuple]]]) -> None:
    connection = sqlite3.connect(path)
    try:
        for table, (columns, rows) in tables.items():
            connection.execute(f"CREATE TABLE {table} ({columns})")
            if rows:
                placeholders = ", ".join("?" for _ in rows[0])
                connection.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
        connection.commit()
    finally:
        connection.close()


def _person(index: int) -> tuple[str, str]:
    """A first and last name; the pair is unique for every index below ``len(FIRST_NAMES) * len(LAST_NAMES)``."""
    return FIRST_NAMES[index % len(FIRST_NAMES)], LAST_NAMES[(index // len(FIRST_NAMES)) % len(LAST_NAMES)]


def _unique_name(index: int) -> str:
    first, last = _person(index)
    generation = index // (len(FIRST_NAMES) * len(LAST_NAMES))
    return f"{first} {last}" if generation == 0 else f"{first} {last} {generation + 1}"


def _column(name: str, description: str) -> dict[str, str]:
    return {"name": name, "description": description}


def _file_connection(table: str, file_format: str, path: str, sync_mode: str, separator: str | None = None) -> dict:
    configuration: dict[str, Any] = {"dataset_name": table, "format": file_format, "url": path}
    if separator:
        configuration["reader_options"] = json.dumps({"sep": separator})
    return {
        "name": f"{table}_file",
        "source": {"type": "file", "configuration": configuration},
        "streams": [{"name": table, "sync_mode": sync_mode}],
    }


# --------------------------------------------------------------------------------------------------------------
# library
# --------------------------------------------------------------------------------------------------------------


def _build_library_sources(sources: Path, rng: random.Random, scale: int) -> None:
    member_count = 40 * scale
    members = [
        [index, _unique_name(index), rng.choice(CITIES), rng.randint(2010, 2024)] for index in range(1, member_count + 1)
    ]
    borrowers = [member[0] for member in members if rng.random() < 0.85]
    loans = [
        [index, rng.choice(borrowers), rng.choice(BOOKS), rng.choice((0, 0, 0, 2, 5, 9, 16, 21, 30))]
        for index in range(1, 3 * member_count + 1)
    ]
    _write_csv(sources / "members.csv", ["member_id", "name", "city", "joined_year"], members)
    _write_csv(sources / "loans.csv", ["loan_id", "member_id", "book_title", "days_late"], loans)


LIBRARY = LocalTask(
    name="library",
    build_sources=_build_library_sources,
    config={
        "destination": {"warehouse": "duckdb"},
        "flat_files": [
            {"table": "members", "format": "csv", "path": "members.csv", "sync_mode": "full_refresh_overwrite"},
            {"table": "loans", "format": "csv", "path": "loans.csv", "sync_mode": "full_refresh_overwrite"},
        ],
    },
    data_models=[
        {
            "name": "member_activity",
            "description": "Each record represents a library member.",
            "columns": [
                _column("member_id", "Unique identifier of the member."),
                _column("name", "Name of the member."),
                _column("num_loans", "The number of loans of the member, replace NULL values with 0."),
                _column(
                    "total_days_late",
                    "The total number of days late over all loans of the member, replace NULL values with 0.",
                ),
                _column(
                    "has_overdue_loan",
                    "Set to 1 if the member has at least one loan returned more than 14 days late, otherwise 0.",
                ),
            ],
        }
    ],
    schemas={
        "members": [
            ("member_id", "unique identifier of the member"),
            ("name", "name of the member"),
            ("city", "city the member lives in"),
            ("joined_year", "year the member joined the library"),
        ],
        "loans": [
            ("loan_id", "unique identifier of the loan"),
            ("member_id", "the member who borrowed the book"),
            ("book_title", "title of the borrowed book"),
            ("days_late", "number of days the book was returned after its due date, 0 if on time"),
        ],
    },
    extract_load={
        "connections": [
            _file_connection("members", "csv", "members.csv", "full_refresh_overwrite"),
            _file_connection("loans", "csv", "loans.csv", "full_refresh_overwrite"),
        ]
    },
    models={
        "member_activity": """
select
    m.member_id,
    m.name,
    count(l.loan_id) as num_loans,
    coalesce(sum(l.days_late), 0) as total_days_late,
    coalesce(max(case when l.days_late > 14 then 1 else 0 end), 0) as has_overdue_loan
from members m
left join loans l on l.member_id = m.member_id
group by m.member_id, m.name
"""
    },
    model_keys={"member_activity": ["member_id"]},
    foreign_keys=[{"table": "loans", "column": "member_id", "references": "members.member_id"}],
)


# --------------------------------------------------------------------------------------------------------------
# storefront
# --------------------------------------------------------------------------------------------------------------


def _build_storefront_sources(sources: Path, rng: random.Random, scale: int) -> None:
    customer_count, product_count = 60 * scale, 20 * scale
    customers = [
        (index, _unique_name(index), rng.choice(COUNTRIES), f"{rng.randint(2019, 2024)}-{rng.randint(1, 12):02d}-15")
        for index in range(1, customer_count + 1)
    ]
    products = [
        [index, f"{rng.choice(CATEGORIES)} item {index}", rng.choice(CATEGORIES), round(rng.uniform(5, 400), 2)]
        for index in range(1, product_count + 1)
    ]
    sold_products = [product for product in products if rng.random() < 0.9]
    orders = []
    for index in range(1, 5 * customer_count + 1):
        product = rng.choice(sold_products)
        orders.append(
            (
                index,
                rng.randint(1, max(1, int(customer_count * 0.9))),
                product[0],
                rng.randint(1, 6),
                round(product[3] * rng.choice((0.8, 0.9, 1.0)), 2),
                rng.choice(("completed", "completed", "completed", "cancelled", "pending")),
            )
        )
    reviews = []
    for index in range(1, 6 * product_count + 1):
        review: dict[str, Any] = {
            "review_id": index,
            "product_id": rng.choice(sold_products)[0],
            "text": rng.choice(("great", "fine", "broke quickly", "as described")),
        }
        if rng.random() < 0.85:
            review["rating"] = {"stars": rng.randint(1, 5), "verified": rng.random() < 0.5}
        reviews.append(review)

    _write_sqlite(
        sources / "store.sqlite",
        {
            "customers": ("customer_id INTEGER PRIMARY KEY, name TEXT, country TEXT, signup_date TEXT", customers),
            "orders": (
                "order_id INTEGER PRIMARY KEY, customer_id INTEGER, product_id INTEGER, quantity INTEGER, "
                "unit_price REAL, status TEXT",
                orders,
            ),
        },
    )
    _write_csv(sources / "products.csv", ["product_id", "product_name", "category", "list_price"], products, ";")
    _write_json_lines(sources / "docs" / "feedback" / "reviews.jsonl", reviews)


STOREFRONT = LocalTask(
    name="storefront",
    build_sources=_build_storefront_sources,
    config={
        "destination": {"warehouse": "duckdb"},
        "relational_database": {
            "config": {
                "file": "store.sqlite",
                "sync_mode": "full_refresh_overwrite",
                "tables": ["customers", "orders"],
            }
        },
        "flat_files": [
            {
                "table": "products",
                "format": "csv",
                "path": "products.csv",
                "separator": ";",
                "sync_mode": "full_refresh_overwrite",
            }
        ],
        "document_store": {
            "config": {
                "address": "docstore://docs",
                "database": "feedback",
                "sync_mode": "full_refresh_overwrite",
                "tables": ["reviews"],
            }
        },
    },
    data_models=[
        {
            "name": "customers_summary",
            "description": "Each record represents a customer.",
            "columns": [
                _column("customer_id", "Unique identifier of the customer."),
                _column("name", "Name of the customer."),
                _column("num_orders", "The number of completed orders of the customer, replace NULL values with 0."),
                _column(
                    "total_spent",
                    "The sum of quantity times unit price over the customer's completed orders, replace NULL values "
                    "with 0.",
                ),
                _column(
                    "has_large_order",
                    "Set to 1 if the customer has a completed order whose quantity times unit price is over 500, "
                    "otherwise 0.",
                ),
            ],
        },
        {
            "name": "products_summary",
            "description": "Each record represents a product.",
            "columns": [
                _column("product_id", "Unique identifier of the product."),
                _column("product_name", "Name of the product."),
                _column(
                    "units_sold", "The total quantity of the product in completed orders, replace NULL values with 0."
                ),
                _column("num_reviews", "The number of reviews of the product, replace NULL values with 0."),
                _column("avg_stars", "The average number of stars over the product's reviews that have a rating."),
                _column(
                    "top_buyer",
                    "The name of the customer who bought the largest total quantity of the product in completed "
                    "orders, with ties broken by the ascending order of the customer name.",
                ),
            ],
        },
    ],
    schemas={
        "customers": [
            ("customer_id", "unique identifier of the customer"),
            ("name", "name of the customer"),
            ("country", "two-letter country code"),
            ("signup_date", "date the customer signed up, YYYY-MM-DD"),
        ],
        "orders": [
            ("order_id", "unique identifier of the order"),
            ("customer_id", "the customer who placed the order"),
            ("product_id", "the ordered product"),
            ("quantity", "number of units ordered"),
            ("unit_price", "price paid per unit"),
            ("status", "one of completed, cancelled, pending"),
        ],
        "products": [
            ("product_id", "unique identifier of the product"),
            ("product_name", "name of the product"),
            ("category", "category of the product"),
            ("list_price", "list price of the product"),
        ],
        "reviews": [
            ("review_id", "unique identifier of the review"),
            ("product_id", "the reviewed product"),
            ("text", "text of the review"),
            (
                "rating",
                "object with the fields stars (1 to 5) and verified (boolean); absent when the review has no rating",
            ),
        ],
    },
    extract_load={
        "connections": [
            {
                "name": "store_database",
                "source": {"type": "sqlite", "configuration": {"database": "store.sqlite"}},
                "streams": [
                    {"name": "customers", "sync_mode": "full_refresh_overwrite"},
                    {"name": "orders", "sync_mode": "full_refresh_overwrite"},
                ],
            },
            _file_connection("products", "csv", "products.csv", "full_refresh_overwrite", separator=";"),
            {
                "name": "feedback_documents",
                "source": {
                    "type": "documents",
                    "configuration": {"connection_string": "docstore://docs", "database": "feedback"},
                },
                "streams": [{"name": "reviews", "sync_mode": "full_refresh_overwrite"}],
            },
        ]
    },
    models={
        "customers_summary": """
select
    c.customer_id,
    c.name,
    count(o.order_id) as num_orders,
    coalesce(sum(o.quantity * o.unit_price), 0) as total_spent,
    coalesce(max(case when o.quantity * o.unit_price > 500 then 1 else 0 end), 0) as has_large_order
from customers c
left join orders o on o.customer_id = c.customer_id and o.status = 'completed'
group by c.customer_id, c.name
""",
        "products_summary": """
with sold as (
    select product_id, customer_id, sum(quantity) as units
    from orders
    where status = 'completed'
    group by product_id, customer_id
),
ranked as (
    select
        s.product_id,
        c.name,
        row_number() over (partition by s.product_id order by s.units desc, c.name asc) as position
    from sold s
    join customers c on c.customer_id = s.customer_id
),
units as (
    select product_id, sum(units) as units_sold from sold group by product_id
),
review_stats as (
    select
        product_id,
        count(*) as num_reviews,
        avg(cast(json_extract_string(rating, '$.stars') as double)) as avg_stars
    from reviews
    group by product_id
)
select
    p.product_id,
    p.product_name,
    coalesce(u.units_sold, 0) as units_sold,
    coalesce(r.num_reviews, 0) as num_reviews,
    r.avg_stars,
    t.name as top_buyer
from products p
left join units u on u.product_id = p.product_id
left join review_stats r on r.product_id = p.product_id
left join ranked t on t.product_id = p.product_id and t.position = 1
""",
    },
    model_keys={"customers_summary": ["customer_id"], "products_summary": ["product_id"]},
    foreign_keys=[
        {"table": "orders", "column": "customer_id", "references": "customers.customer_id"},
        {"table": "orders", "column": "product_id", "references": "products.product_id"},
        {"table": "reviews", "column": "product_id", "references": "products.product_id"},
    ],
)


# --------------------------------------------------------------------------------------------------------------
# clinic
# --------------------------------------------------------------------------------------------------------------


def _build_clinic_sources(sources: Path, rng: random.Random, scale: int) -> None:
    doctor_count, patient_count = 12 * scale, 80 * scale
    departments = [[index, name, rng.randint(1, 5)] for index, name in enumerate(DEPARTMENTS, start=1)]
    doctors = []
    for index in range(1, doctor_count + 1):
        first, last = _person(index + 3)
        doctors.append(
            {
                "doctor_id": index,
                "first_name": first,
                "last_name": f"{last}{index}",
                "department_id": rng.randint(1, len(DEPARTMENTS) - 1),
                "hired_year": rng.randint(2005, 2023),
            }
        )
    patients = []
    for index in range(1, patient_count + 1):
        first, last = _person(index)
        patients.append((index, first, last, rng.randint(1940, 2020), rng.choice(("F", "M"))))
    busy_doctors = [doctor["doctor_id"] for doctor in doctors if rng.random() < 0.85] or [1]
    visits = [
        (
            index,
            rng.randint(1, patient_count),
            rng.choice(busy_doctors),
            f"2025-{rng.randint(1, 6):02d}-{rng.randint(1, 28):02d}",
            rng.choice(("new", "follow_up", "follow_up")),
            rng.choice((80, 120, 150, 200)),
        )
        for index in range(1, 8 * patient_count + 1)
    ]
    lab_results = []
    for index in range(1, 3 * patient_count + 1):
        values: dict[str, float] = {"cholesterol": round(rng.uniform(140, 260), 1)}
        if rng.random() < 0.8:
            values["glucose"] = round(rng.uniform(70, 180), 1)
        lab_results.append(
            {"result_id": index, "visit_id": rng.choice(visits)[0], "panel": "metabolic", "values": values}
        )

    _write_sqlite(
        sources / "clinic.sqlite",
        {
            "patients": (
                "patient_id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, birth_year INTEGER, sex TEXT",
                patients,
            ),
            "visits": (
                "visit_id INTEGER PRIMARY KEY, patient_id INTEGER, doctor_id INTEGER, visit_date TEXT, kind TEXT, "
                "fee INTEGER",
                visits,
            ),
        },
    )
    _write_json_lines(sources / "staff" / "doctors.jsonl", doctors)
    _write_csv(sources / "departments.csv", ["department_id", "department_name", "floor"], departments, ";")
    _write_json_lines(sources / "labstore" / "lab" / "lab_results.jsonl", lab_results)


CLINIC = LocalTask(
    name="clinic",
    build_sources=_build_clinic_sources,
    config={
        "destination": {"warehouse": "duckdb"},
        "relational_database": {
            "config": {
                "file": "clinic.sqlite",
                "sync_mode": "full_refresh_overwrite",
                "tables": ["patients", "visits"],
            }
        },
        "flat_files": [
            {
                "table": "doctors",
                "format": "jsonl",
                "path": "staff/doctors.jsonl",
                "sync_mode": "full_refresh_overwrite",
            },
            {
                "table": "departments",
                "format": "csv",
                "path": "departments.csv",
                "separator": ";",
                "sync_mode": "full_refresh_overwrite",
            },
        ],
        "document_store": {
            "config": {
                "address": "docstore://labstore",
                "database": "lab",
                "sync_mode": "full_refresh_overwrite",
                "tables": ["lab_results"],
            }
        },
    },
    data_models=[
        {
            "name": "doctors_summary",
            "description": "Each record represents a doctor.",
            "columns": [
                _column("doctor_id", "Unique identifier of the doctor."),
                _column("full_name", "The first name and the last name of the doctor, separated by a space."),
                _column("department_name", "Name of the doctor's department."),
                _column("num_visits", "The number of visits to the doctor, replace NULL values with 0."),
                _column(
                    "pct_follow_up",
                    "The percentage (0 to 100) of the doctor's visits that are follow-up visits. NULL if the doctor "
                    "has no visits.",
                ),
                _column(
                    "busiest_month",
                    "The month (YYYY-MM) in which the doctor had the most visits, with ties broken by the earliest "
                    "month. NULL if the doctor has no visits.",
                ),
                _column(
                    "above_department_average",
                    "Set to 1 if the doctor's number of visits is greater than the average number of visits per "
                    "doctor in the doctor's department, otherwise 0.",
                ),
            ],
        },
        {
            "name": "departments_summary",
            "description": "Each record represents a department.",
            "columns": [
                _column("department_id", "Unique identifier of the department."),
                _column("department_name", "Name of the department."),
                _column("num_doctors", "The number of doctors in the department, replace NULL values with 0."),
                _column(
                    "num_patients",
                    "The number of distinct patients who visited a doctor of the department, replace NULL values "
                    "with 0.",
                ),
                _column(
                    "total_fees",
                    "The sum of the fees of visits to doctors of the department, replace NULL values with 0.",
                ),
                _column(
                    "avg_glucose",
                    "The average glucose value over lab results of visits to doctors of the department. NULL if "
                    "there is no such value.",
                ),
            ],
        },
    ],
    schemas={
        "patients": [
            ("patient_id", "unique identifier of the patient"),
            ("first_name", "first name of the patient"),
            ("last_name", "last name of the patient"),
            ("birth_year", "year of birth"),
            ("sex", "F or M"),
        ],
        "visits": [
            ("visit_id", "unique identifier of the visit"),
            ("patient_id", "the patient who was seen"),
            ("doctor_id", "the doctor who saw the patient"),
            ("visit_date", "date of the visit, YYYY-MM-DD"),
            ("kind", "new for a first visit, follow_up for a follow-up visit"),
            ("fee", "fee charged for the visit"),
        ],
        "doctors": [
            ("doctor_id", "unique identifier of the doctor"),
            ("first_name", "first name of the doctor"),
            ("last_name", "last name of the doctor"),
            ("department_id", "the department the doctor works in"),
            ("hired_year", "year the doctor was hired"),
        ],
        "departments": [
            ("department_id", "unique identifier of the department"),
            ("department_name", "name of the department"),
            ("floor", "floor of the building the department is on"),
        ],
        "lab_results": [
            ("result_id", "unique identifier of the lab result"),
            ("visit_id", "the visit the lab work was ordered in"),
            ("panel", "name of the lab panel"),
            ("values", "object with the measured values; fields cholesterol and, when measured, glucose"),
        ],
    },
    extract_load={
        "connections": [
            {
                "name": "clinic_database",
                "source": {"type": "sqlite", "configuration": {"database": "clinic.sqlite"}},
                "streams": [
                    {"name": "patients", "sync_mode": "full_refresh_overwrite"},
                    {"name": "visits", "sync_mode": "full_refresh_overwrite"},
                ],
            },
            _file_connection("doctors", "jsonl", "staff/doctors.jsonl", "full_refresh_overwrite"),
            _file_connection("departments", "csv", "departments.csv", "full_refresh_overwrite", separator=";"),
            {
                "name": "lab_documents",
                "source": {
                    "type": "documents",
                    "configuration": {"connection_string": "docstore://labstore", "database": "lab"},
                },
                "streams": [{"name": "lab_results", "sync_mode": "full_refresh_overwrite"}],
            },
        ]
    },
    models={
        "doctor_visits": """
select
    d.doctor_id,
    d.department_id,
    count(v.visit_id) as num_visits,
    coalesce(sum(case when v.kind = 'follow_up' then 1 else 0 end), 0) as num_follow_up
from doctors d
left join visits v on v.doctor_id = d.doctor_id
group by d.doctor_id, d.department_id
""",
        "doctors_summary": """
with monthly as (
    select doctor_id, substr(visit_date, 1, 7) as visit_month, count(*) as visits_in_month
    from visits
    group by doctor_id, substr(visit_date, 1, 7)
),
busiest as (
    select
        doctor_id,
        visit_month,
        row_number() over (partition by doctor_id order by visits_in_month desc, visit_month asc) as position
    from monthly
),
department_average as (
    select department_id, avg(num_visits) as average_visits from {{ ref('doctor_visits') }} group by department_id
)
select
    d.doctor_id,
    d.first_name || ' ' || d.last_name as full_name,
    p.department_name,
    c.num_visits,
    case when c.num_visits = 0 then null else 100.0 * c.num_follow_up / c.num_visits end as pct_follow_up,
    b.visit_month as busiest_month,
    case when c.num_visits > a.average_visits then 1 else 0 end as above_department_average
from doctors d
join {{ ref('doctor_visits') }} c on c.doctor_id = d.doctor_id
join department_average a on a.department_id = d.department_id
left join departments p on p.department_id = d.department_id
left join busiest b on b.doctor_id = d.doctor_id and b.position = 1
""",
        "departments_summary": """
with department_visits as (
    select d.department_id, v.visit_id, v.patient_id, v.fee
    from doctors d
    join visits v on v.doctor_id = d.doctor_id
),
glucose as (
    select
        dv.department_id,
        avg(cast(json_extract_string(l."values", '$.glucose') as double)) as avg_glucose
    from lab_results l
    join department_visits dv on dv.visit_id = l.visit_id
    group by dv.department_id
),
staff as (
    select department_id, count(*) as num_doctors from doctors group by department_id
),
activity as (
    select department_id, count(distinct patient_id) as num_patients, sum(fee) as total_fees
    from department_visits
    group by department_id
)
select
    p.department_id,
    p.department_name,
    coalesce(s.num_doctors, 0) as num_doctors,
    coalesce(a.num_patients, 0) as num_patients,
    coalesce(a.total_fees, 0) as total_fees,
    g.avg_glucose
from departments p
left join staff s on s.department_id = p.department_id
left join activity a on a.department_id = p.department_id
left join glucose g on g.department_id = p.department_id
""",
    },
    model_keys={"doctors_summary": ["doctor_id"], "departments_summary": ["department_id"]},
    foreign_keys=[
        {"table": "doctors", "column": "department_id", "references": "departments.department_id"},
        {"table": "visits", "column": "doctor_id", "references": "doctors.doctor_id"},
        {"table": "visits", "column": "patient_id", "references": "patients.patient_id"},
        {"table": "lab_results", "column": "visit_id", "references": "visits.visit_id"},
    ],
)

LOCAL_TASKS: dict[str, LocalTask] = {task.name: task for task in (LIBRARY, STOREFRONT, CLINIC)}


def write_local_task(task: LocalTask, root: Path, *, seed: int = 0, scale: int = 1) -> Path:
    """Write ``task`` under ``root/<name>`` (replacing it) and return that directory."""
    task_dir = root / task.name
    shutil.rmtree(task_dir, ignore_errors=True)
    workspace, sources, grader = task_dir / "workspace", task_dir / "sources", task_dir / "grader"
    for directory in (workspace / "schemas", workspace / "documentation", sources, grader / "models"):
        directory.mkdir(parents=True)

    task.build_sources(sources, random.Random(f"{task.name}-{seed}"), scale)

    (workspace / "config.yaml").write_text(yaml.safe_dump(task.config, sort_keys=False), encoding="utf-8", newline="\n")
    (workspace / "data_model.yaml").write_text(
        yaml.safe_dump({"models": task.data_models}, sort_keys=False, width=1000), encoding="utf-8", newline="\n"
    )
    for table, columns in task.schemas.items():
        _write_csv(
            workspace / "schemas" / f"{table}.csv", ["column_name", "column_description"], [list(c) for c in columns]
        )
    documentation = resources.files("elt_rlvr.tasks") / "documentation"
    for document in documentation.iterdir():
        (workspace / "documentation" / document.name).write_text(
            document.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
        )

    (grader / "extract_load.yaml").write_text(
        yaml.safe_dump(task.extract_load, sort_keys=False), encoding="utf-8", newline="\n"
    )
    for model, sql in task.models.items():
        (grader / "models" / f"{model}.sql").write_text(sql.strip() + "\n", encoding="utf-8", newline="\n")
    settings = {
        "source_tables": list(task.schemas),
        "model_keys": task.model_keys,
        "foreign_keys": task.foreign_keys,
    }
    (grader / "task.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8", newline="\n")
    return task_dir


def build_local_tasks(root: Path, names: list[str] | None = None, *, seed: int = 0, scale: int = 1) -> list[TaskSpec]:
    """Write the named local tasks (all of them by default) under ``root`` and load them."""
    unknown = sorted(set(names or []) - set(LOCAL_TASKS))
    if unknown:
        raise ValueError(f"unknown local task(s): {', '.join(unknown)}; choose from: {', '.join(LOCAL_TASKS)}")
    return [
        load_local_task(write_local_task(LOCAL_TASKS[name], root, seed=seed, scale=scale))
        for name in (names or list(LOCAL_TASKS))
    ]
