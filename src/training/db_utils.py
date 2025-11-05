from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import RowFactory


def resolve_dsn(cli_dsn: str | None = None) -> str:
    dsn = cli_dsn or os.getenv("DATABASE_URL") or os.getenv("NEON_DATABASE_URL")
    if not dsn:
        raise RuntimeError("DATABASE_URL (or NEON_DATABASE_URL) must be set or provided via --dsn.")
    return dsn


def connect(dsn: str | None = None, *, row_factory: RowFactory | None = None, autocommit: bool = False) -> psycopg.Connection:
    resolved = resolve_dsn(dsn)
    return psycopg.connect(resolved, row_factory=row_factory, autocommit=autocommit)


@contextmanager
def connection_ctx(
    dsn: str | None = None, *, row_factory: RowFactory | None = None, autocommit: bool = False
) -> Iterator[psycopg.Connection]:
    conn = connect(dsn, row_factory=row_factory, autocommit=autocommit)
    try:
        yield conn
    finally:
        conn.close()
