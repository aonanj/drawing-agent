#!/usr/bin/env python3
"""
Initialize the Neon/Postgres database schema for drawing-agent.

Usage:
    python -m training.init_db --dsn postgresql://...
or:
    DATABASE_URL=postgresql://... python src/training/init_db.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:  # Allow running as a script or module.
    from training.db_utils import connection_ctx, resolve_dsn
except ImportError:  # pragma: no cover
    from db_utils import connection_ctx, resolve_dsn


DEFAULT_SCHEMA = Path(__file__).with_name("schema.sql")


def run_schema(dsn: str, schema_path: Path) -> None:
    sql = schema_path.read_text(encoding="utf-8")
    with connection_ctx(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialize the Neon/Postgres schema.")
    parser.add_argument("--dsn", help="Postgres connection string (falls back to DATABASE_URL env).")
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help=f"Path to schema SQL file (default: {DEFAULT_SCHEMA})",
    )
    args = parser.parse_args()

    try:
        dsn = resolve_dsn(args.dsn)
    except RuntimeError as exc:  # pragma: no cover - CLI guard
        raise SystemExit(str(exc)) from exc

    run_schema(dsn, args.schema)


if __name__ == "__main__":
    main()
