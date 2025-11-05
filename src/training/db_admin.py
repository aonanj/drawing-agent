#!/usr/bin/env python3
"""
Administrative helpers for the Neon/Postgres database.
"""
from __future__ import annotations

import argparse
import sys
from typing import Iterable, Sequence

try:  # Allow running as module or script.
    from training.db_utils import connection_ctx
except ImportError:  # pragma: no cover
    from db_utils import connection_ctx


def print_table(rows: Iterable[Sequence[object]]) -> None:
    for row in rows:
        printable = "\t".join("" if v is None else str(v) for v in row)
        print(printable)


def cmd_info(dsn: str | None) -> None:
    with connection_ctx(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database(), current_user, version()")
            db_name, user, version = cur.fetchone()
            print(f"database: {db_name}")
            print(f"user: {user}")
            print(f"version: {version}")

            cur.execute("SHOW timezone")
            print(f"timezone: {cur.fetchone()[0]}")

            cur.execute("SHOW synchronous_commit")
            print(f"synchronous_commit: {cur.fetchone()[0]}")

            cur.execute("SHOW default_transaction_isolation")
            print(f"default_isolation: {cur.fetchone()[0]}")


def cmd_counts(dsn: str | None) -> None:
    with connection_ctx(dsn) as conn:
        with conn.cursor() as cur:
            tables = ["docs", "figures", "text_fts"]
            for table in tables:
                cur.execute(f"SELECT '{table}' AS table, COUNT(*) FROM {table}")
                name, count = cur.fetchone()
                print(f"{name}\t{count}")


def cmd_vacuum(dsn: str | None) -> None:
    with connection_ctx(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("VACUUM (ANALYZE)")
    print("VACUUM completed")


def cmd_integrity(dsn: str | None) -> None:
    with connection_ctx(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT relname, n_live_tup, n_dead_tup
                FROM pg_stat_user_tables
                ORDER BY relname
                """
            )
            print_table(cur.fetchall())


def cmd_clear(dsn: str | None) -> None:
    with connection_ctx(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE text_fts")
            cur.execute("TRUNCATE TABLE figures")
            cur.execute("TRUNCATE TABLE docs")
    print("Cleared docs, figures, and text_fts tables.")


COMMANDS = {
    "info": cmd_info,
    "counts": cmd_counts,
    "vacuum": cmd_vacuum,
    "integrity": cmd_integrity,
    "clear": cmd_clear,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Database admin commands for Neon/Postgres.")
    parser.add_argument("command", choices=sorted(COMMANDS.keys()), help="Command to execute")
    parser.add_argument("--dsn", help="Postgres connection string; falls back to DATABASE_URL env variable.")
    args = parser.parse_args(argv)

    handler = COMMANDS[args.command]
    try:
        handler(args.dsn)
    except RuntimeError as exc:  # pragma: no cover - CLI guard
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main(sys.argv[1:])
