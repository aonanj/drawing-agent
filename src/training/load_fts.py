#!/usr/bin/env python3
import argparse

from parse_xml import parse_doc

try:  # Allow running as script or module.
    from training.db_utils import connection_ctx
except ImportError:  # pragma: no cover
    from db_utils import connection_ctx


def populate_text_index(dsn: str | None) -> None:
    with connection_ctx(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT doc_id, xml FROM docs")
            docs = cur.fetchall()
            for doc_id, xml in docs:
                meta = parse_doc(xml)
                rows = [
                    (doc_id, "caption", "\n".join(meta.get("titles", []))),
                    (doc_id, "paragraph", "\n".join(meta.get("figure_paras", []))),
                    (doc_id, "claim", meta.get("claims", "") or ""),
                ]
                cur.executemany(
                    """
                    INSERT INTO text_fts (doc_id, section, content)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (doc_id, section)
                    DO UPDATE SET content = EXCLUDED.content
                    """,
                    rows,
                )
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate Postgres text_fts table from XML metadata.")
    parser.add_argument("--dsn", help="Postgres connection string; falls back to DATABASE_URL env variable.")
    args = parser.parse_args()
    try:
        populate_text_index(args.dsn)
    except RuntimeError as exc:  # pragma: no cover - CLI guard
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
