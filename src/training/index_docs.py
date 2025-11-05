import argparse
import re
import zipfile
from pathlib import Path

from psycopg.rows import tuple_row
from psycopg.types.json import Json

try:  # Allow running via module or direct script.
    from training.db_utils import connection_ctx, resolve_dsn
except ImportError:  # pragma: no cover
    from db_utils import connection_ctx, resolve_dsn


def doc_id_from_name(name: str) -> str:
    s = Path(name).stem.upper()
    return re.sub(r'[^A-Z0-9]', '', s)

def extract_bundle(zp: zipfile.ZipFile, xml_member: zipfile.ZipInfo, out_dir: Path) -> tuple[str, list[str]]:
    """
    Extract exactly one XML and all TIFFs from the zip into out_dir.
    Returns (xml_path, [tiff_paths]).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    # Extract XML
    xml_bytes = zp.read(xml_member)
    xml_path = out_dir / Path(xml_member.filename).name
    xml_path.write_bytes(xml_bytes)

    # Extract TIFFs
    tiff_paths = []
    for m in zp.infolist():
        name = m.filename
        if name.lower().endswith((".tif", ".tiff")):
            data = zp.read(m)
            p = out_dir / Path(name).name
            p.write_bytes(data)
            tiff_paths.append(str(p))
    return str(xml_path), tiff_paths

def index_zips(raw_root: Path, work_root: Path, dsn: str | None, force: bool=False) -> None:
    dsn_resolved = resolve_dsn(dsn)
    zips = list(raw_root.rglob("*.zip")) + list(raw_root.rglob("*.ZIP"))
    with connection_ctx(dsn_resolved, row_factory=tuple_row) as conn:
        with conn.cursor() as cur:
            for z in zips:
                try:
                    with zipfile.ZipFile(z) as zp:
                        members = zp.infolist()
                        # Skip bundles that contain chemical drawing files
                        if any(m.filename.lower().endswith((".cdx", ".mol")) for m in members):
                            continue
                        # find XMLs
                        xml_members = [m for m in members if m.filename.lower().endswith(".xml")]
                        if len(xml_members) != 1:
                            # ignore zips with 0 or >1 XML
                            continue
                        xml_member = xml_members[0]
                        doc_id = doc_id_from_name(xml_member.filename)

                        out_dir = work_root / "extracted" / doc_id
                        if out_dir.exists() and not force:
                            cur.execute("SELECT 1 FROM docs WHERE doc_id = %s", (doc_id,))
                            if cur.fetchone():
                                continue  # skip
                        # fresh extract
                        if out_dir.exists() and force:
                            for p in out_dir.iterdir():
                                p.unlink()
                        out_dir.mkdir(parents=True, exist_ok=True)

                        xml_path, tiffs = extract_bundle(zp, xml_member, out_dir)
                        if not tiffs:
                            # keep record anyway? choice: skip if no tiffs
                            # skip: unusable for drawings
                            # clean dir
                            for p in out_dir.iterdir():
                                p.unlink()
                            out_dir.rmdir()
                            continue

                        cur.execute(
                            """
                            INSERT INTO docs (doc_id, xml, tiffs)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (doc_id)
                            DO UPDATE SET xml = EXCLUDED.xml, tiffs = EXCLUDED.tiffs
                            """,
                            (doc_id, str(xml_path), Json(tiffs)),
                        )
                except zipfile.BadZipFile:
                    continue
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Index ZIP bundles into the Neon/Postgres database.")
    parser.add_argument("raw_root", type=Path, help="Directory holding .zip files")
    parser.add_argument("work_root", type=Path, help="Working directory (will create extracted/{doc_id})")
    parser.add_argument("--dsn", help="Postgres connection string (falls back to DATABASE_URL env).")
    parser.add_argument("--force", action="store_true", help="Re-extract and overwrite existing records.")
    args = parser.parse_args()

    try:
        index_zips(args.raw_root, args.work_root, args.dsn, args.force)
    except RuntimeError as exc:  # pragma: no cover - CLI guard
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
