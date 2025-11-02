import json
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

def doc_id_from_name(name: str) -> str:
    s = Path(name).stem.upper()
    return re.sub(r'[^A-Z0-9]', '', s)

def ensure_db(db_path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(db_path))
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS docs(
        doc_id TEXT PRIMARY KEY,
        xml    TEXT NOT NULL,
        tiffs  TEXT NOT NULL
    )""")
    return db

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

def index_zips(raw_root: Path, work_root: Path, db_path: Path, force: bool=False) -> None:
    db = ensure_db(db_path)
    cur = db.cursor()

    zips = list(raw_root.rglob("*.zip")) + list(raw_root.rglob("*.ZIP"))
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
                    # already extracted and indexed? check db
                    row = cur.execute("SELECT 1 FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
                    if row:
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
                    "REPLACE INTO docs(doc_id, xml, tiffs) VALUES(?,?,?)",
                    (doc_id, xml_path, json.dumps(tiffs))
                )
        except zipfile.BadZipFile:
            continue
    db.commit()
    db.close()

if __name__ == "__main__":
    # Usage:
    # python src/training/index_docs.py data/raw data/work data/work/index.sqlite
    raw = Path(sys.argv[1])   # directory holding .zip files
    work = Path(sys.argv[2])  # working dir (will create extracted/{doc_id})
    dbp  = Path(sys.argv[3])
    force = bool(len(sys.argv) > 4 and sys.argv[4] == "--force")
    index_zips(raw, work, dbp, force)
