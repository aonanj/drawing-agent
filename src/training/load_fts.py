import sqlite3
from parse_xml import parse_doc

db = sqlite3.connect("data/work/index.sqlite")
db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS text_fts USING fts5(doc_id, section, content)")
for (doc_id, xml, _) in db.execute("SELECT doc_id, xml, tiffs FROM docs"):
    meta = parse_doc(xml)
    rows = [
        (doc_id, "caption", "\n".join(meta.get("titles", []))),
        (doc_id, "paragraph", "\n".join(meta.get("figure_paras", []))),
        (doc_id, "claim", meta.get("claims","")),
    ]
    db.executemany("INSERT INTO text_fts(doc_id,section,content) VALUES(?,?,?)", rows)
db.commit()
