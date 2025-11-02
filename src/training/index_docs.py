
import sqlite3
import pathlib
import re
import sys
import json

root = pathlib.Path(sys.argv[1])
db = sqlite3.connect(sys.argv[2])
c = db.cursor()
c.execute("""CREATE TABLE IF NOT EXISTS docs(
 doc_id TEXT PRIMARY KEY, xml TEXT, tiffs TEXT)""")

def doc_id_from_name(p):
    s = p.stem
    return re.sub(r'[^A-Z0-9]', '', s.upper())

xmls = list(root.rglob("*.XML")) or list(root.rglob("*.xml"))
for x in xmls:
    doc = doc_id_from_name(x)
    glob = x.parent.glob("*.tif*") or x.parent.glob("*.TIF*")
    tiffs = [str(p) for p in glob]
    c.execute("REPLACE INTO docs VALUES (?,?,?)",
              (doc, str(x), json.dumps(tiffs)))
db.commit()
