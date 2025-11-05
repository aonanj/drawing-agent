-- Postgres schema for the drawing-agent index database.

-- Core index of documents (one XML + many TIFFs)
CREATE TABLE IF NOT EXISTS docs (
  doc_id TEXT PRIMARY KEY,          -- e.g., US20230001234A1 or US1234567B2
  xml     TEXT NOT NULL,            -- absolute or project-relative path to the XML
  tiffs   JSONB NOT NULL            -- JSON array of absolute or project-relative TIFF paths
);

CREATE INDEX IF NOT EXISTS idx_docs_xml ON docs(xml);

-- Optional per-figure registry (filled later if you persist crops/bboxes)
CREATE TABLE IF NOT EXISTS figures (
  id        TEXT PRIMARY KEY,       -- stable id, e.g., {doc_id}:{page_or_file}:{fig_label}
  doc_id    TEXT NOT NULL,
  tiff_path TEXT NOT NULL,
  fig_label TEXT,                   -- e.g., "FIG. 3A"
  bbox_x1   INTEGER,                -- original page coords
  bbox_y1   INTEGER,
  bbox_x2   INTEGER,
  bbox_y2   INTEGER,
  FOREIGN KEY(doc_id) REFERENCES docs(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_fig_doc   ON figures(doc_id);
CREATE INDEX IF NOT EXISTS idx_fig_label ON figures(fig_label);

-- Optional full-text search over extracted text
CREATE TABLE IF NOT EXISTS text_fts (
  doc_id TEXT NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
  section TEXT NOT NULL,            -- 'caption' | 'paragraph' | 'claim' | etc.
  content TEXT NOT NULL,
  content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED,
  PRIMARY KEY (doc_id, section)
);

CREATE INDEX IF NOT EXISTS idx_text_fts_content_tsv
  ON text_fts USING GIN (content_tsv);

-- Convenience view to see counts of TIFFs per doc
CREATE OR REPLACE VIEW doc_overview AS
SELECT
  d.doc_id,
  d.xml,
  jsonb_array_length(d.tiffs) AS tiff_count
FROM docs d;
