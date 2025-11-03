# ==== Config ====
PY        ?= .venv/bin/python3
SQLITE3   ?= sqlite3

RAW       ?= data/raw
WORK      ?= data/work
EXTRACT   ?= $(WORK)/extracted
DSDIR     ?= data/ds
DB        ?= $(WORK)/index.sqlite

SCHEMA    ?= src/training/schema.sql
INDEXER   ?= src/training/index_docs.py
BUILDER   ?= src/training/build_dataset.py
FTS_LOADER?= src/training/load_fts.py

TRAIN_OUT ?= $(DSDIR)/train.jsonl
VAL_OUT   ?= $(DSDIR)/val.jsonl
TEST_OUT  ?= $(DSDIR)/test.jsonl

# ==== Phony ====
.PHONY: help all dirs init-db index reindex fts dataset train val test \
        check dbinfo counts vacuum integrity clean-outputs clean-all

# ==== Help ====
help:
	@echo "Targets:"
	@echo "  dirs            Create required directories"
	@echo "  init-db         Create SQLite schema"
	@echo "  index           Index ZIP bundles into docs table"
	@echo "  reindex         Re-extract and re-index with --force"
	@echo "  train           Build train JSONL"
	@echo "  val             Build val JSONL"
	@echo "  test            Build test JSONL"
	@echo "  dataset         Build all splits (train/val/test)"
	@echo "  fts             Load captions/paras/claims into FTS table"
	@echo "  check           Show quick DB stats"
	@echo "  dbinfo          Show SQLite PRAGMAs"
	@echo "  counts          Count rows per table"
	@echo "  vacuum          VACUUM database"
	@echo "  integrity       PRAGMA integrity_check"
	@echo "  clean-outputs   Remove generated PNGs, control maps, JSONL"
	@echo "  clean-all       Remove DB, extracted bundles, and outputs"

# ==== Directory setup ====
dirs:
	mkdir -p $(RAW) $(WORK) $(EXTRACT) $(DSDIR)
	mkdir -p $(WORK)/images $(WORK)/control

# ==== Database ====
init-db: dirs
	$(SQLITE3) $(DB) < $(SCHEMA)

dbinfo:
	@$(SQLITE3) $(DB) "PRAGMA database_list; PRAGMA journal_mode; PRAGMA synchronous;"

counts:
	@$(SQLITE3) $(DB) "SELECT 'docs', COUNT(*) FROM docs;"
	@$(SQLITE3) $(DB) "SELECT 'figures', COUNT(*) FROM figures;" || true
	@$(SQLITE3) $(DB) "SELECT 'text_fts', COUNT(*) FROM text_fts;" || true

vacuum:
	@$(SQLITE3) $(DB) "VACUUM;"

integrity:
	@$(SQLITE3) $(DB) "PRAGMA integrity_check;"

# ==== Indexing from ZIPs ====
# Scans $(RAW) for .zip files. Ignores zips with 0 or >1 XML.
index: init-db
	$(PY) $(INDEXER) $(RAW) $(WORK) $(DB)

reindex: init-db
	$(PY) $(INDEXER) $(RAW) $(WORK) $(DB) --force

# ==== Build datasets ====
dataset: train val test

train: dirs
	$(PY) $(BUILDER) --index $(DB) --out_dir $(WORK) --jsonl_out $(TRAIN_OUT) --split train

val: dirs
	$(PY) $(BUILDER) --index $(DB) --out_dir $(WORK) --jsonl_out $(VAL_OUT) --split val

test: dirs
	$(PY) $(BUILDER) --index $(DB) --out_dir $(WORK) --jsonl_out $(TEST_OUT) --split test

# ==== Optional FTS load ====
fts:
	$(PY) $(FTS_LOADER)

# ==== QA utilities ====
check: counts dbinfo

# ==== Cleanup ====
clean-outputs:
	rm -rf $(WORK)/images $(WORK)/control $(DSDIR)
	mkdir -p $(WORK)/images $(WORK)/control $(DSDIR)

clear-database:
	$(SQLITE3) $(DB) "DELETE FROM docs; DELETE FROM figures; DELETE FROM text_fts;"

clean-all:
	rm -rf $(WORK) $(DSDIR)
	mkdir -p $(WORK) $(DSDIR)
	$(SQLITE3) $(DB) "DELETE FROM docs; DELETE FROM figures; DELETE FROM text_fts;"
