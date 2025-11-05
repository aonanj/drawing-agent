# ==== Config ====
PY        ?= python3
export DATABASE_URL
export NEON_DATABASE_URL

RAW       ?= data/raw
WORK      ?= data/work
EXTRACT   ?= $(WORK)/extracted
DSDIR     ?= data/ds

SCHEMA    ?= src/training/schema.sql
INDEXER   ?= src/training/index_docs.py
BUILDER   ?= src/training/build_dataset.py
FTS_LOADER?= src/training/load_fts.py
INIT_DB   ?= src/training/init_db.py
DB_ADMIN  ?= src/training/db_admin.py

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
	@echo "  init-db         Create Neon/Postgres schema"
	@echo "  index           Index ZIP bundles into docs table"
	@echo "  reindex         Re-extract and re-index with --force"
	@echo "  train           Build train JSONL"
	@echo "  val             Build val JSONL"
	@echo "  test            Build test JSONL"
	@echo "  dataset         Build all splits (train/val/test)"
	@echo "  fts             Load captions/paras/claims into FTS table"
	@echo "  check           Show quick DB stats"
	@echo "  dbinfo          Show connection details"
	@echo "  counts          Count rows per table"
	@echo "  vacuum          VACUUM database"
	@echo "  integrity       Show live/dead tuple counts"
	@echo "  clean-outputs   Remove generated PNGs, control maps, JSONL"
	@echo "  clean-all       Remove extracted bundles, outputs, and clear database tables"

# ==== Directory setup ====
dirs:
	mkdir -p $(RAW) $(WORK) $(EXTRACT) $(DSDIR)
	mkdir -p $(WORK)/images $(WORK)/control

# ==== Database ====
init-db: dirs
	$(PY) $(INIT_DB) --schema $(SCHEMA)

dbinfo:
	@$(PY) $(DB_ADMIN) info

counts:
	@$(PY) $(DB_ADMIN) counts

vacuum:
	@$(PY) $(DB_ADMIN) vacuum

integrity:
	@$(PY) $(DB_ADMIN) integrity

# ==== Indexing from ZIPs ====
# Scans $(RAW) for .zip files. Ignores zips with 0 or >1 XML.
index: init-db
	$(PY) $(INDEXER) $(RAW) $(WORK)

reindex: init-db
	$(PY) $(INDEXER) $(RAW) $(WORK) --force

# ==== Build datasets ====
dataset: train val test

train: dirs
	$(PY) $(BUILDER) --out_dir $(WORK) --jsonl_out $(TRAIN_OUT) --split train

val: dirs
	$(PY) $(BUILDER) --out_dir $(WORK) --jsonl_out $(VAL_OUT) --split val

test: dirs
	$(PY) $(BUILDER) --out_dir $(WORK) --jsonl_out $(TEST_OUT) --split test

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
	$(PY) $(DB_ADMIN) clear

clean-all:
	rm -rf $(WORK) $(DSDIR)
	mkdir -p $(WORK) $(DSDIR)
	$(PY) $(DB_ADMIN) clear
