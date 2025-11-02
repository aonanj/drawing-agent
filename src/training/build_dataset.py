#!/usr/bin/env python3
"""
Build a text→image training dataset from USPTO Red Book XML + TIFF pages.

Each output row (JSONL) corresponds to ONE figure crop:
{
  "id": "US1234567:page123_FIG_3A.png",
  "image_path": "data/work/images/US1234567_page123_FIG_3A.png",
  "control_path": "data/work/control/US1234567_page123_FIG_3A_canny.png",
  "prompt": "...",
  "source_xml": "data/work/xml/US1234567.xml",
  "doc_id": "US1234567",
  "sha256": "...",
  "bbox": [x1,y1,x2,y2],
  "fig_label": "FIG. 3A",
  "tiff_source": "data/work/tiff/US1234567_page123.tif",
  "label_in_xml": true | false
}

Assumptions:
- XML and TIFF paths are pre-indexed in SQLite table `docs(doc_id TEXT PRIMARY KEY, xml TEXT, tiffs TEXT_JSON)`.
- `img_norm.py` defines: load_tiff, deskew, binarize, split_figures, pad_square.
- `prompt.py` defines: build_prompt(fig_text, claims_subset=None, fig_label=None).
- `control.py` defines: canny_map(png_in, png_out).

Usage:
  python scripts/build_dataset.py \
    --index data/work/index.sqlite \
    --out_dir data/work \
    --jsonl_out data/ds/train.jsonl \
    --split train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Dict, Iterable

# Local modules
from parse_xml import parse_doc
from img_norm import load_tiff, deskew, binarize, split_figures, pad_square
from control import canny_map
from prompt import build_prompt


# ------------ FS utils ------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize(s: str) -> str:
    # For filenames: keep alnum, convert separators to underscore
    keep = []
    for ch in (s or ""):
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", ".", "-", "/"):
            keep.append("_")
        else:
            # drop
            pass
    out = "".join(keep).strip("_")
    return out or "UNLABELED"


# ------------ Core builders ------------

def _choose_fig_text(meta: Dict) -> str:
    """Prefer the first two figure-referencing paragraphs, else caption titles, else empty."""
    ps = [p for p in (meta.get("figure_paras") or []) if p and p.strip()]
    if not ps:
        titles = [t for t in (meta.get("titles") or []) if t and t.strip()]
        return " ".join(titles[:2])
    return " ".join(ps[:2])


def process_tiff(
    doc_id: str,
    xml_path: Path,
    tiff_path: Path,
    out_img_dir: Path,
    out_ctrl_dir: Path,
    jsonl_file,
    claims_text: str,
    fig_nums_from_xml: Iterable[str],
    target_size: int = 1024,
) -> int:
    """
    Split a TIFF page into figure crops, save PNGs + control maps, and emit JSONL rows.
    Returns the number of figures written.
    """
    # Load and normalize page
    img = load_tiff(str(tiff_path))
    if img is None:
        return 0
    img = deskew(img)
    img = binarize(img)

    # Detect figure regions: list of (crop_img, fig_label, (x1,y1,x2,y2))
    tiles = split_figures(img)
    if not tiles:
        return 0

    # Prepare shared text block for prompts
    meta = parse_doc(str(xml_path))
    fig_text = _choose_fig_text(meta)
    fig_nums_xml_set = {s.upper() for s in (fig_nums_from_xml or [])} or set(meta.get("figure_nums", []))

    written = 0
    base = f"{doc_id}_{Path(tiff_path).stem}"

    ensure_dir(out_img_dir)
    ensure_dir(out_ctrl_dir)

    for idx, (crop, fig_label, bbox) in enumerate(tiles, 1):
        # Resize to square canvas
        crop = pad_square(crop, size=target_size)

        # Name with OCR label if present
        label_token = sanitize((fig_label or f"FIG_{idx}").upper().replace("FIGURE", "FIG"))
        img_name = f"{base}_{label_token}.png"
        ctrl_name = f"{base}_{label_token}_canny.png"

        img_out = out_img_dir / img_name
        ctrl_out = out_ctrl_dir / ctrl_name

        # Save image
        import cv2  # local import to keep module scope clean
        cv2.imwrite(str(img_out), crop)

        # Control map
        canny_map(str(img_out), str(ctrl_out))

        # Prompt with fig_label cue
        prompt = build_prompt(fig_text, claims_subset=claims_text[:500], fig_label=fig_label)

        # Label↔XML consistency flag
        match_xml = False
        if fig_label:
            # Accept "FIG. 3A" -> "3A"
            num = fig_label.upper().replace("FIGURE", "FIG").replace("FIG.", "FIG").replace("FIG ", "FIG")
            num = num.replace("FIG", "").strip()
            match_xml = num in fig_nums_xml_set if num else False

        row = {
            "id": f"{doc_id}:{img_name}",
            "image_path": str(img_out),
            "control_path": str(ctrl_out),
            "prompt": prompt,
            "source_xml": str(xml_path),
            "doc_id": doc_id,
            "sha256": sha256_file(img_out),
            "bbox": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "fig_label": fig_label,
            "tiff_source": str(tiff_path),
            "label_in_xml": bool(match_xml),
        }
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1
    return written


def main(index_db: Path, out_dir: Path, jsonl_out: Path, split: str) -> None:
    ensure_dir(jsonl_out.parent)

    # Output subdirs
    out_img_dir = out_dir / "images"
    out_ctrl_dir = out_dir / "control"

    conn = sqlite3.connect(str(index_db))
    cur = conn.cursor()
    cur.execute("SELECT doc_id, xml, tiffs FROM docs")

    total_docs = 0
    total_pages = 0
    total_figs = 0

    with open(jsonl_out, "w", encoding="utf-8") as jf:
        for doc_id, xml, tiffs_json in cur.fetchall():
            xml_path = Path(xml)
            if not xml_path.exists():
                continue

            # Parse once per doc to get claims and figure numbers
            meta = parse_doc(str(xml_path))
            claims_text = meta.get("claims", "") or ""
            fig_nums_from_xml = meta.get("figure_nums", []) or []

            try:
                tiffs = json.loads(tiffs_json or "[]")
            except Exception:
                tiffs = []
            if not tiffs:
                continue

            total_docs += 1
            for tiff in tiffs:
                tiff_path = Path(tiff)
                if not tiff_path.exists():
                    continue
                total_pages += 1

                figs = process_tiff(
                    doc_id=doc_id,
                    xml_path=xml_path,
                    tiff_path=tiff_path,
                    out_img_dir=out_img_dir,
                    out_ctrl_dir=out_ctrl_dir,
                    jsonl_file=jf,
                    claims_text=claims_text,
                    fig_nums_from_xml=fig_nums_from_xml,
                    target_size=1024,
                )
                total_figs += figs

    conn.close()

    # Minimal run summary to stderr
    import sys
    print(
        f"[build_dataset] split={split} docs={total_docs} pages={total_pages} figures={total_figs} -> {jsonl_out}",
        file=sys.stderr,
    )


# ------------ CLI ------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build patent-drawing training dataset")
    ap.add_argument("--index", required=True, help="SQLite index with table docs(doc_id, xml, tiffs)")
    ap.add_argument("--out_dir", required=True, help="Working dir for images/control")
    ap.add_argument("--jsonl_out", required=True, help="Output JSONL path")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dataset split label")
    args = ap.parse_args()

    main(
        index_db=Path(args.index),
        out_dir=Path(args.out_dir),
        jsonl_out=Path(args.jsonl_out),
        split=args.split,
    )
