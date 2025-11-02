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
  "bbox_initial": [x1a,y1a,x2a,y2a],
  "original_size": [width,height],
  "resize_meta": {
      "orig_width": ...,
      "orig_height": ...,
      "resized_width": ...,
      "resized_height": ...,
      "offset_x": ...,
      "offset_y": ...,
      "scale": ...,
      "canvas_size": ...
  },
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
  python src/training/build_dataset.py \
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
from img_norm import load_tiff, deskew, binarize, split_figures, pad_square, expand_bbox
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


def canonical_fig_num(num: str | None) -> str:
    """
    Normalize figure identifiers to a canonical form:
      - uppercase
      - collapse internal spaces
      - trim leading zeros on the numeric part (preserving a single zero)
      - allow a single hyphen to denote ranges (e.g., 3-4)
    Returns an empty string if the token is invalid.
    """
    if not num:
        return ""
    raw = num.strip().upper().replace(" ", "")
    if not raw:
        return ""
    parts = raw.split("-")
    if len(parts) > 2:
        return ""
    normalized_parts = []
    for part in parts:
        if not part:
            return ""
        digits = []
        suffix = []
        in_suffix = False
        for ch in part:
            if ch.isdigit() and not in_suffix:
                digits.append(ch)
                continue
            if ch.isalpha():
                in_suffix = True
                suffix.append(ch)
                continue
            return ""
        if not digits:
            return ""
        num_part = "".join(digits).lstrip("0") or "0"
        if len(num_part) > 4 or len(suffix) > 2:
            return ""
        normalized_parts.append(num_part + "".join(suffix))
    return "-".join(normalized_parts)


def extract_fig_num(label: str | None) -> str | None:
    """
    Extract a canonical figure identifier (e.g., '3A') from OCR-provided labels like 'FIG. 3A'.
    Returns None if a valid figure number cannot be located.
    """
    if not label:
        return None
    upper = label.upper()
    for marker in ("FIGURE", "FIG"):
        if marker in upper:
            suffix = upper.split(marker, 1)[1]
            break
    else:
        return None
    suffix = suffix.replace(".", " ").replace(":", " ").replace("/", " ")
    suffix = suffix.strip(" .:_-")
    if not suffix:
        return None
    suffix = suffix.replace(" ", "")
    canon = canonical_fig_num(suffix)
    return canon or None


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
    target_size: int = 2048,
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
    fig_nums_xml_set: set[str] = set()
    fig_nums_source = fig_nums_from_xml or meta.get("figure_nums", []) or []
    for raw_num in fig_nums_source:
        canon = canonical_fig_num(raw_num)
        if canon:
            fig_nums_xml_set.add(canon)
    if not fig_nums_xml_set:
        return 0

    written = 0
    base = f"{doc_id}_{Path(tiff_path).stem}"

    ensure_dir(out_img_dir)
    ensure_dir(out_ctrl_dir)

    for idx, (_, fig_label, bbox) in enumerate(tiles, 1):
        fig_num = extract_fig_num(fig_label)
        if not fig_num:
            continue
        if fig_num not in fig_nums_xml_set:
            continue
        # Expand bbox to include full inked content before cropping
        x1, y1, x2, y2 = expand_bbox(img, bbox)
        raw_crop = img[y1:y2, x1:x2]
        if raw_crop.size == 0:
            continue

        # Resize to square canvas without losing aspect ratio
        crop, resize_meta = pad_square(raw_crop, size=target_size, return_meta=True)

        # Name with OCR label if present
        label_token = sanitize(f"FIG_{fig_num}")
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
        match_xml = fig_num in fig_nums_xml_set

        row = {
            "id": f"{doc_id}:{img_name}",
            "image_path": str(img_out),
            "control_path": str(ctrl_out),
            "prompt": prompt,
            "source_xml": str(xml_path),
            "doc_id": doc_id,
            "sha256": sha256_file(img_out),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "bbox_initial": [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
            "original_size": [int(raw_crop.shape[1]), int(raw_crop.shape[0])],  # width, height
            "resize_meta": resize_meta,
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
                name_upper = tiff_path.name.upper()
                if name_upper.endswith("D00000.TIF") or name_upper.endswith("D00000.TIFF"):
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
                    target_size=2048,
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
