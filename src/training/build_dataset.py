#!/usr/bin/env python3
"""
Build a text→image training dataset from USPTO Red Book XML + TIFF pages using a Neon/Postgres index.

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
- XML and TIFF paths are pre-indexed in Postgres table `docs(doc_id TEXT PRIMARY KEY, xml TEXT, tiffs JSONB)`.
- `img_norm.py` defines: load_tiff, deskew, binarize, split_figures, pad_square.
- `prompt.py` defines: build_prompt(fig_text, claims_subset=None, fig_label=None).
- `control.py` defines: canny_map(png_in, png_out).

Usage:
  python src/training/build_dataset.py \
    --dsn postgresql://... \
    --out_dir data/work \
    --jsonl_out data/ds/train.jsonl \
    --split train
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict

# Local modules
from psycopg.rows import tuple_row

from parse_xml import parse_doc, FIG_RX
from img_norm import load_tiff, binarize, split_figures, pad_square
from control import canny_map
from prompt import build_prompt, classify_diagram_type

try:  # Allow running as script or module.
    from training.db_utils import connection_ctx
except ImportError:  # pragma: no cover
    from db_utils import connection_ctx


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


def _upsert_text_sections(cur, doc_id: str, meta: Dict) -> None:
    """Populate caption/paragraph/claim text so downstream FTS queries work."""
    captions = "\n".join(meta.get("titles") or [])
    paragraphs = "\n".join(meta.get("figure_paras") or [])
    claims = meta.get("claims", "") or ""
    rows = [
        (doc_id, "caption", captions),
        (doc_id, "paragraph", paragraphs),
        (doc_id, "claim", claims),
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


def _upsert_figure(
    cur,
    *,
    figure_id: str,
    doc_id: str,
    tiff_path: str,
    fig_label: str | None,
    bbox: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = bbox
    cur.execute(
        """
        INSERT INTO figures (id, doc_id, tiff_path, fig_label, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id)
        DO UPDATE SET
          doc_id = EXCLUDED.doc_id,
          tiff_path = EXCLUDED.tiff_path,
          fig_label = EXCLUDED.fig_label,
          bbox_x1 = EXCLUDED.bbox_x1,
          bbox_y1 = EXCLUDED.bbox_y1,
          bbox_x2 = EXCLUDED.bbox_x2,
          bbox_y2 = EXCLUDED.bbox_y2
        """,
        (figure_id, doc_id, tiff_path, fig_label, x1, y1, x2, y2),
    )


# ------------ Core builders ------------

def _choose_fig_text(meta: Dict, fig_num: str | None) -> str:
    """
    Choose paragraphs describing a specific figure:
    - start at the first paragraph referencing `fig_num`
    - include following paragraphs until one references other figures
    - ensure at least two paragraphs by appending the immediate next paragraph when available
    Falls back to generic figure paragraphs or titles if no match is found.
    """
    paragraphs = [p for p in (meta.get("figure_paras") or []) if p and p.strip()]
    titles = [t for t in (meta.get("titles") or []) if t and t.strip()]
    if not paragraphs:
        return " ".join(titles[:2])

    if not fig_num:
        # No figure identifier; fall back to the first couple of paragraphs.
        return " ".join(paragraphs[:2]) or " ".join(titles[:2])

    target = canonical_fig_num(fig_num)
    if not target:
        return " ".join(paragraphs[:2]) or " ".join(titles[:2])

    para_refs = []
    for p in paragraphs:
        refs = set()
        for m in FIG_RX.finditer(p):
            canon = canonical_fig_num(m.group(1))
            if canon:
                refs.add(canon)
        para_refs.append(refs)

    start_idx = None
    for idx, refs in enumerate(para_refs):
        if target in refs:
            start_idx = idx
            break
    if start_idx is None:
        normalized_target = target.replace("-", "")
        for idx, paragraph in enumerate(paragraphs):
            para_norm = (
                paragraph.upper()
                .replace(" ", "")
                .replace(".", "")
                .replace(",", "")
                .replace("-", "")
            )
            if "FIG" in para_norm and normalized_target in para_norm:
                start_idx = idx
                break
    if start_idx is None:
        return " ".join(paragraphs[:2]) or " ".join(titles[:2])

    selected = []
    idx = start_idx
    while idx < len(paragraphs):
        refs = para_refs[idx]
        if idx != start_idx and refs and target not in refs:
            break
        selected.append(paragraphs[idx])
        idx += 1

    if len(selected) < 2 and start_idx + 1 < len(paragraphs):
        next_para = paragraphs[start_idx + 1]
        if next_para not in selected:
            selected.append(next_para)

    if selected:
        return " ".join(selected)

    return " ".join(paragraphs[:2]) or " ".join(titles[:2])


def process_tiff(
    doc_id: str,
    xml_path: Path,
    tiff_path: Path,
    out_img_dir: Path,
    out_ctrl_dir: Path,
    jsonl_file,
    meta: Dict,
    target_size: int = 2048,
    figure_recorder: Callable[[str, str, str, str | None, tuple[int, int, int, int]], None] | None = None,
) -> int:
    """
    Split a TIFF page into figure crops, save PNGs + control maps, and emit JSONL rows.
    Returns the number of figures written.
    """
    # Load and normalize page
    img = load_tiff(str(tiff_path))
    if img is None:
        return 0
    # Skip deskew to preserve original orientation
    img = binarize(img)

    # Detect figure regions: list of (crop_img, fig_label, (x1,y1,x2,y2))
    tiles = split_figures(img)
    if not tiles:
        return 0

    # Prepare shared text block for prompts
    fig_nums_xml_set: set[str] = set()
    fig_nums_source = meta.get("figure_nums", []) or []
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

    for idx, (tile_img, fig_label, bbox) in enumerate(tiles, 1):
        fig_num = extract_fig_num(fig_label)
        if not fig_num:
            continue
        if fig_num not in fig_nums_xml_set:
            continue

        fig_text = _choose_fig_text(meta, fig_num)
        diagram_type = classify_diagram_type(fig_text)
        method_claim = meta.get("method_claim") or ""
        first_independent_claim = meta.get("first_independent_claim") or ""
        claims_text = meta.get("claims", "") or ""
        default_claims = first_independent_claim or claims_text
        claims_source = method_claim if (diagram_type == "flowchart" and method_claim) else default_claims

        # Use the full tile image (no cropping)
        # tile_img is already the full page from split_figures
        if tile_img.size == 0:
            continue

        # Resize to square canvas without losing aspect ratio
        crop, resize_meta = pad_square(tile_img, size=target_size, return_meta=True)
        
        # Store original dimensions before padding
        orig_height, orig_width = tile_img.shape[:2]
        x1, y1, x2, y2 = bbox  # Keep original bbox for metadata

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
        prompt = build_prompt(fig_text, claims_subset=claims_source, fig_label=fig_label)

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
            "bbox_initial": [int(x1), int(y1), int(x2), int(y2)],
            "original_size": [int(orig_width), int(orig_height)],  # width, height
            "resize_meta": resize_meta,
            "fig_label": fig_label,
            "tiff_source": str(tiff_path),
            "label_in_xml": bool(match_xml),
        }
        jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        if figure_recorder:
            figure_recorder(
                row["id"],
                doc_id,
                str(tiff_path),
                fig_label,
                (int(x1), int(y1), int(x2), int(y2)),
            )
        written += 1
    return written


def main(dsn: str | None, out_dir: Path, jsonl_out: Path, split: str) -> None:
    ensure_dir(jsonl_out.parent)

    # Output subdirs
    out_img_dir = out_dir / "images"
    out_ctrl_dir = out_dir / "control"

    total_docs = 0
    total_pages = 0
    total_figs = 0

    with connection_ctx(dsn, row_factory=tuple_row) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT doc_id, xml, tiffs FROM docs")
            rows = cur.fetchall()

        with conn.cursor() as write_cur:
            def record_figure(
                figure_id: str,
                doc_id: str,
                tiff_path: str,
                fig_label: str | None,
                bbox: tuple[int, int, int, int],
            ) -> None:
                _upsert_figure(
                    write_cur,
                    figure_id=figure_id,
                    doc_id=doc_id,
                    tiff_path=tiff_path,
                    fig_label=fig_label,
                    bbox=bbox,
                )

            with open(jsonl_out, "w", encoding="utf-8") as jf:
                for doc_id, xml, tiffs_value in rows:
                    xml_path = Path(xml)
                    if not xml_path.exists():
                        continue

                    # Parse once per doc to get claims, figure numbers, and text sections
                    meta = parse_doc(str(xml_path))
                    _upsert_text_sections(write_cur, doc_id, meta)

                    if isinstance(tiffs_value, str):
                        try:
                            tiffs = json.loads(tiffs_value or "[]")
                        except Exception:
                            tiffs = []
                    elif tiffs_value is None:
                        tiffs = []
                    else:
                        tiffs = list(tiffs_value)
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
                            meta=meta,
                            target_size=2048,
                            figure_recorder=record_figure,
                        )
                        total_figs += figs
            conn.commit()

    # Minimal run summary to stderr
    import sys
    print(
        f"[build_dataset] split={split} docs={total_docs} pages={total_pages} figures={total_figs} -> {jsonl_out}",
        file=sys.stderr,
    )


# ------------ CLI ------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build patent-drawing training dataset")
    ap.add_argument("--dsn", help="Postgres connection string; falls back to DATABASE_URL env variable.")
    ap.add_argument("--out_dir", required=True, help="Working dir for images/control")
    ap.add_argument("--jsonl_out", required=True, help="Output JSONL path")
    ap.add_argument("--split", required=True, choices=["train", "val", "test"], help="Dataset split label")
    args = ap.parse_args()

    try:
        main(
            dsn=args.dsn,
            out_dir=Path(args.out_dir),
            jsonl_out=Path(args.jsonl_out),
            split=args.split,
        )
    except RuntimeError as exc:  # pragma: no cover - CLI guard
        raise SystemExit(str(exc)) from exc
