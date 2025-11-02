import pytesseract
from pytesseract import Output
import re
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Union, Literal, overload, TypeAlias

FIG_NO_PAT = re.compile(r"^[0-9]{1,4}[A-Z]{0,2}(?:-[0-9]{1,4})?$")
DASH_TRANSLATION = str.maketrans({
    "–": "-",  # en dash
    "—": "-",  # em dash
    "―": "-",  # horizontal bar
    "−": "-",  # minus sign
    "‐": "-",  # hyphen bullet
    "‑": "-",  # non-breaking hyphen
    "‒": "-",  # figure dash
    "﹘": "-",  # small em dash
    "﹣": "-",  # small hyphen-minus
    "－": "-",  # fullwidth hyphen-minus
})
STRIP_CHARS = "\"'()[]{}.,;!?`~:"

def ocr_words(img):
    """
    Run Tesseract word-level OCR. Expects dark ink on white.
    Returns list of dicts: {text, x, y, w, h, cx, cy}
    """
    data = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 6")
    words = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        words.append({
            "text": txt,
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "cx": int(x + w/2), "cy": int(y + h/2)
        })
    return words

def find_fig_labels(words):
    """
    Locate tokens 'FIG'/'FIG.'/'FIGURE' followed by a number token.
    Returns list of dicts: {label:'FIG. 3A', x,y,w,h,cx,cy, num:'3A', row_y, col_x}
    """
    if not words:
        return []

    def normalize_token(text):
        return text.translate(DASH_TRANSLATION)

    def indicator_from_text(text):
        token = normalize_token(text).strip()
        if not token:
            return False, ""
        token = token.lstrip("\"'([{").rstrip("\"')]}.,;!?")
        if not token:
            return False, ""
        upper = token.upper()
        for prefix in ("FIGURE", "FIG"):
            if not upper.startswith(prefix):
                continue
            next_idx = len(prefix)
            if next_idx < len(upper) and upper[next_idx].isalpha():
                # guard against FIGURES, FIGARO, etc.
                continue
            suffix = token[next_idx:]
            suffix = suffix.lstrip(".:- ")
            suffix = suffix.replace(" ", "")
            return True, normalize_token(suffix).upper()
        return False, ""

    labels = []
    max_lookahead = 6
    for i, w in enumerate(words):
        has_indicator, inline_fragment = indicator_from_text(w["text"])
        if not has_indicator:
            continue

        candidate_chars = []
        digits_first = 0
        letters_first = 0
        digits_second = 0
        hyphen_used = False
        digits_seen = False
        last_idx = i

        def process_chars(seq):
            nonlocal digits_first, letters_first, digits_second, hyphen_used, digits_seen
            for ch in seq:
                if ch.isdigit():
                    digits_seen = True
                    if not hyphen_used:
                        if letters_first:
                            return False
                        if digits_first >= 4:
                            return False
                        digits_first += 1
                    else:
                        if digits_second >= 4:
                            return False
                        digits_second += 1
                    candidate_chars.append(ch)
                elif ch == "-":
                    if hyphen_used or not digits_first:
                        return False
                    hyphen_used = True
                    candidate_chars.append("-")
                elif "A" <= ch <= "Z":
                    if hyphen_used or not digits_first:
                        return False
                    if letters_first >= 2:
                        return False
                    letters_first += 1
                    candidate_chars.append(ch)
                else:
                    return False
            return True

        if inline_fragment:
            if not process_chars(inline_fragment):
                continue

        j = i + 1
        while j < len(words) and (j - i) <= max_lookahead:
            next_word = words[j]
            next_indicator, _ = indicator_from_text(next_word["text"])
            if next_indicator:
                break
            if abs(next_word["cy"] - w["cy"]) > max(w["h"], next_word["h"]) * 1.8:
                break
            token = normalize_token(next_word["text"]).strip()
            if not token:
                j += 1
                continue
            token_core = token.strip(STRIP_CHARS).replace(" ", "")
            if not token_core:
                j += 1
                continue
            token_upper = token_core.upper()
            if not candidate_chars and token_upper in {"-", ":"}:
                j += 1
                continue
            if not process_chars(token_upper):
                break
            last_idx = j
            j += 1

        if not candidate_chars or not digits_seen:
            continue
        if hyphen_used and digits_second == 0:
            continue
        num = "".join(candidate_chars)
        if not FIG_NO_PAT.match(num):
            continue

        tokens = words[i:last_idx + 1]
        x1 = min(t["x"] for t in tokens)
        y1 = min(t["y"] for t in tokens)
        x2 = max(t["x"] + t["w"] for t in tokens)
        y2 = max(t["y"] + t["h"] for t in tokens)
        labels.append({
            "label": f"FIG. {num}",
            "num": num,
            "x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1,
            "cx": int((x1 + x2) / 2), "cy": int((y1 + y2) / 2),
            "row_y": int((y1 + y2) / 2), "col_x": int((x1 + x2) / 2)
        })
    return labels

def cluster_by_axis(vals, tol):
    """
    Simple 1D clustering. vals: list of ints. tol: max gap to join.
    Returns list of clusters, each is list of indices into the original list.
    """
    if not vals:
        return []
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    clusters, cur = [], [order[0]]
    for idx in order[1:]:
        if abs(vals[idx] - vals[cur[-1]]) <= tol:
            cur.append(idx)
        else:
            clusters.append(cur)
            cur = [idx]
    clusters.append(cur)
    return clusters

def compute_splits_from_labels(img_shape, labels):
    """
    Build rectangular cells using row clusters (y) then column clusters (x) per row.
    Returns list of cells: dict with bounds and the chosen label for the cell.
    """
    H, W = img_shape[:2]
    if not labels:
        return [{"x1": 0, "y1": 0, "x2": W, "y2": H, "label": None}]

    # cluster rows
    row_tol = max(12, int(0.03 * H))
    row_clusters = cluster_by_axis([lab["row_y"] for lab in labels], row_tol)

    cells = []
    # For each row, cluster columns, then create horizontal bands between adjacent row midlines
    row_centers = [int(sum(labels[i]["row_y"] for i in cl) / len(cl)) for cl in row_clusters]
    row_mid = []
    row_centers_sorted = sorted(row_centers)
    for k, rc in enumerate(row_centers_sorted):
        if k == 0:
            y1 = 0
        else:
            y1 = int((row_centers_sorted[k - 1] + rc) / 2)
        if k == len(row_centers_sorted) - 1:
            y2 = H
        else:
            y2 = int((rc + row_centers_sorted[k + 1]) / 2)
        row_mid.append((y1, y2))

    # Map cluster order back to rows
    # Sort clusters by their mean row_y
    row_clusters_sorted = sorted(row_clusters, key=lambda cl: int(sum(labels[i]["row_y"] for i in cl) / len(cl)))

    for row_idx, cl in enumerate(row_clusters_sorted):
        cols = [labels[i]["col_x"] for i in cl]
        col_tol = max(12, int(0.03 * W))
        col_clusters = cluster_by_axis(cols, col_tol)
        # Column cut points within this row band
        col_centers = [int(sum(cols[j] for j in c) / len(c)) for c in col_clusters]
        col_centers_sorted = sorted(col_centers)
        # Build vertical slices
        for k, cc in enumerate(col_centers_sorted):
            if k == 0:
                x1 = 0
            else:
                x1 = int((col_centers_sorted[k - 1] + cc) / 2)
            if k == len(col_centers_sorted) - 1:
                x2 = W
            else:
                x2 = int((cc + col_centers_sorted[k + 1]) / 2)
            y1, y2 = row_mid[row_idx]
            # choose a representative label inside this cell: nearest label by L2 to cell center
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            inside = [labels[i] for i in cl if (x1 <= labels[i]["cx"] <= x2 and y1 <= labels[i]["cy"] <= y2)]
            if not inside:
                # fallback: nearest in row
                inside = [min((labels[i] for i in cl), key=lambda lab: (lab["cx"] - cx) ** 2 + (lab["cy"] - cy) ** 2)]
            rep = inside[0]
            # add margin padding
            pad = int(0.01 * max(W, H))
            cells.append({
                "x1": max(0, x1 - pad), "y1": max(0, y1 - pad),
                "x2": min(W, x2 + pad), "y2": min(H, y2 + pad),
                "label": rep["label"], "num": rep["num"]
            })
    # Deduplicate overlapping cells when there are fewer labels than grid slots
    dedup = []
    for c in cells:
        area = (c["x2"] - c["x1"]) * (c["y2"] - c["y1"])
        if area < 0.02 * W * H:
            continue
        if not any(
            abs(c["x1"] - d["x1"]) < 8 and abs(c["y1"] - d["y1"]) < 8 and
            abs(c["x2"] - d["x2"]) < 8 and abs(c["y2"] - d["y2"]) < 8 for d in dedup
        ):
            dedup.append(c)
    return dedup

def split_figures(img):
    """
    Detect per-figure regions on a patent drawings page by OCRing figure labels.
    Returns a list of tuples: (crop_img, fig_label, (x1,y1,x2,y2))
    Heuristics:
      - Find 'FIG', 'FIG.', or 'FIGURE' followed by a number token (e.g., 3A).
      - Cluster labels into rows and columns.
      - Split the page into cells around label midpoints.
    Fallback: if no labels, return the full page as a single figure with label=None.
    """
    H, W = img.shape[:2]

    # Ensure OCR-friendly input: light background, dark ink
    work = img
    # If image is inverted for any reason, flip by checking mean
    if work.mean() < 127:
        work = 255 - work

    words = ocr_words(work)
    labels = find_fig_labels(words)

    cells = compute_splits_from_labels((H, W), labels)

    out = []
    for c in cells:
        x1, y1, x2, y2 = c["x1"], c["y1"], c["x2"], c["y2"]
        # clip and enforce bounds
        x1 = max(0, min(x1, W - 1))
        x2 = max(1, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(1, min(y2, H))
        if x2 - x1 < 32 or y2 - y1 < 32:
            continue
        crop = work[y1:y2, x1:x2].copy()
        label = c.get("label")
        out.append((crop, label, (x1, y1, x2, y2)))
    # Sort by top-left for stable order
    out.sort(key=lambda t: (t[2][1], t[2][0]))
    return out

def load_tiff(path):
    # cv2 can read bilevel/group4 as grayscale
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    assert img is not None
    return img

def deskew(img):
    # Hough or minAreaRect on edges
    edges = cv2.Canny(img, 50, 150)
    coords = np.column_stack(np.where(edges>0))
    angle = cv2.minAreaRect(coords)[-1]
    angle = -(90+angle) if angle<-45 else -angle
    (h,w)=img.shape
    M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderValue=255)

def binarize(img):
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

def expand_bbox(img, bbox, max_expand=512, min_fraction=0.005, margin=12):
    """
    Expand a bounding box outward until the added border no longer contains ink.

    Parameters
    ----------
    img : np.ndarray
        2D grayscale image (ink is dark / low values).
    bbox : tuple[int, int, int, int]
        (x1, y1, x2, y2) bounds inside img.
    max_expand : int
        Maximum pixels to grow per direction.
    min_fraction : float
        Minimum proportion of inked pixels required in the candidate border
        to continue expanding.
    margin : int
        Extra padding (in pixels) added after content-based expansion.
    """
    H, W = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, min(x1, W - 1))
    x2 = max(x1 + 1, min(x2, W))
    y1 = max(0, min(y1, H - 1))
    y2 = max(y1 + 1, min(y2, H))

    mask = img < 250  # True where ink is present

    def grow_left(x1, y1, y2):
        steps = 0
        while x1 > 0 and steps < max_expand:
            column = mask[y1:y2, x1 - 1]
            if column.mean() > min_fraction:
                x1 -= 1
                steps += 1
            else:
                break
        return x1

    def grow_right(x2, y1, y2):
        steps = 0
        while x2 < W and steps < max_expand:
            column = mask[y1:y2, x2]
            if column.mean() > min_fraction:
                x2 += 1
                steps += 1
            else:
                break
        return x2

    def grow_up(y1, x1, x2):
        steps = 0
        while y1 > 0 and steps < max_expand:
            row = mask[y1 - 1, x1:x2]
            if row.mean() > min_fraction:
                y1 -= 1
                steps += 1
            else:
                break
        return y1

    def grow_down(y2, x1, x2):
        steps = 0
        while y2 < H and steps < max_expand:
            row = mask[y2, x1:x2]
            if row.mean() > min_fraction:
                y2 += 1
                steps += 1
            else:
                break
        return y2

    for _ in range(6):
        new_x1 = grow_left(x1, y1, y2)
        new_x2 = grow_right(x2, y1, y2)
        new_y1 = grow_up(y1, new_x1, new_x2)
        new_y2 = grow_down(y2, new_x1, new_x2)
        if (new_x1, new_y1, new_x2, new_y2) == (x1, y1, x2, y2):
            break
        x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2

    # Final margin padding
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(W, x2 + margin)
    y2 = min(H, y2 + margin)
    return x1, y1, x2, y2


PadSquareMeta = Dict[str, Union[int, float]]
PadSquareImage: TypeAlias = np.ndarray


@overload
def pad_square(img: PadSquareImage, size: int = 2048, *, return_meta: Literal[False] = False) -> PadSquareImage:
    ...


@overload
def pad_square(img: PadSquareImage, size: int = 2048, *, return_meta: Literal[True]) -> Tuple[PadSquareImage, PadSquareMeta]:
    ...


def pad_square(img: PadSquareImage, size: int = 2048, *, return_meta: bool = False) -> Union[PadSquareImage, Tuple[PadSquareImage, PadSquareMeta]]:
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Cannot pad empty image")

    max_dim = max(h, w)
    if max_dim > size:
        scale = size / float(max_dim)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    else:
        scale = 1.0
        new_w = int(w)
        new_h = int(h)
        resized = img.copy()

    canvas: PadSquareImage = np.full((size, size), 255, np.uint8)
    offset_y = (size - new_h) // 2
    offset_x = (size - new_w) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    if return_meta:
        meta: PadSquareMeta = {
            "orig_width": int(w),
            "orig_height": int(h),
            "resized_width": int(new_w),
            "resized_height": int(new_h),
            "offset_x": int(offset_x),
            "offset_y": int(offset_y),
            "scale": float(scale),
            "canvas_size": int(size),
        }
        return canvas, meta

    return canvas

def normalize_tiff(tiff_path, out_png):
    img = load_tiff(tiff_path)
    img = deskew(img)
    img = binarize(img)

    # split into figure crops
    tiles = split_figures(img)  # list of (crop, label, bbox)
    outs = []
    base = Path(out_png).with_suffix("").as_posix()

    for i, (tile, label, _) in enumerate(tiles, 1):
        # pad and resize
        tile = pad_square(tile, 2048)
        # sanitize label for filename
        lbl = (label or f"FIG_{i}").upper().replace(".", "").replace(" ", "_")
        out = f"{base}_{lbl}.png"
        cv2.imwrite(out, tile)
        outs.append(out)
    return outs
