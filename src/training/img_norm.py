import pytesseract
from pytesseract import Output
import re
import cv2
import numpy as np
from pathlib import Path

FIG_PAT = re.compile(r"^(FIG(?:\.|URE)?)$", re.IGNORECASE)
FIG_NO_PAT = re.compile(r"^([0-9]{1,3}[A-Za-z]?)$")  # 1, 2, 10, 3A, etc.

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
    labels = []
    for i, w in enumerate(words):
        if not FIG_PAT.match(w["text"]):
            continue
        # look ahead for the next token that looks like a number/letter combo
        j = i + 1
        while j < len(words) and words[j]["text"] in {":", "-", "—"}:
            j += 1
        if j < len(words):
            match = FIG_NO_PAT.match(words[j]["text"])
            if match:
                num = match.group(1)
            else:
                continue
            # union bbox of 'FIG' and number token
            x1 = min(w["x"], words[j]["x"])
            y1 = min(w["y"], words[j]["y"])
            x2 = max(w["x"] + w["w"], words[j]["x"] + words[j]["w"])
            y2 = max(w["y"] + w["h"], words[j]["y"] + words[j]["h"])
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

def pad_square(img, size=1024):
    h,w = img.shape
    s = max(h,w)
    canvas = np.full((s,s), 255, np.uint8)
    y = (s-h)//2
    x=(s-w)//2
    canvas[y:y+h, x:x+w] = img
    return cv2.resize(canvas, (size,size), interpolation=cv2.INTER_AREA)

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
        tile = pad_square(tile, 1024)
        # sanitize label for filename
        lbl = (label or f"FIG_{i}").upper().replace(".", "").replace(" ", "_")
        out = f"{base}_{lbl}.png"
        cv2.imwrite(out, tile)
        outs.append(out)
    return outs

