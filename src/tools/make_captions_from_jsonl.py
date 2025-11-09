# tools/make_captions_from_jsonl.py
from pathlib import Path
import json
import shutil
import sys


def find_repo_root(start: Path) -> Path:
    """
    Walk up from ``start`` until we find a directory that contains `.git`.
    Falls back to ``start`` if no repository root is detected.
    """
    start = start.resolve()
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return start


def resolve_data_path(raw_path: str, bases: list[Path], fallback: Path) -> Path:
    """
    Interpret ``raw_path`` as absolute; otherwise try each base directory
    until an existing path is found. If nothing exists yet (fresh dataset),
    still return the fallback-based absolute path for informative logging.
    """
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    for base in bases:
        candidate = (base / path).resolve()
        if candidate.exists():
            return candidate
    return (fallback / path).resolve()


jsonl = Path(sys.argv[1]).expanduser().resolve()  # e.g., data/ds/train.jsonl
out = Path(sys.argv[2]).expanduser().resolve()    # e.g., data/sdxl_train
out.mkdir(parents=True, exist_ok=True)

repo_root = find_repo_root(Path(__file__).resolve().parent)
base_candidates: list[Path] = []
for candidate in (jsonl.parent, Path.cwd(), repo_root):
    resolved = candidate.resolve()
    if resolved not in base_candidates:
        base_candidates.append(resolved)

n = 0
missing = 0
with open(jsonl, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        img_path = row.get("image_path")
        cap = row.get("prompt")
        if not img_path or not cap:
            continue

        img = resolve_data_path(img_path, base_candidates, repo_root)
        if not img.exists():
            missing += 1
            if missing <= 5:
                print(f"[WARN] missing image for id={row.get('id')} -> {img}")
            continue

        dst_img = out / img.name
        if dst_img.resolve() != img.resolve():
            shutil.copyfile(img, dst_img)
        (out / (img.stem + ".txt")).write_text(cap, encoding="utf-8")
        n += 1

print(f"wrote {n} pairs to {out}")
if missing:
    print(f"skipped {missing} entries with missing image files")
