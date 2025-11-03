# tools/make_captions_from_jsonl.py
from pathlib import Path
import json
import shutil
import sys

jsonl = Path(sys.argv[1])         # e.g., data/ds/train.jsonl
out   = Path(sys.argv[2])         # e.g., data/sdxl_train
out.mkdir(parents=True, exist_ok=True)

n=0
with open(jsonl, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        img = Path(row["image_path"])
        cap = row["prompt"]
        if not img.exists(): 
            continue
        # copy image
        dst_img = out / img.name
        if dst_img.resolve() != img.resolve():
            shutil.copyfile(img, dst_img)
        # write caption .txt
        (out / (img.stem + ".txt")).write_text(cap, encoding="utf-8")
        n+=1
print(f"wrote {n} pairs to {out}")
