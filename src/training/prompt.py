# prompt.py
import re
from collections import Counter

# ---------- Helpers ----------
# Diagram types by signal terms
_TYPES = [
    ("flowchart", r"\b(flow|flowchart|step|process|operation|block\s*([0-9]+)?)\b"),
    ("block diagram", r"\b(block\s*diagram|module|unit|processor|controller|bus|interface)\b"),
    ("mechanical", r"\b(shaft|gear|housing|valve|bracket|piston|bearing|assembly|fastener|linkage|spring)\b"),
    ("electrical schematic", r"\b(resistor|capacitor|inductor|transistor|op[- ]?amp|schematic|circuit)\b"),
    ("graph", r"\b(graph|plot|chart|curve|vs\.|x[- ]?axis|y[- ]?axis)\b"),
    ("perspective", r"\b(perspective|isometric|3d)\b"),
    ("orthographic", r"\b(front\s*view|side\s*view|top\s*view|section|cross[- ]?section|plan\s*view)\b"),
]
_TYPE_RX = [(name, re.compile(pat, re.I)) for name, pat in _TYPES]

# View cues
_VIEW_RX = re.compile(
    r"\b(perspective|isometric|front\s*view|side\s*view|top\s*view|rear\s*view|section|cross[- ]?section|plan\s*view)\b",
    re.I,
)

# Reference numerals like “110”, “110A”, “12′”
_NUMERAL_RX = re.compile(r"\b([0-9]{1,4}[A-Za-z]?['′]?)\b")

# Simple relation phrases
_REL_PHRASES = [
    r"connected to", r"coupled to", r"attached to", r"mounted to", r"joined to",
    r"in communication with", r"fluidly connected to", r"electrically connected to",
    r"between", r"within", r"inside", r"outside", r"adjacent to", r"above", r"below",
]
_REL_RX = re.compile("|".join(rf"\b{p}\b" for p in _REL_PHRASES), re.I)

# Tokenization and cleanup
_WS_RX = re.compile(r"\s+")
_WORD_RX = re.compile(r"[A-Za-z][A-Za-z\-]+")

def _clean(s: str) -> str:
    return _WS_RX.sub(" ", (s or "").strip())

def _classify(text: str) -> str:
    t = text.lower()
    for name, rx in _TYPE_RX:
        if rx.search(t):
            return name
    return "line art"

def _extract_view(text: str) -> str:
    m = _VIEW_RX.search(text or "")
    return m.group(0).lower() if m else "unspecified"

def _top_k_words(text: str, k=14):
    # crude noun-ish terms by stoplist
    stop = {
        "the","a","an","of","and","or","to","in","on","for","from","with","without","into","onto",
        "is","are","be","as","by","at","over","under","between","within","this","that","those","these",
        "system","device","method","apparatus","component","unit","module","figure","fig","fig.",
        "diagram","view","example","exemplary","may","can","one","two","three","first","second","third"
    }
    words = [w.lower() for w in _WORD_RX.findall(text)]
    words = [w for w in words if w not in stop and len(w) > 2]
    counts = Counter(words)
    return ", ".join(w for w,_ in counts.most_common(k))

def _extract_relations(text: str, limit=4):
    rels = _REL_RX.findall(text or "")
    # unique preserve order
    seen, out = set(), []
    for r in rels:
        r = r.lower()
        if r not in seen:
            seen.add(r)
            out.append(r)
        if len(out) >= limit:
            break
    return ", ".join(out) if out else "unspecified"

def _extract_labels(text: str, limit=12):
    nums = []
    for n in _NUMERAL_RX.findall(text or ""):
        n = n.strip().rstrip("'′")
        if n and n.isascii():
            nums.append(n.upper())
    # dedupe keep order
    seen, out = set(), []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(n)
        if len(out) >= limit:
            break
    return ", ".join(out) if out else "if present"

def _trim(s: str, max_len=360):
    s = _clean(s)
    return (s[: max_len - 1] + "…") if len(s) > max_len else s

# ---------- Public API ----------
TEMPLATE = (
    "Style: USPTO patent line art, monochrome, 300dpi, white background. "
    "Figure: {fig_no}. Type: {diagram_type}. View: {view}. "
    "Objects: {objects}. Relations: {relations}. Labels: {labels}. "
    "Prohibitions: no shading, no color, no photo textures."
)

def build_prompt(fig_text: str, claims_subset: str|None = None, fig_label: str|None = None) -> str:
    """
    fig_text: caption/paragraphs referencing the figure
    claims_subset: optional claim fragments already pre-trimmed
    fig_label: OCR'd 'FIG. n' label (e.g., 'FIG. 3A'); improves Figure field
    """
    base_text = _clean(fig_text or "")
    diagram_type = _classify(base_text)
    view = _extract_view(base_text)
    objects = _top_k_words(base_text, k=14) or "{unspecified}"
    relations = _extract_relations(base_text)
    labels = _extract_labels(base_text)

    fig_no = (fig_label or "{unknown}").replace("FIGURE", "FIG.").replace("FIG ", "FIG. ").strip()
    prompt = TEMPLATE.format(
        fig_no=fig_no,
        diagram_type=diagram_type,
        view=view,
        objects=objects,
        relations=relations,
        labels=labels,
    )

    if claims_subset:
        # Add only visualizable constraints: short, trimmed, de-duplicated numerals retained
        claims_cue = _trim(claims_subset, 320)
        claim_labels = _extract_labels(claims_cue, limit=12)
        prompt += f" Constraints: {claims_cue}"
        if claim_labels != "if present":
            prompt += f" Claim labels: {claim_labels}"
    return prompt

