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

_REL_VARIANTS = [
    ("connected to", r"\bconnect(?:ed|s|ing)?\s+(?:to|with)\b"),
    ("coupled to", r"\bcoupl(?:ed|es|ing)?\s+(?:to|with)\b"),
    ("attached to", r"\battach(?:ed|es|ing)?\s+(?:to|with|on)\b"),
    ("mounted to", r"\bmount(?:ed|s|ing)?\s+(?:to|on|over|above)\b"),
    ("joined to", r"\bjoin(?:ed|s|ing)?\s+(?:to|with)\b"),
    ("in communication with", r"\bin\s+communication\s+(?:with|to)\b"),
    ("fluidly connected to", r"\bfluid(?:ly)?\s+connect(?:ed|s|ing)?\s+(?:to|with)\b"),
    ("electrically connected to", r"\belectrical(?:ly)?\s+connect(?:ed|s|ing)?\s+(?:to|with)\b"),
    ("between", r"\bbetween\b"),
    ("within", r"\bwithin\b"),
    ("inside", r"\binside\b"),
    ("outside", r"\boutside\b"),
    ("adjacent to", r"\badjacent\s+(?:to|with)\b"),
    ("above", r"\babove\b"),
    ("below", r"\bbelow\b"),
]
_REL_VARIANT_RX = [(label, re.compile(pat, re.I)) for label, pat in _REL_VARIANTS]

_TOKEN_RX = re.compile(r"[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)*")
_REL_STOPWORDS = {
    "the","a","an","and","or","of","to","with","in","on","at","by","for",
    "is","are","was","were","be","been","being","that","this","these","those",
}
_REL_BOUNDARY_TOKENS = {
    "connect","connected","connection","connections","connecting","couple","coupled","couples","coupling",
    "attach","attached","attaches","attaching","mount","mounted","mounting","mounts",
    "join","joined","joining","joins","communication","communicate","communicating",
    "fluidly","electrically","between","within","inside","outside","adjacent","above","below",
}

def _trim_relation_tokens(tokens, *, tail: bool, limit: int = 4):
    tokens = list(tokens)
    if tail:
        while tokens and tokens[-1].lower() in _REL_STOPWORDS:
            tokens.pop()
        result = tokens[-limit:]
    else:
        while tokens and tokens[0].lower() in _REL_STOPWORDS:
            tokens.pop(0)
        result = tokens[:limit]
    if not tail:
        pruned = []
        for tok in result:
            if tok.lower() in _REL_BOUNDARY_TOKENS:
                break
            pruned.append(tok)
        result = pruned
    while result and result[0].lower() in _REL_STOPWORDS:
        result = result[1:]
    while result and result[-1].lower() in _REL_STOPWORDS:
        result = result[:-1]
    return result

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
    if not text:
        return "unspecified"

    sentences = re.split(r"(?<=[.;:])\s+|\n+", text)
    seen, out = set(), []

    def _add_relation(snippet: str):
        key = snippet.lower()
        if key not in seen:
            seen.add(key)
            out.append(snippet)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        for label, rx in _REL_VARIANT_RX:
            for match in rx.finditer(sentence):
                left_tokens = _TOKEN_RX.findall(sentence[: match.start()])
                right_tokens = _TOKEN_RX.findall(sentence[match.end():])
                left = " ".join(_trim_relation_tokens(left_tokens, tail=True))
                right = " ".join(_trim_relation_tokens(right_tokens, tail=False))
                if left and right:
                    snippet = f"{left} {label} {right}"
                elif left:
                    snippet = f"{left} {label}"
                elif right:
                    snippet = f"{label} {right}"
                else:
                    snippet = label
                snippet = _clean(snippet)
                if snippet:
                    _add_relation(snippet)
                    if len(out) >= limit:
                        return ", ".join(out)
        if len(out) >= limit:
            break

    if out:
        return ", ".join(out[:limit])

    # Fallback to basic phrase detection if contextual extraction failed
    rels = _REL_RX.findall(text)
    for r in rels:
        snippet = _clean(r.lower())
        if snippet:
            _add_relation(snippet)
        if len(out) >= limit:
            break
    return ", ".join(out[:limit]) if out else "unspecified"

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
        claims_cue = _trim(claims_subset, 1000)
        claim_labels = _extract_labels(claims_cue, limit=12)
        prompt += f" Constraints: {claims_cue}"
        if claim_labels != "if present":
            prompt += f" Claim labels: {claim_labels}"
    return prompt
