# prompt.py
import re
from collections import Counter

# ---------- Helpers ----------
# Diagram types by signal terms
_TYPES = [
    ("flowchart", r"\b(flow|flowchart|method|flow\s*diagram|step|process|operation|block\s*([0-9]+)?)\b"),
    ("block diagram", r"\b(block\s*diagram|module|unit|processor|controller|bus|interface|CPU|GPU|memory)\b"),
    ("mechanical", r"\b(shaft|gear|housing|valve|bracket|piston|bearing|assembly|fastener|linkage|spring)\b"),
    ("electrical schematic", r"\b(resistor|capacitor|inductor|transistor|op[- ]?amp|schematic|circuit)\b"),
    ("graph", r"\b(graph|plot|chart|curve|vs\.|x[- ]?axis|y[- ]?axis)\b"),
    ("perspective", r"\b(perspective|isometric|3d|exploded)\b"),
    ("orthographic", r"\b(front\s*view|side\s*view|top\s*view|section|cross[- ]?section|plan\s*view)\b"),
]
_TYPE_RX = [(name, re.compile(pat, re.I)) for name, pat in _TYPES]

# View cues
_VIEW_RX = re.compile(
    r"\b(perspective|isometric|front\s*view|side\s*view|top\s*view|rear\s*view|section|cross[- ]?section|plan\s*view)\b",
    re.I,
)

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

def _top_k_words(text: str, k=25):
    # crude noun-ish terms by stoplist
    stop = {
        "the","a","an","of","and","or","to","in","on","for","from","with","without","into","onto",
        "is","are","be","as","by","at","over","under","between","within","this","that","those","these",
        "system","device","method","apparatus","component","unit","module","figure","fig","fig.",
        "diagram","view","example","exemplary","may","can","one","two","three","first","second","third", "accordance", "embodiment", "present", "disclosure", "invention", "figs.", "figs", "etc", "etc.", "such", "including", "shown", "includes", "used", "using", "also", "use", "different", "various", "variety"  
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

def _extract_labels(text: str, fig_label: str | None = None, limit=20):
    if not text:
        return "if present"

    def _figure_prefix() -> str | None:
        sources = [fig_label or ""]
        # Scan caption text only if label missing
        if not fig_label:
            sources.append(text)
        fig_rx = re.compile(r"\bfig(?:\.|ure)?\s*(\d+)\b", re.I)
        for source in sources:
            if not source:
                continue
            match = fig_rx.search(source)
            if match:
                return match.group(1)
        return None

    prefix = _figure_prefix()
    if not prefix:
        return "if present"

    label_rx = re.compile(rf"\b({re.escape(prefix)}[0-9]+[A-Za-z]?['′]?)\b")
    token_rx = re.compile(r"\b[0-9A-Za-z]+(?:['′])?\b")
    stop_words = {"a", "an", "the", "said", "mentioned", "aforementioned"}
    connector_stops = {"and", "or"}
    tokens = list(token_rx.finditer(text))
    if not tokens:
        return "if present"

    token_index_by_span = {(tok.start(), tok.end()): idx for idx, tok in enumerate(tokens)}

    seen, labels = set(), []
    leading_fillers = {"of", "to", "from", "in", "on", "at", "by", "for", "with", "within", "between", "into", "onto"}

    for match in label_rx.finditer(text):
        span = (match.start(), match.end())
        token_idx = token_index_by_span.get(span)
        if token_idx is None:
            continue
        parts = [match.group(1).rstrip("'′").upper()]
        i = token_idx - 1
        while i >= 0:
            tok = tokens[i]
            gap = text[tok.end(): tokens[i + 1].start()]
            if any(ch in gap for ch in ".;:?!\n"):
                break
            word = tok.group()
            lower = word.lower()
            if lower in stop_words:
                break
            if label_rx.match(word):
                break
            if lower in connector_stops:
                break
            if word.isnumeric() or any(ch.isdigit() for ch in word):
                break
            # prepend descriptor words
            parts.insert(0, word)
            i -= 1
        while len(parts) > 1 and parts[0].lower() in leading_fillers:
            parts.pop(0)
        label_text = _clean(" ".join(parts))
        key = label_text.lower()
        if key and key not in seen:
            seen.add(key)
            labels.append(label_text)
        if len(labels) >= limit:
            break

    return ", ".join(labels) if labels else "if present"

def _trim(s: str, max_len=360):
    s = _clean(s)
    return (s[: max_len - 1] + "…") if len(s) > max_len else s

# ---------- Public API ----------
def classify_diagram_type(text: str | None) -> str:
    """Return the inferred diagram type for given figure context."""
    return _classify(_clean(text or ""))

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
    diagram_type = classify_diagram_type(fig_text)
    view = _extract_view(base_text)
    objects = _top_k_words(base_text, k=14) or "{unspecified}"
    relations = _extract_relations(base_text)
    labels = _extract_labels(base_text, fig_label=fig_label)

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
        prompt += f" Constraints: {claims_subset}."
    return prompt
