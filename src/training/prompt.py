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
    r"connected to", r"connected with", r"coupled to", r"coupled with", r"attached to", r"mounted to", r"joined to",
    r"linked to", r"engaged with", r"supported by", r"supported on", r"in communication with", r"communicates with",
    r"fluidly connected to", r"in fluid communication with", r"electrically connected to", r"in electrical communication with",
    r"interfaces with", r"between", r"extends between", r"within", r"inside", r"outside", r"adjacent to", r"above", r"below",
]
_REL_RX = re.compile("|".join(rf"\b{p}\b" for p in _REL_PHRASES), re.I)

_REL_VARIANTS = [
    ("connected to", r"\bconnect(?:ed|s|ing)?\s+(?:to|with)\b"),
    ("connected with", r"\bconnect(?:ed|s|ing)?\s+with\b"),
    ("coupled to", r"\bcoupl(?:ed|es|ing)?\s+(?:to|with)\b"),
    ("coupled with", r"\bcoupl(?:ed|es|ing)?\s+with\b"),
    ("attached to", r"\battach(?:ed|es|ing)?\s+(?:to|with|on)\b"),
    ("mounted to", r"\bmount(?:ed|s|ing)?\s+(?:to|on|over|above)\b"),
    ("joined to", r"\bjoin(?:ed|s|ing)?\s+(?:to|with)\b"),
    ("linked to", r"\blink(?:ed|s|ing)?\s+(?:to|with)\b"),
    ("engaged with", r"\bengag(?:ed|es|ing)?\s+(?:with|to)\b"),
    ("supported by", r"\bsupport(?:ed|s|ing)?\s+(?:by|on|upon)\b"),
    ("communicates with", r"\bcommunicat(?:es|ed|ing)?\s+(?:with|to)\b"),
    ("in communication with", r"\bin\s+communication\s+(?:with|to)\b"),
    ("in fluid communication with", r"\bin\s+fluid\s+communication\s+(?:with|to)\b"),
    ("fluidly connected to", r"\bfluid(?:ly)?\s+connect(?:ed|s|ing)?\s+(?:to|with)\b"),
    ("electrically connected to", r"\belectrical(?:ly)?\s+connect(?:ed|s|ing)?\s+(?:to|with)\b"),
    ("in electrical communication with", r"\bin\s+electrical\s+communication\s+(?:with|to)\b"),
    ("interfaces with", r"\binterface(?:s|d|ing)?\s+(?:with|to)\b"),
    ("between", r"\bbetween\b"),
    ("extends between", r"\bextend(?:s|ed|ing)?\s+between\b"),
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
    "the", "a", "an", "and", "or", "of", "to", "with", "in", "on", "at", "by", "for",
    "is", "are", "was", "were", "be", "been", "being", "that", "this", "these", "those",
}
_REL_BOUNDARY_TOKENS = {
    "connect", "connected", "connection", "connections", "connecting",
    "couple", "coupled", "couples", "coupling",
    "attach", "attached", "attaches", "attaching",
    "mount", "mounted", "mounting", "mounts",
    "join", "joined", "joining", "joins",
    "link", "linked", "links", "linking",
    "engage", "engaged", "engages", "engaging",
    "support", "supported", "supports", "supporting",
    "communicate", "communicates", "communicated", "communicating",
    "communication",
    "interface", "interfaces", "interfacing", "interfaced",
    "extend", "extends", "extended", "extending",
    "fluidly", "electrically", "between", "within", "inside", "outside", "adjacent", "above", "below",
}

_ENTITY_LEADING_FILLERS = {
    "a", "an", "the", "this", "that", "these", "those", "said", "such", "another", "its", "their",
    "fig", "figs", "figure", "to", "with", "of", "for", "in", "on", "at",
}
_ENTITY_TRAILING_FILLERS = {
    "and", "or", "of", "to", "with", "for", "in", "on", "at", "respectively",
}
_ENTITY_SKIP_MIDDLE = {
    "is", "are", "was", "were", "be", "been", "being",
    "may", "can", "will", "shall", "would", "could", "should", "might", "must",
}
_CLAUSE_BREAK_TOKENS = {
    "wherein", "where", "which", "while", "when", "because", "since", "however", "thereby",
    "thereof", "therein", "thereon", "therefrom", "therewith", "thereafter", "thereto",
    "whereby", "whereof", "whereon",
    "illustrate", "illustrates", "illustrated", "illustrating",
    "depict", "depicts", "depicted", "depicting",
    "show", "shows", "showed", "showing",
    "display", "displays", "displayed", "displaying",
    "present", "presents", "presented", "presenting",
    "describe", "describes", "described", "describing",
    "via", "through",
}
_REL_PUNCT_BOUNDARIES = ".;:?!\n()[]"
_MAX_RELATION_SIDE_TOKENS = 8


def _tokenize_with_spans(sentence: str):
    return [(match.group(0), match.start(), match.end()) for match in _TOKEN_RX.finditer(sentence)]


def _gather_relation_side(tokens, sentence: str, *, boundary_pos: int, start_idx: int | None, direction: int) -> str:
    if start_idx is None or not tokens:
        return ""
    collected: list[str] = []
    idx = start_idx
    steps = 0
    while 0 <= idx < len(tokens) and steps < _MAX_RELATION_SIDE_TOKENS:
        token_text, tok_start, tok_end = tokens[idx]
        lower = token_text.lower()
        if lower in _CLAUSE_BREAK_TOKENS or lower in _REL_BOUNDARY_TOKENS:
            break
        include_token = True
        if lower in _ENTITY_SKIP_MIDDLE:
            include_token = False
        if direction == -1:
            if idx == start_idx:
                gap_start = tok_end
                gap_end = boundary_pos
            else:
                gap_start = tok_end
                gap_end = tokens[idx + 1][1] if idx + 1 < len(tokens) else boundary_pos
        else:
            if idx == start_idx:
                gap_start = boundary_pos
                gap_end = tok_start
            else:
                gap_start = tokens[idx - 1][2]
                gap_end = tok_start
        if gap_end < gap_start:
            gap_text = ""
        else:
            gap_text = sentence[gap_start:gap_end]
        if any(ch in gap_text for ch in _REL_PUNCT_BOUNDARIES):
            break
        if direction == -1 and "," in gap_text:
            break
        if include_token:
            if direction == -1:
                collected.insert(0, token_text)
            else:
                collected.append(token_text)
            steps += 1
        idx += direction
    while collected and collected[0].lower() in _ENTITY_LEADING_FILLERS:
        collected.pop(0)
    while collected and collected[-1].lower() in _ENTITY_TRAILING_FILLERS:
        collected.pop()
    while collected and collected[0].lower() in _REL_STOPWORDS:
        collected.pop(0)
    while collected and collected[-1].lower() in _REL_STOPWORDS:
        collected.pop()
    return _clean(" ".join(collected)) if collected else ""


def _split_entity_phrase(entity: str) -> list[str]:
    entity = _clean(entity)
    if not entity:
        return []
    words = entity.split()
    if len(words) <= 2:
        return [entity]
    parts = [part.strip(" ,") for part in re.split(r"\b(?:and|or)\b", entity) if part.strip(" ,")]
    if 1 < len(parts) <= 3 and any(any(ch.isdigit() for ch in part) for part in parts):
        cleaned_parts = []
        for part in parts:
            tokens = part.split()
            while tokens and tokens[0].lower() in _ENTITY_LEADING_FILLERS:
                tokens.pop(0)
            while tokens and tokens[-1].lower() in _ENTITY_TRAILING_FILLERS:
                tokens.pop()
            if tokens:
                cleaned_parts.append(" ".join(tokens))
        return cleaned_parts or [entity]
    return [entity]


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
        "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "from", "with", "without", "into", "onto",
        "is", "are", "be", "as", "by", "at", "over", "under", "between", "within", "this", "that", "those", "these",
        "system", "device", "method", "apparatus", "component", "unit", "module", "figure", "fig", "fig.",
        "diagram", "view", "example", "exemplary", "may", "can", "one", "two", "three", "first", "second", "third", "accordance", "embodiment", "present", "disclosure", "invention", "figs.", "figs", "etc", "etc.", "such", "including", "shown", "includes", "used", "using", "also", "use", "different", "various", "variety", "conceptual", "illustrating", "more", "techniques", "scenario", "which", "where", "when", "according", "combining", "aspect", "comprising",
    }
    words = [w.lower() for w in _WORD_RX.findall(text)]
    words = [w for w in words if w not in stop and len(w) > 2]
    counts = Counter(words)
    return ", ".join(w for w,_ in counts.most_common(k))

def _extract_relations(text: str, limit=4):
    if not text:
        return "unspecified"

    sentences = re.split(r"(?<=[.;:])\s+|\n+", text)
    seen: set[str] = set()
    out: list[str] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        tokens = _tokenize_with_spans(sentence)
        for label, rx in _REL_VARIANT_RX:
            for match in rx.finditer(sentence):
                left_idx: int | None = None
                for idx, (_, _, tok_end) in enumerate(tokens):
                    if tok_end <= match.start():
                        left_idx = idx
                    else:
                        break
                right_idx: int | None = None
                for idx, (_, tok_start, _) in enumerate(tokens):
                    if tok_start >= match.end():
                        right_idx = idx
                        break
                left_phrase = _gather_relation_side(tokens, sentence, boundary_pos=match.start(), start_idx=left_idx, direction=-1)
                right_phrase = _gather_relation_side(tokens, sentence, boundary_pos=match.end(), start_idx=right_idx, direction=1)
                if not left_phrase and not right_phrase:
                    continue
                left_candidates = _split_entity_phrase(left_phrase) if left_phrase else [""]
                right_candidates = _split_entity_phrase(right_phrase) if right_phrase else [""]
                for left_candidate in left_candidates:
                    for right_candidate in right_candidates:
                        parts = []
                        if left_candidate:
                            parts.append(left_candidate)
                        parts.append(label)
                        if right_candidate:
                            parts.append(right_candidate)
                        snippet = _clean(" ".join(parts))
                        key = snippet.lower()
                        if snippet and key not in seen:
                            seen.add(key)
                            out.append(snippet)
                            if len(out) >= limit:
                                return ", ".join(out[:limit])
            if len(out) >= limit:
                break
        if len(out) >= limit:
            break

    if out:
        return ", ".join(out[:limit])

    # Fallback to basic phrase detection if contextual extraction failed
    rels = _REL_RX.findall(text)
    for r in rels:
        snippet = _clean(r.lower())
        if snippet and snippet not in seen:
            seen.add(snippet)
            out.append(snippet)
        if len(out) >= limit:
            break
    return ", ".join(out[:limit]) if out else "unspecified"

def _extract_labels(text: str, fig_label: str | None = None, limit=20):
    if not text:
        return "if present"

    def _figure_prefixes() -> set[str]:
        sources = [fig_label or ""]
        if not fig_label:
            sources.append(text)
        fig_rx = re.compile(r"\bfig(?:\.|ure)?\s*(\d+)([A-Za-z]?)\b", re.I)
        prefixes: set[str] = set()
        for source in sources:
            if not source:
                continue
            match = fig_rx.search(source)
            if match:
                digits, suffix = match.group(1), match.group(2)
                prefixes.add(digits)
                if suffix:
                    prefixes.add(digits + suffix.upper())
        return prefixes

    prefixes = _figure_prefixes()
    if not prefixes:
        return "if present"

    prefix_pattern = "|".join(sorted((re.escape(p) for p in prefixes), key=len, reverse=True))
    label_rx = re.compile(
        rf"""
        \b
        (?P<label>
            (?P<prefix>{prefix_pattern})
            (?P<body>
                (?:
                    \d+[A-Za-z]*           # digits with optional trailing letters (e.g., 302, 302A)
                    |
                    [A-Za-z]+\d+           # letter-digit combos (e.g., 3A1)
                )
                (?:[-/][0-9A-Za-z]+)*     # hyphenated or slashed extensions (e.g., 302-1, 302A/302B)
            )
            (?P<prime>['′])?
        )
        \b
        """,
        re.IGNORECASE | re.VERBOSE,
    )
    token_rx = re.compile(r"\b[0-9A-Za-z]+(?:['′])?\b")
    skip_tokens = {
        "a", "an", "the", "said", "mentioned", "aforementioned",
        "is", "are", "was", "were", "be", "been", "being",
        "may", "can", "will", "shall", "would", "could", "should",
        "this", "that", "these", "those", "another", "such",
    }
    connector_stops = {"and", "or"}
    punctuation_stops = {".", ",", ";", ":", "!", "?", "(", ")", "[", "]", "{", "}", "\""}
    ordinal_suffixes = {"st", "nd", "rd", "th"}
    context_break_tokens = {
        "illustrate", "illustrates", "illustrated", "illustrating",
        "describe", "describes", "described", "describing",
        "depict", "depicts", "depicted", "depicting",
        "show", "shows", "showed", "showing",
        "include", "includes", "included", "including",
        "comprise", "comprises", "comprised", "comprising",
        "communicate", "communicates", "communicated", "communicating",
        "provide", "provides", "provided", "providing",
        "represent", "represents", "represented", "representing",
        "present", "presents", "presented", "presenting",
        "define", "defines", "defined", "defining",
    }
    tokens = list(token_rx.finditer(text))
    if not tokens:
        return "if present"

    index_by_start = {tok.start(): idx for idx, tok in enumerate(tokens)}
    seen, labels = set(), []
    leading_fillers = {"of", "to", "from", "in", "on", "at", "by", "for", "with", "within", "between", "into", "onto"}

    def _collect_context(start_idx: int, end_idx: int, display_label: str) -> str:
        parts = []
        i = start_idx - 1
        while i >= 0 and len(parts) < 4:
            tok = tokens[i]
            gap = text[tok.end(): tokens[i + 1].start()]
            if any(ch in gap for ch in ".;:?!\n"):
                break
            word = tok.group()
            lower = word.lower()
            if lower in connector_stops or lower in punctuation_stops:
                break
            if lower in context_break_tokens or lower in _REL_BOUNDARY_TOKENS:
                break
            if lower in skip_tokens:
                i -= 1
                continue
            if any(ch.isdigit() for ch in word):
                break
            parts.insert(0, word)
            i -= 1
        while len(parts) > 1 and parts and parts[0].lower() in leading_fillers:
            parts.pop(0)
        while len(parts) > 1 and parts and parts[0].lower() in skip_tokens:
            parts.pop(0)
        if parts:
            return _clean(" ".join(parts + [display_label]))

        # Try to look to the right if no meaningful left context was found.
        right_parts = []
        j = end_idx + 1
        skipped_leading = False
        while j < len(tokens) and len(right_parts) < 4:
            prev = tokens[j - 1]
            tok = tokens[j]
            gap = text[prev.end(): tok.start()]
            if any(ch in gap for ch in ".;:?!\n"):
                break
            word = tok.group()
            lower = word.lower()
            if lower in connector_stops or lower in punctuation_stops:
                break
            if lower in context_break_tokens or lower in _REL_BOUNDARY_TOKENS:
                break
            if lower in skip_tokens:
                skipped_leading = True
                j += 1
                continue
            if any(ch.isdigit() for ch in word):
                break
            right_parts.append(word)
            skipped_leading = True
            j += 1
        while right_parts and right_parts[-1].lower() in skip_tokens:
            right_parts.pop()
        if right_parts:
            return _clean(" ".join([display_label] + right_parts))
        return display_label

    for match in label_rx.finditer(text):
        prefix_part = match.group("prefix") or ""
        body_part = match.group("body") or ""
        prime_part = match.group("prime") or ""
        cleaned_label = (prefix_part + body_part)
        if len(cleaned_label) <= 1:
            continue
        if not any(ch.isdigit() for ch in body_part):
            continue
        lower_label = cleaned_label.lower()
        if any(lower_label.endswith(suffix) for suffix in ordinal_suffixes):
            continue
        start_idx = index_by_start.get(match.start())
        if start_idx is None:
            continue
        end_idx = start_idx
        label_end = match.end()
        while end_idx + 1 < len(tokens) and tokens[end_idx + 1].start() < label_end:
            end_idx += 1
        display_label = cleaned_label.upper() + (prime_part or "")
        snippet = _collect_context(start_idx, end_idx, display_label)
        key = snippet.lower()
        if key and key not in seen:
            seen.add(key)
            labels.append(snippet)
        if len(labels) >= limit:
            break

    if labels:
        return ", ".join(labels[:limit])

    # Fallback: return bare labels if context extraction failed.
    bare_labels = []
    for match in label_rx.finditer(text):
        prefix_part = match.group("prefix") or ""
        body_part = match.group("body") or ""
        prime_part = match.group("prime") or ""
        cleaned_label = (prefix_part + body_part).upper()
        if len(cleaned_label) <= 1:
            continue
        if any(cleaned_label.lower().endswith(suffix) for suffix in ordinal_suffixes):
            continue
        key = (cleaned_label + prime_part).lower()
        if key not in seen:
            seen.add(key)
            bare_labels.append(cleaned_label + prime_part)
        if len(bare_labels) >= limit:
            break
    return ", ".join(bare_labels[:limit]) if bare_labels else "if present"

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
