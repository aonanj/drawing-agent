import re
from lxml import etree

FIG_RX = re.compile(r"\bFIG(?:\.|URE)?\s*\.?[-\s]*([0-9]{1,3}[A-Za-z]?)\b", re.IGNORECASE)
METHOD_RX = re.compile(r"\bmethod\b", re.IGNORECASE)

def _textify(nodes):
    """Yield clean strings from a list of lxml nodes or raw values."""
    for n in nodes:
        if n is None:
            continue
        # element -> all descendant text
        if hasattr(n, "itertext"):
            yield " ".join(t.strip() for t in n.itertext() if t and t.strip())
            continue
        # attribute or string/bytes/nums/bools
        s = str(n)
        if s:
            yield s

def parse_doc(xml_path):
    x = etree.parse(xml_path)

    # 1) Figure paragraphs: prefer elements under <description>, then filter for FIG mentions
    desc_ps_result = x.xpath("//description//p | //description//paragraph | //description//text")
    desc_ps = list(desc_ps_result) if isinstance(desc_ps_result, list) else [desc_ps_result] if desc_ps_result else []
    fig_ps_all = list(_textify(desc_ps))
    fig_ps = fig_ps_all #[p for p in fig_ps_all if "FIG" in p.upper()]  # now safe

    # 2) Titles/captions: common locations vary across Red Book vintages
    title_nodes = x.xpath(
        "//description//brief-description-of-drawings//description-of-drawings//p | "
        "//drawings//figure//figcaption | "
        "//drawings//figure//p | "
        "//figures//figure//caption | "
        "//figures//figure//p | "
        "//drawings//p"
    )
    titles = [t for t in _textify(title_nodes) if t]

    # 3) Claims text and first method claim
    claim_elements = x.xpath("//claims//claim")
    claims_segments = []
    method_claim_text = None
    for claim_el in claim_elements:
        claim_text = " ".join(_textify([claim_el]))
        if not claim_text:
            continue
        claims_segments.append(claim_text)
        if method_claim_text is None and METHOD_RX.search(claim_text):
            method_claim_text = claim_text
    if claims_segments:
        claims_text = " ".join(claims_segments)
    else:
        claim_nodes = x.xpath("//claims//claim//claim-text | //claims//claim//p | //claims//claim")
        claims_text = " ".join(_textify(claim_nodes))

    # 4) Extract per-figure numbers from titles to aid image matching
    figure_nums = []
    for t in titles + fig_ps:
        for m in FIG_RX.finditer(t):
            figure_nums.append(m.group(1).upper())

    return {
        "figure_paras": fig_ps,      # cleaned paragraph strings that mention figures
        "titles": titles,            # cleaned captions/titles
        "claims": claims_text,       # big flat string
        "method_claim": method_claim_text,
        "figure_nums": sorted(set(figure_nums))  # e.g., ["1", "2", "3A"]
    }
