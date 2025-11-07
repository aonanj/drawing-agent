import re
from lxml import etree

FIG_RX = re.compile(r"\bFIG(?:\.|URE)?\s*\.?[-\s]*([0-9]{1,3}[A-Za-z]?)\b", re.IGNORECASE)
METHOD_RX = re.compile(r"\bmethod\b", re.IGNORECASE)
DEPENDENCY_ATTRS = {
    "claim-ref"
}


def _normalize_attr_name(name):
    """Strip namespaces and normalize separators for attribute keys."""
    if "}" in name:
        name = name.split("}", 1)[1]
    return name.lower().replace("_", "-")


def _is_independent_claim(claim_el):
    """
    Attempt to detect a claim that stands alone by checking for dependency metadata
    or explicit claim cross references. Falls back to treating claims without
    dependency hints as independent.
    """
    attr_map = {_normalize_attr_name(k): (v or "").strip().lower() for k, v in claim_el.attrib.items()}
    claim_type = attr_map.get("claim-type") or attr_map.get("claimtype") or attr_map.get("type")
    if claim_type and claim_type != "independent":
        return False
    for dep_key in DEPENDENCY_ATTRS:
        if attr_map.get(dep_key):
            return False
            
    # CHANGED: Added hasattr check to satisfy linters
    if hasattr(claim_el, "xpath"):
        claim_refs = claim_el.xpath(".//*[local-name()='claim-ref' or local-name()='claim-reference']")
        return not claim_refs
    
    # Fallback if xpath isn't present for some reason
    return True

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
    claim_elements_result = x.xpath("//claims//claim")
    if isinstance(claim_elements_result, list):
        claim_elements = [el for el in claim_elements_result if hasattr(el, "itertext")]
    elif hasattr(claim_elements_result, "itertext"):
        claim_elements = [claim_elements_result]
    else:
        claim_elements = []
        
    claims_segments = []
    method_claim_text = None
    first_independent_claim = None

    # Per user request: first_independent_claim is *always* the first claim.
    if claim_elements:
        first_independent_claim = " ".join(_textify([claim_elements[0]]))

    for claim_el in claim_elements:
        claim_text = " ".join(_textify([claim_el]))
        if not claim_text:
            continue
        claims_segments.append(claim_text)
        
        # Per user request: method_claim_text is the *first* claim
        # where "method" appears in the *first* <claim-text> tag.
        
        # CHANGED: Added hasattr check to satisfy linters
        if method_claim_text is None and hasattr(claim_el, "xpath"):
            # Find the first <claim-text> sub-element
            first_claim_text_nodes = claim_el.xpath("(.//*[local-name()='claim-text'])[1]") #type: ignore
            if first_claim_text_nodes:
                # Get the text *only* of that sub-element
                first_claim_text = " ".join(_textify(first_claim_text_nodes))
                if METHOD_RX.search(first_claim_text):
                    # If it matches, store the text of the *entire* claim
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
        "claims": claims_text,       # big flat string (all claims)
        "method_claim": method_claim_text, # Text of *first* method claim
        "first_independent_claim": first_independent_claim or "", # Text of *first* claim
        "figure_nums": sorted(set(figure_nums))  # e.g., ["1", "2", "3A"]
    }