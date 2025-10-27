from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FigureDescription:
    """Parsed description for a single figure."""
    figure_no: str
    description: str
    diagram_type: Optional[str] = None


@dataclass
class ClaimElement:
    """Extracted visualizable element from a claim."""
    claim_num: str
    components: List[str]
    relations: List[str]
    is_independent: bool


@dataclass
class PatentData:
    """Structured patent data extracted from XML."""
    patent_id: str
    pub_kind: str
    family_id: str
    cpc_codes: List[str]
    figure_descriptions: Dict[str, FigureDescription]
    detailed_description: str
    claims: List[ClaimElement]
    abstract: str


class USPTOXMLParser:
    """Parse USPTO patent XML files (DOCDB format)."""

    # Common diagram types to detect
    DIAGRAM_TYPES = {
        "block diagram": ["block diagram", "schematic diagram"],
        "flowchart": ["flowchart", "flow chart", "flow diagram"],
        "perspective": ["perspective view", "isometric view"],
        "exploded": ["exploded view", "exploded perspective"],
        "orthographic": ["side view", "top view", "front view", "cross-sectional", "cross section"],
        "graph": ["graph", "chart", "plot"],
        "circuit": ["circuit diagram", "circuit schematic"],
    }

    def __init__(self, xml_path: Path):
        self.xml_path = xml_path
        self.tree = None
        self.root = None

    def parse(self) -> Optional[PatentData]:
        """Parse the XML file and extract structured data."""
        try:
            self.tree = ET.parse(self.xml_path)
            self.root = self.tree.getroot()
        except ET.ParseError as e:
            logger.error("Failed to parse XML %s: %s", self.xml_path, e)
            return None

        patent_id = self._extract_patent_id()
        if not patent_id:
            logger.warning("No patent ID found in %s", self.xml_path)
            return None

        return PatentData(
            patent_id=patent_id,
            pub_kind=self._extract_pub_kind(),
            family_id=self._extract_family_id(),
            cpc_codes=self._extract_cpc_codes(),
            figure_descriptions=self._parse_figure_descriptions(),
            detailed_description=self._extract_detailed_description(),
            claims=self._parse_claims(),
            abstract=self._extract_abstract(),
        )

    def _extract_patent_id(self) -> str:
        """Extract patent publication number."""
        # Try multiple common paths
        paths = [
            ".//publication-reference//doc-number",
            ".//document-id//doc-number",
            ".//us-bibliographic-data-grant//publication-reference//doc-number",
        ]
        for path in paths:
            if self.root is not None:
                elem = self.root.find(path)
                if elem is not None and elem.text:
                    return elem.text.strip()
        return ""

    def _extract_pub_kind(self) -> str:
        """Extract publication kind (A1, B1, B2, etc.)."""
        paths = [
            ".//publication-reference//kind",
            ".//document-id//kind",
        ]
        for path in paths:
            if self.root is not None:
                elem = self.root.find(path)
                if elem is not None and elem.text:
                    return elem.text.strip()
        return ""

    def _extract_family_id(self) -> str:
        """Extract patent family ID."""
        if self.root is not None:
            elem = self.root.find(".//family-id")
            if elem is not None and elem.text:
                return elem.text.strip()
        return ""

    def _extract_cpc_codes(self) -> List[str]:
        """Extract CPC classification codes."""
        cpc_codes = []
        # Try different CPC paths
        if self.root is not None:
            # Modern USPTO XML format: <classification-cpc> with child elements
            for cpc_elem in self.root.findall(".//classification-cpc"):
                section = cpc_elem.find("section")
                class_elem = cpc_elem.find("class")
                subclass = cpc_elem.find("subclass")
                main_group = cpc_elem.find("main-group")

                if section is not None and section.text:
                    # Build CPC code from components
                    cpc_parts = [section.text.strip()]
                    if class_elem is not None and class_elem.text:
                        cpc_parts.append(class_elem.text.strip())
                    if subclass is not None and subclass.text:
                        cpc_parts.append(subclass.text.strip())
                    if main_group is not None and main_group.text:
                        cpc_parts.append(main_group.text.strip())

                    cpc_code = "".join(cpc_parts)
                    if cpc_code:
                        cpc_codes.append(cpc_code)

            # Legacy format: check for text-based classifications
            for elem in self.root.findall(".//classification-cpc//main-classification"):
                if elem.text:
                    cpc_codes.append(elem.text.strip())
            for elem in self.root.findall(".//classification-cpc//further-classification"):
                if elem.text:
                    cpc_codes.append(elem.text.strip())

        return list(set(cpc_codes))

    def _extract_abstract(self) -> str:
        """Extract patent abstract."""
        if self.root is not None:
            abstract_elem = self.root.find(".//abstract")
            if abstract_elem is not None:
                return self._extract_text_recursive(abstract_elem)
        return ""

    def _parse_figure_descriptions(self) -> Dict[str, FigureDescription]:
        """Parse <description-of-drawings> or <brief-description-of-drawings> section."""
        figures = {}
        
        # Try multiple possible paths
        drawing_desc_paths = [
            ".//description-of-drawings",
        ]
        
        drawing_desc = None
        for path in drawing_desc_paths:
            if self.root is not None:
                drawing_desc = self.root.find(path)
                if drawing_desc is not None:
                    break
        
        if drawing_desc is None:
            logger.warning("No figure descriptions found in %s", self.xml_path)
            return figures

        # Extract all paragraphs or p elements
        text = self._extract_text_recursive(drawing_desc)

        # Parse figure descriptions using multiple patterns
        # Pattern to match various figure reference formats: FIG, Fig, FIGS, Figure, Figures
        fig_ref = r"(?:FIGURES?|FIGS?\.?|Fig(?:ures?)?\.?)"

        # Pattern 1: Individual figures - "FIG. 1 is..." "Figure 2 illustrates..." "FIG. 9A shows..."
        # Handles single figures with optional letter suffix (1, 9A, 13B)
        fig_pattern1 = re.compile(
            rf"{fig_ref}\s+(\d+[A-Z]?)\s+(?:is|are|shows?|depicts?|illustrates?|represents?)\s+(.+?)(?=\s*{fig_ref}\s+\d+|$)",
            re.IGNORECASE | re.DOTALL
        )

        # Pattern 2: Grouped figures - "FIGS. 1 and 2 show..." "Figures 3-5 illustrate..."
        # Handles multiple figures mentioned together
        fig_pattern2 = re.compile(
            rf"{fig_ref}\s+([\d\s,\-andto]+)\s+(?:show|depict|illustrate|represent)s?\s+(.+?)(?=\s*{fig_ref}\s+\d+|$)",
            re.IGNORECASE | re.DOTALL
        )

        # Pattern 3: Individual figures with letter suffixes in separate tags
        # "FIG. 9 A illustrates..." or "FIG. 13 B illustrates..."
        fig_pattern3 = re.compile(
            rf"{fig_ref}\s+(\d+)\s+([A-Z])\s+(?:is|are|shows?|depicts?|illustrates?|represents?)\s+(.+?)(?=\s*{fig_ref}\s+\d+|$)",
            re.IGNORECASE | re.DOTALL
        )

        # Try pattern 1 - individual figures
        for match in fig_pattern1.finditer(text):
            fig_no = f"FIG{match.group(1).upper()}"
            description = match.group(2).strip().rstrip(";.,").strip()

            if description and len(description) > 10:  # Meaningful description
                # Detect diagram type
                diagram_type = self._detect_diagram_type(description)

                figures[fig_no] = FigureDescription(
                    figure_no=fig_no,
                    description=description,
                    diagram_type=diagram_type,
                )

        # Try pattern 3 - figures with separated letter suffixes
        for match in fig_pattern3.finditer(text):
            fig_no = f"FIG{match.group(1)}{match.group(2).upper()}"
            description = match.group(3).strip().rstrip(";.,").strip()

            if description and len(description) > 10 and fig_no not in figures:
                diagram_type = self._detect_diagram_type(description)
                figures[fig_no] = FigureDescription(
                    figure_no=fig_no,
                    description=description,
                    diagram_type=diagram_type,
                )

        # Try pattern 2 - grouped figures
        for match in fig_pattern2.finditer(text):
            # Extract individual figure numbers from "1 and 2", "1, 2, and 3", "3-5", "3A-3C"
            fig_nums_text = match.group(1)
            description = match.group(2).strip().rstrip(";.,").strip()

            # Parse out numbers and ranges
            # Handle: "1 and 2", "1, 2, and 3", "3-5", "3A-3C"
            fig_identifiers = []

            # First extract explicit numbers with optional letters
            explicit_matches = re.findall(r'\d+[A-Z]?', fig_nums_text, re.IGNORECASE)
            fig_identifiers.extend(explicit_matches)

            # Handle ranges like "3-5" or "3A-3C"
            range_matches = re.findall(r'(\d+)([A-Z]?)\s*[-to]+\s*(\d+)([A-Z]?)', fig_nums_text, re.IGNORECASE)
            for start_num, start_letter, end_num, end_letter in range_matches:
                start = int(start_num)
                end = int(end_num)
                if start_letter and end_letter:
                    # Range with letters like "3A-3C"
                    for i in range(ord(start_letter.upper()), ord(end_letter.upper()) + 1):
                        fig_identifiers.append(f"{start_num}{chr(i)}")
                else:
                    # Numeric range like "3-5"
                    for i in range(start, end + 1):
                        fig_identifiers.append(str(i))

            # Create figure entries for each identifier
            for fig_id in fig_identifiers:
                fig_no = f"FIG{fig_id.upper()}"
                if fig_no not in figures and description and len(description) > 10:
                    diagram_type = self._detect_diagram_type(description)
                    figures[fig_no] = FigureDescription(
                        figure_no=fig_no,
                        description=description,
                        diagram_type=diagram_type,
                    )

        return figures

    def _extract_detailed_description(self) -> str:
        """Extract detailed description section."""
        # Try multiple paths for detailed description
        desc_paths = [
            ".//description",
            ".//DETDESC",
            ".//detailed-description",
        ]
        
        for path in desc_paths:
            if self.root is not None:
                desc_elem = self.root.find(path)
                if desc_elem is not None:
                    return self._extract_text_recursive(desc_elem)

        return ""

    def _parse_claims(self) -> List[ClaimElement]:
        """Parse claims and extract visualizable elements."""
        claims_list = []
        claims_elem = None
        if self.root is not None:
            claims_elem = self.root.find(".//claims")
            if claims_elem is None:
                return claims_list

        if claims_elem is not None:
            for claim in claims_elem.findall(".//claim"):
                # Try to get claim number from attribute first, then from child element
                claim_num = claim.get("num", "")
                if not claim_num:
                    claim_num_elem = claim.find(".//claim-num")
                    if claim_num_elem is not None and claim_num_elem.text:
                        claim_num = claim_num_elem.text.strip()

                # Get all claim-text elements (claims can have nested structure)
                claim_text_elems = claim.findall(".//claim-text")
                if not claim_text_elems:
                    continue

                # Extract full claim text from all claim-text elements
                claim_text = " ".join(self._extract_text_recursive(elem) for elem in claim_text_elems)
                
                # Determine if independent (usually doesn't reference other claims)
                is_independent = not bool(re.search(r"claim\s+\d+", claim_text, re.IGNORECASE))
                
                # Extract components and relations
                components = self._extract_components(claim_text)
                relations = self._extract_relations(claim_text)
                
                claims_list.append(ClaimElement(
                    claim_num=claim_num,
                    components=components,
                    relations=relations,
                    is_independent=is_independent,
                ))
        
        return claims_list

    def _extract_components(self, claim_text: str) -> List[str]:
        """Extract component names from claim text."""
        components = []
        
        # Common patterns for components
        patterns = [
            r"comprising:\s*([^;]+)",  # Components after "comprising"
            r"including:\s*([^;]+)",
            r"having:\s*([^;]+)",
            r"(\w+\s+(?:module|unit|device|system|apparatus|member|element|component|circuit|processor|controller))",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, claim_text, re.IGNORECASE)
            for match in matches:
                # Clean and split
                parts = re.split(r"[,;]", match)
                for part in parts:
                    part = part.strip()
                    if part and len(part.split()) <= 5:  # Reasonable component name length
                        components.append(part)
        
        return list(set(components))[:20]  # Limit to top 20

    def _extract_relations(self, claim_text: str) -> List[str]:
        """Extract relationships between components."""
        relations = []
        
        # Common relation patterns
        relation_patterns = [
            r"connected to",
            r"coupled to",
            r"attached to",
            r"mounted on",
            r"disposed on",
            r"positioned",
            r"configured to",
            r"operatively connected",
        ]
        
        for pattern in relation_patterns:
            if re.search(pattern, claim_text, re.IGNORECASE):
                # Extract context around the relation
                matches = re.finditer(
                    rf"(\w+(?:\s+\w+){{0,2}})\s+{pattern}\s+(\w+(?:\s+\w+){{0,2}})",
                    claim_text,
                    re.IGNORECASE
                )
                for match in matches:
                    relation = f"{match.group(1)} {pattern} {match.group(2)}"
                    relations.append(relation.strip())
        
        return relations[:10]  # Limit to top 10

    def _detect_diagram_type(self, description: str) -> Optional[str]:
        """Detect diagram type from description text."""
        description_lower = description.lower()
        
        for dtype, keywords in self.DIAGRAM_TYPES.items():
            for keyword in keywords:
                if keyword in description_lower:
                    return dtype
        
        return None

    def _extract_text_recursive(self, element: ET.Element) -> str:
        """Recursively extract all text from an XML element."""
        texts = []
        
        if element.text:
            texts.append(element.text.strip())
        
        for child in element:
            texts.append(self._extract_text_recursive(child))
            if child.tail:
                texts.append(child.tail.strip())
        
        return " ".join(filter(None, texts))


def extract_figure_context(patent_data: PatentData, figure_no: str) -> Optional[str]:
    """Extract relevant context for a specific figure from detailed description."""
    if not patent_data.detailed_description:
        return None
    
    # Find text near references to this figure
    pattern = rf"{re.escape(figure_no)}[^\n]*(?:[^\n]+\n?){{0,5}}"
    matches = re.findall(pattern, patent_data.detailed_description, re.IGNORECASE)
    
    if matches:
        # Combine and clean
        context = " ".join(matches)
        context = re.sub(r"\s+", " ", context).strip()
        return context[:500]  # Limit context length
    
    return None