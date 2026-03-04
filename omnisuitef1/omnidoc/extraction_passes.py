"""5-pass extraction definitions for multi-model PDF analysis.

Each pass targets a specific aspect of engineering documents:
  1. Document Overview — type, title, revision, TOC, standards
  2. Equipment & Tags — KKS codes, pipe refs, nozzles, manufacturer data
  3. Specifications & Rules — design rules, materials, ratings, dimensions
  4. Tables & Data — complete table extraction with all rows/values
  5. Connections & Flow — pipe-to-equipment, flow paths, system boundaries

Adapted from cadAI pdf_pipeline/passes.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from omnidoc.renderer import PageViews


@dataclass
class PassDefinition:
    number: int
    name: str
    focus: str
    system_prompt: str
    user_prompt: str
    include_quadrants: bool = False
    include_zoomed: bool = False
    page_filter: str = "all"  # "all" | "tables_only" | "drawings_only"


def select_images(views: PageViews, pass_def: PassDefinition) -> List[Path]:
    images: List[Path] = []
    if views.full and views.full.exists():
        images.append(views.full)
    if pass_def.include_quadrants and views.quadrants:
        is_image_heavy = views.word_count < 50
        if is_image_heavy or pass_def.page_filter == "drawings_only":
            for qpath in views.quadrants.values():
                if qpath.exists():
                    images.append(qpath)
    if pass_def.include_zoomed and views.zoomed:
        for region in views.zoomed[:3]:
            if region.path.exists():
                images.append(region.path)
    return images


def should_process_page(views: PageViews, pass_def: PassDefinition) -> bool:
    if pass_def.page_filter == "all":
        return True
    if pass_def.page_filter == "tables_only":
        return len(views.native_tables) > 0 or len(views.zoomed) > 0
    if pass_def.page_filter == "drawings_only":
        return views.word_count < 200 or len(views.zoomed) > 0
    return True


# ── System prompt base ────────────────────────────────────────────────────

_SYSTEM_BASE = (
    "You are an expert engineering document analyst specializing in power plant "
    "piping systems, P&ID diagrams, and ASME/ANSI standards. You extract structured "
    "data from engineering documents with extreme precision. "
    "Always respond with valid JSON only — no markdown, no commentary, no explanation. "
    "If a field is not found, use null. If a list is empty, use []. "
    "Extract EVERYTHING visible — do not summarize or omit details."
)

# ── Pass 1: Document Overview ─────────────────────────────────────────────

PASS_1_OVERVIEW = PassDefinition(
    number=1, name="overview",
    focus="Document type, title, revision, date, scope, TOC, standards referenced",
    include_quadrants=False, include_zoomed=False, page_filter="all",
    system_prompt=_SYSTEM_BASE,
    user_prompt="""\
Analyze this engineering document page and extract document-level metadata.

Return JSON with this exact structure:
{
  "page_number": <int>,
  "document_type": "<design_criteria|valve_spec|pid|equipment_layout|piping_class|insulation_spec|mechanical_assembly|valve_drawing|filter_spec|general>",
  "title": "<document title if visible, null otherwise>",
  "revision": "<revision number/letter if visible, null otherwise>",
  "date": "<date if visible, null otherwise>",
  "scope": "<brief scope description if identifiable>",
  "sections": [
    {"number": "<section number>", "title": "<section title>", "page_range": [<start>, <end>]}
  ],
  "standards_referenced": ["<standard code and title>"],
  "equipment_mentioned": ["<equipment tag or KKS code>"],
  "key_topics": ["<topic>"],
  "has_drawings": <true/false>,
  "has_tables": <true/false>,
  "has_pid": <true/false>,
  "notes": "<any other relevant observations>"
}

Important:
- KKS codes look like: 10PGB11BR005, 40PGB20AC005, etc.
- Equipment tags: C1101, P1501A, E1601, etc.
- Standards: ASME B31.1, ANSI B16.5, API 610, etc.
- Pipe references: 100-B-2, 200-B-8, 50-B-9, etc.
- Extract ALL tags and references visible on this page.""",
)

# ── Pass 2: Equipment & Tags ─────────────────────────────────────────────

PASS_2_EQUIPMENT = PassDefinition(
    number=2, name="equipment",
    focus="Equipment tags, KKS codes, pipe references, nozzle IDs, manufacturer data",
    include_quadrants=True, include_zoomed=False, page_filter="all",
    system_prompt=_SYSTEM_BASE,
    user_prompt="""\
Extract ALL equipment identifiers, tags, KKS codes, pipe references, and component data from this page.

Return JSON with this exact structure:
{
  "page_number": <int>,
  "equipment": [
    {
      "tag": "<equipment tag, e.g. C1101, P1501A>",
      "type": "<vessel|pump|heat_exchanger|air_cooler|collector|filter|valve|tank|other>",
      "description": "<equipment description>",
      "kks": "<full KKS code if visible>",
      "pipe_refs": ["<connected pipe references>"],
      "nozzles": [
        {"id": "<nozzle ID>", "nps": "<nominal pipe size>", "service": "<inlet/outlet/drain/vent>"}
      ],
      "specs": {
        "manufacturer": "<manufacturer name>",
        "model": "<model number>",
        "material": "<material specification>",
        "rating": "<pressure rating>",
        "capacity": "<capacity/flow rate>"
      }
    }
  ],
  "pipe_references": [
    {
      "ref": "<pipe reference>",
      "nps": "<nominal pipe size with units>",
      "class": "<piping class>",
      "service": "<service description>",
      "from_equipment": "<source equipment tag>",
      "to_equipment": "<destination equipment tag>",
      "medium": "<fluid medium>"
    }
  ],
  "kks_codes": ["<all KKS codes found on this page>"],
  "instruments": [
    {"tag": "<instrument tag>", "type": "<FI|PI|TI|LI|etc.>", "description": "<what it measures/controls>"}
  ]
}

Important:
- KKS format: 10PGB11BR005 (system/function/component/sequential)
- Pipe refs: NPS-Material-Sequential (e.g. 100-B-2)
- Look for BOTH text labels AND drawing annotations
- Include ALL items, even partially visible ones""",
)

# ── Pass 3: Specifications & Rules ────────────────────────────────────────

PASS_3_SPECS = PassDefinition(
    number=3, name="specifications",
    focus="Design rules, dimensions, materials, pressure/temp ratings, tolerances",
    include_quadrants=True, include_zoomed=True, page_filter="all",
    system_prompt=_SYSTEM_BASE,
    user_prompt="""\
Extract ALL engineering specifications, design rules, material specifications, and dimensional data from this page.

Return JSON with this exact structure:
{
  "page_number": <int>,
  "rules": [
    {
      "id": "<rule identifier>",
      "category": "<clearance|spacing|routing|material|pressure|temperature|velocity|stress|other>",
      "description": "<what the rule specifies>",
      "value": <numeric value or null>,
      "unit": "<mm|inch|ft|bar|psi|degC|degF|m/s|etc.>",
      "condition": "<when this rule applies>",
      "reference": "<section/table/standard reference>",
      "source_standard": "<ASME B31.1, ANSI B16.5, etc.>"
    }
  ],
  "material_specs": [
    {
      "material": "<material designation>",
      "application": "<what it's used for>",
      "temp_range": "<temperature range>",
      "pressure_rating": "<pressure rating>"
    }
  ],
  "pressure_ratings": [
    {"class": "<pressure class>", "temp_c": <temp>, "pressure_bar": <pressure>, "material_group": "<group>"}
  ],
  "dimensional_data": [
    {"component": "<what>", "dimension": "<what dimension>", "nps": "<NPS>", "value": <value>, "unit": "<unit>"}
  ],
  "insulation": [
    {"service": "<service type>", "temp_range": "<range>", "thickness_mm": <thickness>, "material": "<insulation material>"}
  ]
}

Important:
- Extract EVERY numeric value with its unit
- Reference the source (section number, table number, standard)""",
)

# ── Pass 4: Tables & Data ────────────────────────────────────────────────

PASS_4_TABLES = PassDefinition(
    number=4, name="tables",
    focus="Complete table extraction — every header, row, value, and unit",
    include_quadrants=False, include_zoomed=True, page_filter="tables_only",
    system_prompt=_SYSTEM_BASE,
    user_prompt="""\
Extract EVERY table on this page with COMPLETE data — all headers, all rows, all values.

Return JSON with this exact structure:
{
  "page_number": <int>,
  "tables": [
    {
      "table_id": "<sequential ID, e.g. T1, T2>",
      "title": "<table title/caption>",
      "context": "<what section/topic this table belongs to>",
      "headers": ["<column header 1>", "<column header 2>"],
      "rows": [["<cell value>", "<cell value>"]],
      "units": {"<column_name>": "<unit>"},
      "merged_cells": ["<description of any merged cells>"],
      "notes": "<footnotes or notes below the table>",
      "row_count": <number of data rows>,
      "col_count": <number of columns>
    }
  ]
}

CRITICAL RULES:
- Extract EVERY row — do NOT summarize or skip rows
- Preserve exact numeric values — do NOT round
- Include units for every column that has them
- If a cell is empty, use "" (empty string)
- Extract values like "1/2", "2 1/2" exactly as written (fractions)""",
)

# ── Pass 5: Connections & Flow ────────────────────────────────────────────

PASS_5_CONNECTIONS = PassDefinition(
    number=5, name="connections",
    focus="Pipe-to-equipment connections, flow paths, system boundaries, nozzle connections",
    include_quadrants=True, include_zoomed=False, page_filter="drawings_only",
    system_prompt=_SYSTEM_BASE,
    user_prompt="""\
Extract ALL pipe connections, flow paths, and system connectivity from this page.

Return JSON with this exact structure:
{
  "page_number": <int>,
  "connections": [
    {
      "from_equipment": "<source equipment tag>",
      "from_nozzle": "<source nozzle ID>",
      "to_equipment": "<destination equipment tag>",
      "to_nozzle": "<destination nozzle ID>",
      "pipe_ref": "<pipe reference>",
      "nps": "<nominal pipe size>",
      "class": "<piping class>",
      "service": "<service description>",
      "flow_direction": "<from→to description>",
      "medium": "<fluid medium>"
    }
  ],
  "nozzle_connections": [
    {"equipment": "<tag>", "nozzle": "<nozzle ID>", "pipe_ref": "<pipe ref>", "service": "<inlet|outlet|drain|vent>", "nps": "<NPS>"}
  ],
  "flow_paths": [
    {"path": ["<equip1>", "<equip2>", "<equip3>"], "service": "<service>", "medium": "<fluid>", "pipe_refs": ["<refs>"]}
  ],
  "valves_inline": [
    {"tag": "<valve tag>", "type": "<gate|globe|check|butterfly|ball|control>", "pipe_ref": "<on which pipe>", "between": ["<upstream>", "<downstream>"]}
  ]
}

Important:
- Follow lines from equipment to equipment — trace the full path
- Note flow DIRECTION (arrows on P&ID)
- Identify suction vs discharge lines""",
)


# ── Registry ──────────────────────────────────────────────────────────────

ALL_PASSES: list[PassDefinition] = [
    PASS_1_OVERVIEW, PASS_2_EQUIPMENT, PASS_3_SPECS,
    PASS_4_TABLES, PASS_5_CONNECTIONS,
]

PASS_BY_NUMBER: dict[int, PassDefinition] = {p.number: p for p in ALL_PASSES}
PASS_BY_NAME: dict[str, PassDefinition] = {p.name: p for p in ALL_PASSES}


def get_pass(identifier: int | str) -> PassDefinition:
    if isinstance(identifier, int):
        if identifier not in PASS_BY_NUMBER:
            raise ValueError(f"Pass {identifier} not found. Valid: 1-5")
        return PASS_BY_NUMBER[identifier]
    if identifier not in PASS_BY_NAME:
        raise ValueError(f"Pass '{identifier}' not found. Valid: {list(PASS_BY_NAME)}")
    return PASS_BY_NAME[identifier]
