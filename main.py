"""
Graft backend — v2 architecture

Key changes from v1:
- MidSection no longer has `owner` (prompt → Mid ownership dropped)
- MidSection gains `part_id` and `code_marker`
- MidPart is explicit and first-class (not derived from compile ownership)
- Compile endpoint uses soft targeting: focus on a part/sections, patch what needs patching
- Surgical code recompilation: regenerate one section's code block by marker
- PromptHistoryEntry: project-level audit trail of every prompt + what changed
- New .mid file format: PART blocks contain sections
- Migration: old owner-based .mid files load as a single unnamed part
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from typing import Optional
import httpx
import json
import os
import re
import hashlib
from datetime import datetime, timezone
from difflib import SequenceMatcher, unified_diff

app = FastAPI()

PROJECTS_DIR = "projects"
CONFIG_FILE  = "graft_config.json"

os.makedirs(PROJECTS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "intent_compiler": {
        "source": "ollama",
        "ollama_model": "mistral:latest",
        "ollama_url": "http://localhost:11434",
        "anthropic_model": "claude-sonnet-4-20250514",
        "anthropic_key": ""
    },
    "code_compiler": {
        "source": "ollama",
        "ollama_model": "qwen2.5-coder:3b",
        "ollama_url": "http://localhost:11434",
        "anthropic_model": "claude-sonnet-4-20250514",
        "anthropic_key": ""
    },
    "mid_verbosity": "moderate"
}

def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            cfg = json.loads(json.dumps(DEFAULT_CONFIG))
            for role in ["intent_compiler", "code_compiler"]:
                if role in saved:
                    cfg[role].update(saved[role])
            if "mid_verbosity" in saved:
                cfg["mid_verbosity"] = saved["mid_verbosity"]
            return cfg
        except Exception:
            pass
    return json.loads(json.dumps(DEFAULT_CONFIG))

def save_config(cfg: dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ── Data models ───────────────────────────────────────────────────────────────

class MidSection(BaseModel):
    id: str
    keyword: str                     # STATE | TYPE | STRUCTURE | WHEN
    name: str                        # snake_case identifier
    intent_tag: str                  # the human intent phrase
    part_id: str                     # which part this section belongs to
    code_marker: str = ""            # graft:section_name — links to code block
    body: str

class MidPartSnapshot(BaseModel):
    timestamp: str
    sections_hash: str
    sections_text: str               # serialized sections at verification time

class MidPart(BaseModel):
    id: str
    name: str
    status: str = "draft"            # draft | verified | affected
    snapshots: list[MidPartSnapshot] = []

class PromptHistoryEntry(BaseModel):
    id: str
    timestamp: str
    prompt: str
    target_part_id: Optional[str] = None    # None = global
    target_section_ids: list[str] = []
    sections_added: list[str] = []          # section names
    sections_modified: list[str] = []
    sections_removed: list[str] = []
    affected_parts: list[str] = []          # verified parts touched by watchdog

class Project(BaseModel):
    name: str
    # nodes kept for migration compatibility — not used in new architecture
    nodes: list[dict] = []
    root_id: Optional[str] = None

class CompileRequest(BaseModel):
    project_name: str
    prompt: str
    target_part_id: Optional[str] = None
    target_section_ids: list[str] = []
    existing_sections: list[MidSection] = []
    parts: list[MidPart] = []

class SurgicalCodeRequest(BaseModel):
    project_name: str
    section: MidSection
    all_sections: list[MidSection]   # for context
    current_code: str                # full current HTML file

class CompileCodeRequest(BaseModel):
    mid: str

class SaveProjectRequest(BaseModel):
    project: Project

class SaveMidRequest(BaseModel):
    project_name: str
    sections: list[MidSection]
    parts: list[MidPart]

class SavePartsRequest(BaseModel):
    project_name: str
    parts: list[MidPart]

class VerifyPartRequest(BaseModel):
    project_name: str
    part_id: str
    sections: list[MidSection]
    parts: list[MidPart]

class RevertPartRequest(BaseModel):
    project_name: str
    part_id: str
    parts: list[MidPart]
    sections: list[MidSection]

class SaveCodeRequest(BaseModel):
    project_name: str
    filename: str
    content: str

class ConfigUpdate(BaseModel):
    intent_compiler: dict
    code_compiler: dict
    mid_verbosity: Optional[str] = None

# ── New .mid file format ──────────────────────────────────────────────────────
#
# PART: part_id
# NAME: Human readable name
# STATUS: draft
#
#   WHEN: section_name
#   MARKER: graft_section_name
#   INTENT: the intent phrase
#   ---
#   Body prose.
#
# ---PART---
#
# PART: part_id_2
# ...

PART_SEP = "---PART---"

def sections_to_text(sections: list[MidSection], parts: list[MidPart] = []) -> str:
    """Serialize to new PART-grouped format."""
    # Group sections by part_id
    part_map: dict[str, list[MidSection]] = {}
    for s in sections:
        part_map.setdefault(s.part_id, []).append(s)

    # Build ordered part list
    part_order = []
    seen = set()
    for p in parts:
        if p.id in part_map:
            part_order.append(p)
            seen.add(p.id)
    # Orphaned sections with no part metadata
    for pid in part_map:
        if pid not in seen:
            part_order.append(MidPart(id=pid, name=pid))

    blocks = []
    for part in part_order:
        secs = part_map.get(part.id, [])
        if not secs:
            continue
        header = f"PART: {part.id}\nNAME: {part.name}\nSTATUS: {part.status}"
        sec_blocks = []
        for s in secs:
            marker_line = f"MARKER: {s.code_marker}" if s.code_marker else ""
            lines = [f"{s.keyword}: {s.name}"]
            if marker_line:
                lines.append(marker_line)
            lines += [f"INTENT: {s.intent_tag}", "---", s.body.strip()]
            sec_blocks.append("\n".join(lines))
        blocks.append(header + "\n\n" + "\n\n".join(sec_blocks))

    return f"\n{PART_SEP}\n\n".join(blocks)

def text_to_sections_and_parts(text: str) -> tuple[list[MidSection], list[MidPart]]:
    """Parse new PART-grouped format. Falls back to legacy owner-based format."""
    if PART_SEP in text or text.lstrip().startswith("PART:"):
        return _parse_new_format(text)
    else:
        return _parse_legacy_format(text)

def _parse_new_format(text: str) -> tuple[list[MidSection], list[MidPart]]:
    sections = []
    parts = []
    sec_index = 0
    raw_parts = re.split(r'\n' + re.escape(PART_SEP) + r'\n', text)

    for raw_part in raw_parts:
        raw_part = raw_part.strip()
        if not raw_part:
            continue

        # Extract PART header
        part_id = part_name = part_status = ""
        lines = raw_part.split('\n')
        header_end = 0
        for i, line in enumerate(lines):
            if line.startswith("PART:"):
                part_id = line[5:].strip()
            elif line.startswith("NAME:"):
                part_name = line[5:].strip()
            elif line.startswith("STATUS:"):
                part_status = line[7:].strip()
            elif line.strip() == "":
                header_end = i
                break

        if not part_id:
            continue

        part = MidPart(id=part_id, name=part_name or part_id,
                       status=part_status or "draft")
        parts.append(part)

        # Parse sections within this part
        body_text = '\n'.join(lines[header_end:])
        raw_secs = re.split(r'\n(?=(?:STATE|TYPE|STRUCTURE|WHEN):\s)', body_text.strip())
        for raw in raw_secs:
            raw = raw.strip()
            if not raw:
                continue
            s = _parse_section_block(raw, part_id, f"s{sec_index}")
            if s:
                sections.append(s)
                sec_index += 1

    return sections, parts

def _parse_legacy_format(text: str) -> tuple[list[MidSection], list[MidPart]]:
    """Load old owner-based .mid files. All sections go into a single part per owner."""
    raw_sections = re.split(r'\n(?=(?:STATE|TYPE|STRUCTURE|WHEN):\s)', text.strip())
    sections = []
    owner_names: dict[str, str] = {}

    for i, raw in enumerate(raw_sections):
        raw = raw.strip()
        if not raw:
            continue
        # Extract owner from legacy format
        owner = ""
        for line in raw.split('\n'):
            if line.startswith("OWNER:"):
                owner = line[6:].strip()
                break
        part_id = owner or "imported"
        owner_names[part_id] = part_id
        s = _parse_section_block(raw, part_id, f"s{i}")
        if s:
            sections.append(s)

    parts = [MidPart(id=pid, name=name, status="draft")
             for pid, name in owner_names.items()]
    return sections, parts

def _parse_section_block(raw: str, part_id: str, section_id: str) -> Optional[MidSection]:
    lines = raw.split('\n')
    keyword = name = intent_tag = code_marker = ''
    body_lines = []
    in_body = False

    for line in lines:
        if in_body:
            body_lines.append(line)
            continue
        m = re.match(r'^(STATE|TYPE|STRUCTURE|WHEN):\s*(.*)', line, re.IGNORECASE)
        if m and not keyword:
            keyword = m.group(1).upper()
            name = m.group(2).strip()
        elif line.startswith('MARKER:'):
            code_marker = line[7:].strip()
        elif line.startswith('INTENT:'):
            intent_tag = line[7:].strip()
        elif line.startswith('OWNER:'):
            pass  # legacy — ignored in new format
        elif line.strip() == '---':
            in_body = True
        else:
            body_lines.append(line)

    body = clean_body('\n'.join(body_lines).strip())
    if keyword and body and len(body) > 20:
        # Auto-generate marker if missing
        marker = code_marker or f"graft_{name}" if name else ""
        return MidSection(
            id=section_id, keyword=keyword, name=name,
            intent_tag=intent_tag, part_id=part_id,
            code_marker=marker, body=body
        )
    return None

def sections_for_part(part_id: str, sections: list[MidSection]) -> list[MidSection]:
    return [s for s in sections if s.part_id == part_id]

# ── Persistence ───────────────────────────────────────────────────────────────

def mid_path(project_name: str) -> str:
    safe = project_name.replace(" ", "_").lower()
    return os.path.join(PROJECTS_DIR, f"{safe}.mid")

def parts_path(project_name: str) -> str:
    safe = project_name.replace(" ", "_").lower()
    return os.path.join(PROJECTS_DIR, f"{safe}.parts.json")

def history_path(project_name: str) -> str:
    safe = project_name.replace(" ", "_").lower()
    return os.path.join(PROJECTS_DIR, f"{safe}.history.json")

def load_mid(project_name: str) -> tuple[list[MidSection], list[MidPart]]:
    path = mid_path(project_name)
    if not os.path.exists(path):
        return [], []
    with open(path, encoding="utf-8") as f:
        text = f.read()
    sections, parts_from_file = text_to_sections_and_parts(text)

    # Merge with .parts.json for snapshots/status (file format only stores status text)
    saved_parts = _load_parts_meta(project_name)
    saved_map = {p.id: p for p in saved_parts}
    merged_parts = []
    for p in parts_from_file:
        if p.id in saved_map:
            # Restore snapshots + status from meta file
            meta = saved_map[p.id]
            p.snapshots = meta.snapshots
            p.status = meta.status
        merged_parts.append(p)
    return sections, merged_parts

def save_mid(project_name: str, sections: list[MidSection], parts: list[MidPart]):
    with open(mid_path(project_name), "w", encoding="utf-8") as f:
        f.write(sections_to_text(sections, parts))
    _save_parts_meta(project_name, parts)

def _load_parts_meta(project_name: str) -> list[MidPart]:
    path = parts_path(project_name)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return [MidPart(**p) for p in json.load(f)]
    except Exception:
        return []

def _save_parts_meta(project_name: str, parts: list[MidPart]):
    with open(parts_path(project_name), "w", encoding="utf-8") as f:
        json.dump([p.dict() for p in parts], f, indent=2)

def load_history(project_name: str) -> list[PromptHistoryEntry]:
    path = history_path(project_name)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return [PromptHistoryEntry(**e) for e in json.load(f)]
    except Exception:
        return []

def append_history(project_name: str, entry: PromptHistoryEntry):
    history = load_history(project_name)
    history.append(entry)
    # Keep last 200 entries
    if len(history) > 200:
        history = history[-200:]
    with open(history_path(project_name), "w", encoding="utf-8") as f:
        json.dump([e.dict() for e in history], f, indent=2)

# ── Parts helpers ─────────────────────────────────────────────────────────────

def sections_hash(sections: list[MidSection]) -> str:
    text = sections_to_text(sections)
    return hashlib.sha256(text.encode()).hexdigest()[:16]

def sync_parts(existing_parts: list[MidPart], sections: list[MidSection]) -> list[MidPart]:
    """Ensure every part_id that has sections has a MidPart entry. Remove empty parts."""
    part_ids_with_sections = {s.part_id for s in sections if s.part_id}
    parts_by_id = {p.id: p for p in existing_parts}
    result = []
    # Keep existing parts that still have sections
    for p in existing_parts:
        if p.id in part_ids_with_sections:
            result.append(p)
    # Add any new part_ids not yet tracked
    existing_ids = {p.id for p in result}
    for pid in part_ids_with_sections:
        if pid not in existing_ids:
            result.append(MidPart(id=pid, name=pid, status="draft"))
    return result

def watchdog_check(parts: list[MidPart], new_sections: list[MidSection],
                   compiled_part_id: Optional[str]) -> list[dict]:
    """Find verified parts whose sections changed due to this compilation."""
    affected = []
    for part in parts:
        if part.id == compiled_part_id:
            continue
        if part.status != "verified" or not part.snapshots:
            continue
        last_snap = part.snapshots[-1]
        new_secs = sections_for_part(part.id, new_sections)
        new_hash = sections_hash(new_secs)
        if new_hash != last_snap.sections_hash:
            old_lines = last_snap.sections_text.splitlines(keepends=True)
            new_lines = sections_to_text(new_secs).splitlines(keepends=True)
            diff = list(unified_diff(
                old_lines, new_lines,
                fromfile=f"{part.name} (verified)",
                tofile=f"{part.name} (new)",
                lineterm=""
            ))
            affected.append({
                "part_id": part.id,
                "part_name": part.name,
                "diff": "".join(diff[:80]),
                "old_hash": last_snap.sections_hash,
                "new_hash": new_hash,
            })
    return affected

# ── Body cleanup ──────────────────────────────────────────────────────────────

FILLER_PATTERNS = [
    r'(?i)^now[,\s]+(the\s+)?(user|app|system|interface)',
    r'(?i)^at this point[,\s]',
    r'(?i)^with this[,\s]+(the\s+)?(user|app|system)',
    r'(?i)^this (means|allows|enables|lets)',
    r'(?i)^the (user|app|system) (can now|is now able)',
    r'(?i)^in (summary|conclusion|short)',
    r'(?i)^overall[,\s]',
    r'(?i)^as a result[,\s]',
    r'(?i)^to summarize[,\s]',
]
_FILLER_RE = [re.compile(p) for p in FILLER_PATTERNS]

def clean_body(text: str) -> str:
    paragraphs = re.split(r'\n{2,}', text.strip())
    cleaned = []
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?])\s+', para.strip())
        kept = [s.strip() for s in sentences
                if s.strip() and not any(p.match(s.strip()) for p in _FILLER_RE)]
        if kept:
            cleaned.append(' '.join(kept))
    return '\n\n'.join(cleaned)

# ── Verbosity ─────────────────────────────────────────────────────────────────

VERBOSITY_HINTS = {
    "minimal": (
        "SCOPE: Cover only what the prompt explicitly states. "
        "Prefer 1-2 sections. Stop as soon as the intent is covered."
    ),
    "moderate": (
        "SCOPE: Cover what the prompt states plus obvious implied requirements. "
        "Do not invent features not mentioned. Aim for 2-4 sections per part."
    ),
    "free": ""
}

# ── Prompts ───────────────────────────────────────────────────────────────────

def build_compile_prompt(
    user_prompt: str,
    target_part: Optional[MidPart],
    target_sections: list[MidSection],
    all_sections: list[MidSection],
    all_parts: list[MidPart],
    verbosity: str = "moderate"
) -> str:

    # Existing Mid summary (read-only context)
    existing_lines = []
    for part in all_parts:
        part_secs = sections_for_part(part.id, all_sections)
        if not part_secs:
            continue
        existing_lines.append(f"  PART [{part.name}] status:{part.status}")
        for s in part_secs:
            existing_lines.append(f"    {s.keyword}: {s.name} — {s.intent_tag}")
    existing_block = "\n".join(existing_lines) if existing_lines else "  (none)"

    # Verified parts constraint
    verified_lines = []
    for part in all_parts:
        if part.status == "verified" and part.snapshots:
            if target_part and part.id == target_part.id:
                continue  # target part is allowed to change
            verified_lines.append(f"  [{part.name}]")
            for s in sections_for_part(part.id, all_sections):
                verified_lines.append(f"    {s.keyword}: {s.name} — {s.intent_tag}")
    verified_block = "\n".join(verified_lines) if verified_lines else ""

    # Target context
    target_block = ""
    if target_part:
        target_secs = sections_for_part(target_part.id, all_sections)
        if target_sections:
            # Specific sections targeted
            targeted = [s for s in target_secs if s.id in {t.id for t in target_sections}]
            target_block = f"\nFOCUS: Modify the following sections in part [{target_part.name}]:\n"
            for s in targeted:
                target_block += f"  {s.keyword}: {s.name}\n  ---\n  {s.body}\n\n"
            target_block += "You may also modify other sections in this part or related sections if the prompt requires it."
        else:
            target_block = f"\nFOCUS: Modify part [{target_part.name}]. Its current sections:\n"
            for s in target_secs:
                target_block += f"  {s.keyword}: {s.name} — {s.intent_tag}\n"
            target_block += "\nYou may also touch other parts if the change logically requires it."
    else:
        target_block = "\nThis is a global prompt. Create a new part or modify whatever sections best address it."

    verbosity_hint = VERBOSITY_HINTS.get(verbosity, VERBOSITY_HINTS["moderate"])
    verbosity_block = f"\n{verbosity_hint}\n" if verbosity_hint else ""

    verified_constraint = ""
    if verified_block:
        verified_constraint = (
            f"\nVERIFIED PARTS — approved by the human. Do not redefine, contradict, "
            f"or modify these unless the prompt explicitly requires it:\n{verified_block}\n"
        )

    return f"""You are a Mid Language compiler. Mid is the human-readable specification layer between intent and code.

KEYWORDS:
STATE: data in memory — names, types, initial values. snake_case. No behavior.
TYPE: shape of one entity — fields and types. Only when STATE holds a collection.
STRUCTURE: what the user sees — layout, elements, labels. No interaction, no CSS.
WHEN: exactly one user action — trigger, precondition, state changes, feedback.

STYLE: Declarative only. No narrative. No "Now the user can...". No summaries. Every sentence is a fact.
NEVER include: colors, library names, CSS classes, API calls, file paths.
{verbosity_block}
EXISTING PROJECT MID:
{existing_block}
{verified_constraint}{target_block}

OUTPUT FORMAT — output ONLY sections that are new or changed. One block per section:

PART_ID: <short_part_label>
STATE: section_name
MARKER: graft_section_name
INTENT: one phrase describing this section
---
2 to 4 declarative sentences. No bullets. No code.

Use STATE / TYPE / STRUCTURE / WHEN as the keyword line (not "KEYWORD:").
PART_ID is a short label like "data", "ui", "actions" — not a number.
If you are creating a new part, use a descriptive PART_ID.
If you are modifying an existing section, keep its PART_ID unchanged.

To remove a section: REMOVE: section_name

Blank line between sections. No preamble. No explanation. Start immediately.

USER PROMPT: {user_prompt}
"""

def build_surgical_code_prompt(section: MidSection, all_sections: list[MidSection],
                                 current_block: str) -> str:
    # Context: neighbouring sections
    part_secs = sections_for_part(section.part_id, all_sections)
    context = "\n".join(f"  {s.keyword}: {s.name} — {s.intent_tag}"
                        for s in part_secs if s.id != section.id)
    return f"""You are a code compiler making a surgical update to one section of a web app.

Replace ONLY the code block marked // [graft:{section.name}] ... // [/graft:{section.name}]

CURRENT BLOCK TO REPLACE:
{current_block}

MID SPECIFICATION FOR THIS SECTION:
{section.keyword}: {section.name}
INTENT: {section.intent_tag}
---
{section.body}

RELATED SECTIONS (context only, do not change their code):
{context}

OUTPUT: The replacement code block only, starting with // [graft:{section.name}] and ending with // [/graft:{section.name}].
No explanation. No markdown fences.
"""

def mid_to_code_prompt(mid: str) -> str:
    return f"""You are a code compiler. Read the Mid document and produce a single complete working HTML file.

Rules:
- Output ONLY the HTML. Nothing before <!DOCTYPE html>. No markdown fences.
- All CSS inside <style> in <head>. All JS inside <script> before </body>.
- For each WHEN section, wrap the implementation:
    // [graft:section_name]
    ... code ...
    // [/graft:section_name]
- Match STRUCTURE exactly. Use STATE and TYPE for the data model.
- Dark background preferred unless Mid says otherwise.

Mid document:
{mid}

Write the complete HTML file now:
"""

# ── Parse compile output ──────────────────────────────────────────────────────

def parse_compile_output(text: str, existing_sections: list[MidSection],
                          existing_parts: list[MidPart]) -> tuple[list[MidSection], list[MidPart], list[str]]:
    """
    Parse compiler output into sections. Tolerates several model output variations:
    - Correct block format: PART_ID: x\\nKEYWORD: name\\nINTENT: ...\\n---\\nbody
    - Flat inline: PART_ID: 1 KEYWORD: STATE MARKER: x INTENT: y --- body
    - No PART_ID at all (assigns to 'global')
    - Old format with OWNER: field
    Returns (merged_sections, merged_parts, removed_section_names).
    """
    new_or_changed: list[MidSection] = []
    removed_names: list[str] = []

    # Normalise: split inline PART_ID/KEYWORD tokens onto separate lines
    # e.g. "PART_ID: 1 KEYWORD: STATE" → "PART_ID: 1\nKEYWORD: STATE"
    def normalise(t: str) -> str:
        tokens = r'(PART_ID|PART_NAME|KEYWORD|MARKER|INTENT|OWNER|STATE|TYPE|STRUCTURE|WHEN|REMOVE)'
        # Insert newline before each known token when it appears mid-line
        t = re.sub(rf'\s+({tokens}:)', r'\n\1', t)
        return t

    text = normalise(text)

    # Extract REMOVE directives
    for line in text.split('\n'):
        m = re.match(r'^REMOVE:\s*(\S+)', line.strip())
        if m:
            removed_names.append(m.group(1))

    # Split into per-section blocks — split on PART_ID: or on bare keyword lines
    # that start a new section (STATE/TYPE/STRUCTURE/WHEN at line start)
    blocks = re.split(r'\n(?=(?:PART_ID:|PART_NAME:))', text.strip())
    if len(blocks) <= 1:
        # No PART_ID markers — split on keyword lines directly
        blocks = re.split(r'\n(?=(?:STATE|TYPE|STRUCTURE|WHEN):\s)', text.strip())

    for i, block in enumerate(blocks):
        block = block.strip()
        if not block or block.startswith('REMOVE:'):
            continue

        part_id_out = part_name_out = ""
        keyword = name = intent_tag = code_marker = ""
        body_lines = []
        in_body = False

        for line in block.split('\n'):
            if in_body:
                body_lines.append(line)
                continue
            ls = line.strip()
            if ls.startswith('PART_ID:'):
                part_id_out = ls[8:].strip()
            elif ls.startswith('PART_NAME:'):
                part_name_out = ls[10:].strip()
            elif re.match(r'^(STATE|TYPE|STRUCTURE|WHEN):\s', ls, re.IGNORECASE):
                m2 = re.match(r'^(STATE|TYPE|STRUCTURE|WHEN):\s*(.*)', ls, re.IGNORECASE)
                if m2:
                    keyword = m2.group(1).upper()
                    name = m2.group(2).strip()
            elif ls.startswith('KEYWORD:'):
                # Model used "KEYWORD: STATE" instead of "STATE: name"
                kw_val = ls[8:].strip().upper()
                if kw_val in ('STATE', 'TYPE', 'STRUCTURE', 'WHEN'):
                    keyword = kw_val
            elif ls.startswith('MARKER:'):
                code_marker = ls[7:].strip()
            elif ls.startswith('INTENT:'):
                intent_tag = ls[7:].strip()
            elif ls.startswith('OWNER:'):
                part_id_out = part_id_out or re.sub(r'\W+', '_', ls[6:].strip().lower())
            elif ls == '---':
                in_body = True
            elif in_body is False and keyword and ls and not ls.startswith('#'):
                # Loose body line before --- separator
                body_lines.append(line)

        body = clean_body('\n'.join(body_lines).strip())
        if keyword and body and len(body) > 20:
            if not code_marker and name:
                code_marker = "graft_" + re.sub(r'\W+', '_', name.lower())
            pid = part_id_out or part_name_out or "global"
            new_or_changed.append(MidSection(
                id=f"new_{i}", keyword=keyword, name=name,
                intent_tag=intent_tag, part_id=pid,
                code_marker=code_marker, body=body
            ))

    # Merge: start with existing, apply changes
    # Index existing by name for update matching
    existing_by_name = {s.name: s for s in existing_sections}
    result_map: dict[str, MidSection] = dict(existing_by_name)

    # Remove
    for rname in removed_names:
        result_map.pop(rname, None)

    # Add / update
    for s in new_or_changed:
        if s.name in result_map:
            # Update — preserve id, update fields
            old = result_map[s.name]
            result_map[s.name] = MidSection(
                id=old.id, keyword=s.keyword, name=s.name,
                intent_tag=s.intent_tag, part_id=s.part_id or old.part_id,
                code_marker=s.code_marker or old.code_marker, body=s.body
            )
        else:
            # New section
            new_id = f"s{len(result_map)}"
            result_map[s.name] = MidSection(
                id=new_id, keyword=s.keyword, name=s.name,
                intent_tag=s.intent_tag, part_id=s.part_id or "global",
                code_marker=s.code_marker, body=s.body
            )

    merged_sections = list(result_map.values())

    # Re-index IDs
    for i, s in enumerate(merged_sections):
        s.id = f"s{i}"

    # Sync parts
    merged_parts = sync_parts(existing_parts, merged_sections)

    # Ensure new part_ids have a meaningful name
    # For numeric ids like "1", "2" — derive name from first section's intent
    part_ids_with_name = {p.id for p in merged_parts}
    # Build a name hint per part_id from sections
    part_name_hints: dict[str, str] = {}
    for s in merged_sections:
        if s.part_id not in part_name_hints and s.intent_tag:
            # Use first few words of intent as part name
            words = s.intent_tag.strip().split()[:5]
            part_name_hints[s.part_id] = ' '.join(words)
    for s in merged_sections:
        if s.part_id not in part_ids_with_name:
            name_hint = part_name_hints.get(s.part_id, s.part_id)
            # Sanitize numeric-only IDs
            pid = s.part_id
            if pid.isdigit():
                pid = f"part_{pid}"
                s.part_id = pid
            merged_parts.append(MidPart(id=pid, name=name_hint))
            part_ids_with_name.add(pid)

    return merged_sections, merged_parts, removed_names

# ── Streaming ─────────────────────────────────────────────────────────────────

async def stream_ollama(cfg: dict, prompt: str):
    url = cfg.get("ollama_url", "http://localhost:11434")
    model = cfg.get("ollama_model", "mistral:latest")
    full = []
    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream("POST", f"{url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": True}) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    yield f"data: {json.dumps({'error': err.decode()})}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    token = obj.get("response", "")
                    if token:
                        full.append(token)
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    if obj.get("done"):
                        break
        yield f"data: {json.dumps({'done': True, 'full': ''.join(full).strip()})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

async def stream_anthropic(cfg: dict, prompt: str):
    api_key = cfg.get("anthropic_key", "")
    model   = cfg.get("anthropic_model", "claude-sonnet-4-20250514")
    if not api_key:
        yield f"data: {json.dumps({'error': 'No Anthropic API key configured.'})}\n\n"
        return
    full = []
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST", "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model, "max_tokens": 4096, "stream": True,
                    "messages": [{"role": "user", "content": prompt}]
                }
            ) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    yield f"data: {json.dumps({'error': f'Anthropic {resp.status_code}: ' + err.decode()[:200]})}\n\n"
                    return
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        obj = json.loads(raw)
                    except Exception:
                        continue
                    if obj.get("type") == "content_block_delta":
                        token = obj.get("delta", {}).get("text", "")
                        if token:
                            full.append(token)
                            yield f"data: {json.dumps({'token': token})}\n\n"
                    if obj.get("type") == "message_stop":
                        break
        yield f"data: {json.dumps({'done': True, 'full': ''.join(full).strip()})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"

def stream_model(role_cfg: dict, prompt: str):
    if role_cfg.get("source") == "anthropic":
        return stream_anthropic(role_cfg, prompt)
    return stream_ollama(role_cfg, prompt)

# ── Routes: config ────────────────────────────────────────────────────────────

@app.get("/config")
async def get_config():
    cfg = load_config()
    for role in ["intent_compiler", "code_compiler"]:
        key = cfg[role].get("anthropic_key", "")
        cfg[role]["anthropic_key_set"] = bool(key)
        if key:
            cfg[role]["anthropic_key"] = key[:8] + "…"
    return cfg

@app.post("/config")
async def post_config(update: ConfigUpdate):
    cfg = load_config()
    for role in ["intent_compiler", "code_compiler"]:
        incoming = dict(getattr(update, role))
        if "…" in incoming.get("anthropic_key", ""):
            incoming.pop("anthropic_key", None)
        incoming.pop("anthropic_key_set", None)
        cfg[role].update(incoming)
    if update.mid_verbosity in ("minimal", "moderate", "free"):
        cfg["mid_verbosity"] = update.mid_verbosity
    save_config(cfg)
    return {"saved": True}

@app.get("/config/ollama/models")
async def list_ollama_models():
    cfg = load_config()
    urls = {cfg["intent_compiler"].get("ollama_url", "http://localhost:11434"),
            cfg["code_compiler"].get("ollama_url", "http://localhost:11434")}
    models = []
    for url in urls:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{url}/api/tags")
                if r.status_code == 200:
                    for m in r.json().get("models", []):
                        name = m.get("name", "")
                        if name and name not in models:
                            models.append(name)
        except Exception:
            pass
    return {"models": sorted(models)}

# ── Routes: compile ───────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse("index.html", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache"
    })

@app.post("/compile/mid")
async def compile_mid(req: CompileRequest):
    cfg = load_config()
    role_cfg = cfg["intent_compiler"]
    verbosity = cfg.get("mid_verbosity", "moderate")

    target_part = next((p for p in req.parts if p.id == req.target_part_id), None) \
                  if req.target_part_id else None
    target_sections = [s for s in req.existing_sections
                       if s.id in set(req.target_section_ids)]

    prompt = build_compile_prompt(
        req.prompt, target_part, target_sections,
        req.existing_sections, req.parts, verbosity
    )

    async def generate():
        full_parts = []
        async for chunk in stream_model(role_cfg, prompt):
            yield chunk
            try:
                obj = json.loads(chunk[5:].strip())
                if obj.get("token"):
                    full_parts.append(obj["token"])
                if obj.get("done"):
                    full_text = obj.get("full", "".join(full_parts)).strip()

                    merged_sections, merged_parts, removed = parse_compile_output(
                        full_text, req.existing_sections, req.parts
                    )

                    if not merged_sections and not req.existing_sections:
                        yield f"data: {json.dumps({'error': 'No valid Mid sections generated. Try rephrasing your prompt as a single clear sentence.'})}\n\n"
                        return

                    # Watchdog
                    affected = watchdog_check(merged_parts, merged_sections, req.target_part_id)
                    for p in merged_parts:
                        for aff in affected:
                            if aff["part_id"] == p.id:
                                p.status = "affected"

                    save_mid(req.project_name, merged_sections, merged_parts)

                    # Build history entry
                    old_names = {s.name for s in req.existing_sections}
                    new_names = {s.name for s in merged_sections}
                    entry = PromptHistoryEntry(
                        id=f"h{int(datetime.now(timezone.utc).timestamp()*1000)}",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        prompt=req.prompt,
                        target_part_id=req.target_part_id,
                        target_section_ids=req.target_section_ids,
                        sections_added=list(new_names - old_names),
                        sections_modified=[s.name for s in merged_sections
                                           if s.name in old_names and
                                           s.body != next((x.body for x in req.existing_sections
                                                           if x.name == s.name), s.body)],
                        sections_removed=removed,
                        affected_parts=[a["part_id"] for a in affected]
                    )
                    append_history(req.project_name, entry)

                    yield f"data: {json.dumps({'saved': True, 'sections': [s.dict() for s in merged_sections], 'parts': [p.dict() for p in merged_parts], 'affected': affected, 'history_entry': entry.dict()})}\n\n"
            except Exception:
                pass

    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/compile/code")
async def compile_code(req: CompileCodeRequest):
    cfg = load_config()
    prompt = mid_to_code_prompt(req.mid)
    return StreamingResponse(stream_model(cfg["code_compiler"], prompt),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/compile/code/surgical")
async def compile_code_surgical(req: SurgicalCodeRequest):
    """Regenerate only the code block for one Mid section."""
    cfg = load_config()
    marker = req.section.code_marker or f"graft_{req.section.name}"

    # Extract current block from code
    pattern = rf'(//\s*\[{re.escape(marker)}\].*?//\s*\[/{re.escape(marker)}\])'
    m = re.search(pattern, req.current_code, re.DOTALL)
    current_block = m.group(1) if m else f"// [{marker}]\n// [/{marker}]"

    prompt = build_surgical_code_prompt(req.section, req.all_sections, current_block)

    async def generate():
        full_parts = []
        async for chunk in stream_model(cfg["code_compiler"], prompt):
            yield chunk
            try:
                obj = json.loads(chunk[5:].strip())
                if obj.get("token"):
                    full_parts.append(obj["token"])
                if obj.get("done"):
                    new_block = obj.get("full", "".join(full_parts)).strip()
                    # Splice new block into full code
                    if m:
                        new_code = req.current_code[:m.start()] + new_block + req.current_code[m.end():]
                    else:
                        # Append before </script>
                        new_code = req.current_code.replace(
                            '</script>', f'\n{new_block}\n</script>', 1)
                    yield f"data: {json.dumps({'done': True, 'code': new_code, 'block': new_block})}\n\n"
            except Exception:
                pass

    return StreamingResponse(generate(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── Routes: parts ─────────────────────────────────────────────────────────────

@app.post("/projects/parts/verify")
async def verify_part(req: VerifyPartRequest):
    part = next((p for p in req.parts if p.id == req.part_id), None)
    if not part:
        raise HTTPException(status_code=404, detail="Part not found")
    secs = sections_for_part(req.part_id, req.sections)
    part.snapshots.append(MidPartSnapshot(
        timestamp=datetime.now(timezone.utc).isoformat(),
        sections_hash=sections_hash(secs),
        sections_text=sections_to_text(secs)
    ))
    part.status = "verified"
    _save_parts_meta(req.project_name, req.parts)
    return {"saved": True, "parts": [p.dict() for p in req.parts]}

@app.post("/projects/parts/revert")
async def revert_part(req: RevertPartRequest):
    part = next((p for p in req.parts if p.id == req.part_id), None)
    if not part or not part.snapshots:
        raise HTTPException(status_code=400, detail="No snapshot to revert to")
    last_snap = part.snapshots[-1]
    reverted, _ = text_to_sections_and_parts(last_snap.sections_text)
    # Force correct part_id
    for s in reverted:
        s.part_id = req.part_id
    merged = [s for s in req.sections if s.part_id != req.part_id] + reverted
    for i, s in enumerate(merged):
        s.id = f"s{i}"
    part.status = "verified"
    save_mid(req.project_name, merged, req.parts)
    return {"saved": True, "sections": [s.dict() for s in merged],
            "parts": [p.dict() for p in req.parts]}

# ── Routes: Mid + history ─────────────────────────────────────────────────────

@app.get("/projects/{name}/mid")
async def get_mid(name: str):
    sections, parts = load_mid(name)
    return {"sections": [s.dict() for s in sections],
            "parts": [p.dict() for p in parts],
            "text": sections_to_text(sections, parts)}

@app.post("/projects/mid")
async def post_mid(req: SaveMidRequest):
    save_mid(req.project_name, req.sections, req.parts)
    return {"saved": True}

@app.get("/projects/{name}/history")
async def get_history(name: str):
    return {"history": [e.dict() for e in load_history(name)]}

# ── Routes: code ──────────────────────────────────────────────────────────────

@app.post("/projects/code")
async def save_code(req: SaveCodeRequest):
    safe = req.project_name.replace(" ", "_").lower()
    code_dir = os.path.join(PROJECTS_DIR, safe + "_output")
    os.makedirs(code_dir, exist_ok=True)
    fname = re.sub(r'[^\w.\-]', '_', req.filename)
    with open(os.path.join(code_dir, fname), "w", encoding="utf-8") as f:
        f.write(req.content)
    return {"saved": fname}

@app.get("/projects/{name}/code")
async def list_code_files(name: str):
    safe = name.replace(" ", "_").lower()
    code_dir = os.path.join(PROJECTS_DIR, safe + "_output")
    if not os.path.exists(code_dir):
        return {"files": []}
    return {"files": [{"name": f, "size": os.path.getsize(os.path.join(code_dir, f))}
                      for f in os.listdir(code_dir)]}

@app.get("/projects/{name}_output/{filename}")
async def serve_output_file(name: str, filename: str):
    safe = name.replace(" ", "_").lower()
    path = os.path.join(PROJECTS_DIR, f"{safe}_output", filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return PlainTextResponse(content,
        media_type="text/html" if filename.endswith(".html") else "text/plain")

# ── Routes: projects ──────────────────────────────────────────────────────────

@app.get("/projects")
async def list_projects():
    files = [f.replace(".json", "") for f in os.listdir(PROJECTS_DIR)
             if f.endswith(".json") and not f.endswith(".parts.json")]
    return {"projects": sorted(files)}

@app.get("/projects/{name}")
async def load_project(name: str):
    path = os.path.join(PROJECTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Project not found")
    with open(path) as f:
        return json.load(f)

@app.post("/projects")
async def save_project(req: SaveProjectRequest):
    safe = req.project.name.replace(" ", "_").lower()
    with open(os.path.join(PROJECTS_DIR, f"{safe}.json"), "w") as f:
        json.dump(req.project.dict(), f, indent=2)
    return {"saved": safe}

@app.delete("/projects/{name}")
async def delete_project(name: str):
    for ext in [".json", ".mid", ".parts.json", ".history.json"]:
        p = os.path.join(PROJECTS_DIR, f"{name}{ext}")
        if os.path.exists(p):
            os.remove(p)
    return {"deleted": name}