#!/usr/bin/env python3
"""Extract module-level narrative docstrings from ore_algebra source files.

Module docstrings contain tutorial content, worked examples, and usage
narratives that complement the per-symbol API docs already in symbols.jsonl.
This script extracts them, splits long docstrings into RST sections, and
parses cross-references to build edges between narrative chunks and API
symbols.

Output: a JSONL file where each record is one narrative section, with
fields for text, source module, section title, and referenced symbol IDs.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

# ---------------------------------------------------------------------------
# Cross-reference parsing
# ---------------------------------------------------------------------------

# RST cross-ref directives:  :meth:`target`, :class:`~mod.Class`, etc.
RST_XREF_RE = re.compile(
    r":(?:meth|func|class|mod|attr|obj|data|exc)"
    r":`~?([^`]+)`"
)

# Sage example lines:  sage: from ore_algebra import OreAlgebra
SAGE_IMPORT_RE = re.compile(
    r"sage:\s*from\s+([\w.]+)\s+import\s+(.+)"
)

# Sage method calls:  sage: dop.numerical_solution(...)
SAGE_METHOD_CALL_RE = re.compile(
    r"sage:.*?\.([A-Za-z_][A-Za-z0-9_]*)\s*\("
)

# Sage function calls:  sage: guess([...], ...)
SAGE_FUNC_CALL_RE = re.compile(
    r"sage:.*?(?<![.\w])([A-Za-z_][A-Za-z0-9_]*)\s*\("
)

# RST section underlines
RST_SECTION_RE = re.compile(r"^(={3,}|-{3,}|~{3,}|\^{3,}|#{3,})")

# RST rubric directive
RST_RUBRIC_RE = re.compile(r"^\.\.\s+rubric::\s*(.*)")


def parse_rst_xrefs(text: str) -> list[str]:
    """Extract symbol references from RST :meth:/:class:/:func: etc."""
    refs: list[str] = []
    for match in RST_XREF_RE.finditer(text):
        target = match.group(1).strip()
        # Strip leading ~ (used for abbreviated display)
        if target.startswith("~"):
            target = target[1:]
        # Take the final dotted component path
        if target:
            refs.append(target)
    return refs


def parse_sage_refs(text: str) -> list[str]:
    """Extract symbol references from sage: example lines."""
    refs: list[str] = []
    for match in SAGE_IMPORT_RE.finditer(text):
        module = match.group(1).strip()
        imports = match.group(2).strip()
        for name in imports.split(","):
            name = name.strip()
            if name and name != "*":
                refs.append(name)
        if module and "ore_algebra" in module:
            refs.append(module)

    for match in SAGE_METHOD_CALL_RE.finditer(text):
        name = match.group(1)
        if name and not name.startswith("_"):
            refs.append(name)

    for match in SAGE_FUNC_CALL_RE.finditer(text):
        name = match.group(1)
        # Only keep names that look like ore_algebra API, not Sage builtins
        if name and name[0].isupper() and not name.startswith("_"):
            refs.append(name)

    return refs


def normalize_ref(ref: str) -> str:
    """Normalize a reference string for matching against symbol IDs."""
    ref = ref.strip()
    # Remove trailing parentheses from method refs
    if ref.endswith("()"):
        ref = ref[:-2]
    # Remove leading module prefix that matches ore_algebra.*
    parts = ref.split(".")
    # Find the shortest suffix that could be a qualname
    return ref


def resolve_refs_to_symbol_ids(
    raw_refs: list[str],
    symbol_index: dict[str, str],
    tail_index: dict[str, list[str]],
) -> list[str]:
    """Resolve raw reference strings to known symbol IDs.

    symbol_index: qualname -> symbol_id (exact match)
    tail_index: tail_name -> [symbol_ids] (suffix match)
    """
    resolved: list[str] = []
    seen: set[str] = set()

    for raw in raw_refs:
        ref = normalize_ref(raw)
        if not ref:
            continue

        # Try exact match on qualname or full ID
        lower = ref.lower()
        if lower in symbol_index:
            sid = symbol_index[lower]
            if sid not in seen:
                seen.add(sid)
                resolved.append(sid)
            continue

        # Try suffix match: ref might be "numerical_solution" matching
        # "UnivariateDifferentialOperator...numerical_solution"
        tail = ref.rsplit(".", 1)[-1].lower()
        if tail in tail_index:
            for sid in tail_index[tail]:
                if sid not in seen:
                    seen.add(sid)
                    resolved.append(sid)
            continue

        # Try dotted suffix match
        for qualname_lower, sid in symbol_index.items():
            if qualname_lower.endswith(f".{lower}") or qualname_lower.endswith(f".{tail}"):
                if sid not in seen:
                    seen.add(sid)
                    resolved.append(sid)

    return resolved


def build_symbol_lookup(symbols_path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Build lookup indexes from symbols.jsonl.

    Returns (qualname_to_id, tail_to_ids).
    """
    qualname_to_id: dict[str, str] = {}
    tail_to_ids: dict[str, list[str]] = {}

    for raw_line in symbols_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict):
            continue
        symbol_id = (rec.get("id") or "").strip()
        qualname = (rec.get("qualname") or "").strip()
        if not symbol_id:
            continue

        # Index by full ID
        qualname_to_id[symbol_id.lower()] = symbol_id
        # Index by qualname
        if qualname:
            qualname_to_id[qualname.lower()] = symbol_id
        # Index by tail (last component)
        tail = qualname.rsplit(".", 1)[-1].lower() if qualname else symbol_id.rsplit(".", 1)[-1].lower()
        if tail:
            tail_to_ids.setdefault(tail, []).append(symbol_id)

    return qualname_to_id, tail_to_ids


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

@dataclass
class NarrativeSection:
    """One section from a module-level docstring."""
    section_id: str
    module: str
    section_title: str
    text: str
    source_file: str
    raw_refs: list[str] = field(default_factory=list)
    resolved_symbol_ids: list[str] = field(default_factory=list)


def _is_section_underline(line: str) -> bool:
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    return bool(RST_SECTION_RE.match(stripped))


def _is_rubric(line: str) -> Optional[str]:
    m = RST_RUBRIC_RE.match(line.strip())
    return m.group(1).strip() if m else None


def split_docstring_into_sections(
    module: str,
    docstring: str,
    source_file: str,
    min_section_chars: int = 200,
) -> list[NarrativeSection]:
    """Split a module docstring into sections based on RST headings.

    Detects both underline-style headings and .. rubric:: directives.
    Short sections are merged into the previous section.
    """
    lines = docstring.split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_title = module  # default title for preamble
    current_lines: list[str] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check for RST rubric
        rubric_title = _is_rubric(line)
        if rubric_title:
            # Save current section if non-empty
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = rubric_title
            current_lines = []
            i += 1
            continue

        # Check for underline-style heading: title on line i, underline on i+1
        if i + 1 < len(lines) and _is_section_underline(lines[i + 1]):
            candidate_title = line.strip()
            if candidate_title and not candidate_title.startswith(".."):
                if current_lines:
                    sections.append((current_title, current_lines))
                current_title = candidate_title
                current_lines = []
                i += 2  # skip both title and underline
                continue

        # Check for underline-style heading: underline on line i, title was line i-1
        # (already handled above by lookahead)

        current_lines.append(line)
        i += 1

    # Flush final section
    if current_lines:
        sections.append((current_title, current_lines))

    # Filter out autosummary-only sections and build NarrativeSection objects
    result: list[NarrativeSection] = []
    for title, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        # Skip sections that are only autosummary directives (no real content)
        stripped = re.sub(r"\.\.\s+autosummary::.*?(?=\n\S|\Z)", "", text, flags=re.DOTALL).strip()
        stripped = re.sub(r":toctree:.*", "", stripped).strip()
        stripped = re.sub(r"^\s+ore_algebra\.\S+\s*$", "", stripped, flags=re.MULTILINE).strip()
        if len(stripped) < min_section_chars:
            # Merge short sections into previous if possible
            if result:
                prev = result[-1]
                merged_text = prev.text + "\n\n" + title + "\n" + text
                result[-1] = NarrativeSection(
                    section_id=prev.section_id,
                    module=prev.module,
                    section_title=prev.section_title,
                    text=merged_text.strip(),
                    source_file=prev.source_file,
                )
            elif text.strip():
                # First section is too short but has content — keep it
                section_id = f"narrative:{module}:{_slugify(title)}"
                result.append(NarrativeSection(
                    section_id=section_id,
                    module=module,
                    section_title=title,
                    text=text,
                    source_file=source_file,
                ))
            continue

        section_id = f"narrative:{module}:{_slugify(title)}"
        result.append(NarrativeSection(
            section_id=section_id,
            module=module,
            section_title=title,
            text=text,
            source_file=source_file,
        ))

    return result


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return slug[:60] if slug else "untitled"


# ---------------------------------------------------------------------------
# Module docstring extraction
# ---------------------------------------------------------------------------

def extract_module_docstring(path: Path) -> Optional[str]:
    """Extract the module-level docstring from a Python file via AST."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        return ast.get_docstring(tree)
    except (SyntaxError, UnicodeDecodeError):
        return None


def compute_module_name(input_root: Path, path: Path) -> str:
    """Compute dotted module name from file path relative to input root."""
    rel = path.relative_to(input_root)
    without_suffix = rel.as_posix()
    for suffix in (".py", ".pyx", ".spyx"):
        if without_suffix.endswith(suffix):
            without_suffix = without_suffix[: -len(suffix)]
            break
    parts = [p for p in without_suffix.split("/") if p]
    # __init__ becomes the parent package name
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else "ore_algebra"


def extract_all_narratives(
    input_root: Path,
    *,
    min_docstring_chars: int = 300,
    min_section_chars: int = 200,
    extensions: tuple[str, ...] = (".py",),
) -> list[NarrativeSection]:
    """Walk the source tree and extract all module-level narrative sections."""
    files = sorted(
        p for p in input_root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )

    all_sections: list[NarrativeSection] = []
    for path in files:
        doc = extract_module_docstring(path)
        if not doc or len(doc) < min_docstring_chars:
            continue

        module = compute_module_name(input_root, path)
        try:
            display_path = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            display_path = path.as_posix()

        sections = split_docstring_into_sections(
            module=module,
            docstring=doc,
            source_file=display_path,
            min_section_chars=min_section_chars,
        )
        all_sections.extend(sections)

    return all_sections


def populate_cross_references(
    sections: list[NarrativeSection],
    symbol_index: dict[str, str],
    tail_index: dict[str, list[str]],
) -> None:
    """Parse and resolve cross-references for each narrative section in-place."""
    for section in sections:
        raw_refs = parse_rst_xrefs(section.text) + parse_sage_refs(section.text)
        # Dedupe while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for ref in raw_refs:
            key = ref.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(ref)
        section.raw_refs = deduped
        section.resolved_symbol_ids = resolve_refs_to_symbol_ids(
            deduped, symbol_index, tail_index
        )


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------

def narrative_to_jsonl_record(section: NarrativeSection) -> dict[str, object]:
    return {
        "section_id": section.section_id,
        "module": section.module,
        "section_title": section.section_title,
        "text": section.text,
        "source_file": section.source_file,
        "raw_refs": section.raw_refs,
        "resolved_symbol_ids": section.resolved_symbol_ids,
        "resolved_count": len(section.resolved_symbol_ids),
    }


def write_narratives_jsonl(sections: list[NarrativeSection], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, dir=str(output_path.parent), newline=""
    ) as tmp:
        for section in sections:
            tmp.write(json.dumps(narrative_to_jsonl_record(section), ensure_ascii=True) + "\n")
        tmp_name = tmp.name
    os.replace(tmp_name, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract module-level narrative docstrings from ore_algebra source."
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Path to ore_algebra source root (e.g. ../ore_algebra-master/src/ore_algebra).",
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="Path to symbols.jsonl for cross-reference resolution.",
    )
    parser.add_argument(
        "--output",
        default="generated/module_narratives.jsonl",
        help="Output JSONL path (default: generated/module_narratives.jsonl).",
    )
    parser.add_argument(
        "--min-docstring-chars",
        type=int,
        default=300,
        help="Minimum module docstring length to include (default: 300).",
    )
    parser.add_argument(
        "--min-section-chars",
        type=int,
        default=200,
        help="Minimum section length before merging with neighbor (default: 200).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    symbols_path = Path(args.symbols).resolve()
    output_path = Path(args.output).resolve()

    if not input_root.is_dir():
        print(f"ERROR: input root not found: {input_root}", file=sys.stderr)
        return 2
    if not symbols_path.is_file():
        print(f"ERROR: symbols file not found: {symbols_path}", file=sys.stderr)
        return 2

    print(f"Input root: {input_root}")
    print(f"Symbols: {symbols_path}")

    # Step 1: Extract module-level docstrings and split into sections
    sections = extract_all_narratives(
        input_root,
        min_docstring_chars=args.min_docstring_chars,
        min_section_chars=args.min_section_chars,
    )
    print(f"Extracted {len(sections)} narrative sections from module docstrings")

    # Step 2: Build symbol lookup and resolve cross-references
    symbol_index, tail_index = build_symbol_lookup(symbols_path)
    populate_cross_references(sections, symbol_index, tail_index)
    total_refs = sum(len(s.resolved_symbol_ids) for s in sections)
    print(f"Resolved {total_refs} cross-references to known symbols")

    # Step 3: Write output
    write_narratives_jsonl(sections, output_path)
    print(f"Wrote {len(sections)} records to {output_path}")

    # Summary
    for section in sections:
        ref_count = len(section.resolved_symbol_ids)
        chars = len(section.text)
        print(f"  {section.section_id} ({chars} chars, {ref_count} refs)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
