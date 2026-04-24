#!/usr/bin/env python3
"""Refresh local knowledge artifacts from the configured ore_algebra source profile."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.knowledge_base import PROFILE_ENV_VAR, INDEX_MODES, load_knowledge_base_profile
from retrieval.knowledge_graph import build_graph_payload_from_symbols_file, write_graph_payload
from retrieval.narrative_extractor import (
    build_symbol_lookup,
    extract_all_narratives,
    populate_cross_references,
    write_narratives_jsonl,
)
from retrieval.precondition_graph import (
    build_precondition_graph_from_file,
    write_precondition_graph,
)


def _run(command: list[str]) -> None:
    print(f"+ {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"synced {src} -> {dst}")


def _refresh_upstream_generated_docs(profile_id: str | None = None) -> None:
    profile = load_knowledge_base_profile(profile_id)
    _run(
        [
            sys.executable,
            str(profile.upstream_generator_script_path),
            "--input-root",
            str(profile.upstream_input_root_path),
            "--output-dir",
            str(profile.upstream_generated_dir_path),
        ]
    )


def _sync_local_generated_docs(profile_id: str | None = None) -> None:
    profile = load_knowledge_base_profile(profile_id)
    upstream_dir = profile.upstream_generated_dir_path
    _copy_file(upstream_dir / "symbols.jsonl", profile.local_generated_symbols_path)
    _copy_file(upstream_dir / "API_REFERENCE.md", profile.local_generated_api_md_path)


def _build_index_for_mode(
    mode: str,
    *,
    include_generated_api_md: bool,
    no_dense: bool,
    profile_id: str | None = None,
) -> None:
    profile = load_knowledge_base_profile(profile_id)
    command = [
        sys.executable,
        str(ROOT / "ore_rag_assistant.py"),
        "build-index",
        "--source-mode",
        mode,
        "--generated-symbols",
        profile.local_generated_symbols,
        "--generated-api-md",
        profile.local_generated_api_md,
        "--index-path",
        profile.local_index_path(mode),
    ]
    if mode in {"pdf", "both"}:
        for pdf_path in profile.local_pdf_paths:
            command.extend(["--pdf", pdf_path])
    if include_generated_api_md and mode in {"generated", "both"}:
        command.append("--include-generated-api-md")
    # Include narrative chunks when available
    if mode in {"generated", "both"} and profile.local_narratives_path.exists():
        command.extend(["--narratives", profile.local_narratives])
    if no_dense:
        command.append("--no-dense")
    _run(command)


def _extract_narratives(profile_id: str | None = None) -> None:
    profile = load_knowledge_base_profile(profile_id)
    input_root = profile.upstream_input_root_path
    symbols_path = profile.local_generated_symbols_path
    output_path = profile.local_narratives_path

    print(f"Extracting narratives from: {input_root}")
    sections = extract_all_narratives(input_root)
    symbol_index, tail_index = build_symbol_lookup(symbols_path)
    populate_cross_references(sections, symbol_index, tail_index)
    write_narratives_jsonl(sections, output_path)
    total_refs = sum(len(s.resolved_symbol_ids) for s in sections)
    print(f"wrote {len(sections)} narrative sections ({total_refs} cross-refs): {output_path}")


def _build_graph_artifact(profile_id: str | None = None) -> None:
    profile = load_knowledge_base_profile(profile_id)
    narratives_path = profile.local_narratives_path
    payload = build_graph_payload_from_symbols_file(
        profile.local_generated_symbols_path,
        narratives_path=narratives_path if narratives_path.exists() else None,
    )
    write_graph_payload(profile.local_graph_resolved_path, payload)
    n_narratives = len(payload.get("narratives", []))
    print(
        f"wrote graph artifact: {profile.local_graph_resolved_path}"
        + (f" (with {n_narratives} narrative nodes)" if n_narratives else "")
    )


def _build_precondition_graph_artifact(profile_id: str | None = None) -> None:
    profile = load_knowledge_base_profile(profile_id)
    graph = build_precondition_graph_from_file(profile.local_generated_symbols_path)
    target = write_precondition_graph(profile.local_precondition_graph_resolved_path, graph)
    print(f"wrote precondition graph: {target}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refresh local generated docs and retrieval indexes from the active "
            "ore_algebra knowledge-base profile."
        )
    )
    parser.add_argument(
        "--profile",
        default="",
        help=(
            "Knowledge-base profile id. Defaults to the active profile in "
            f"config/knowledge_base.json or the {PROFILE_ENV_VAR} environment variable."
        ),
    )
    parser.add_argument(
        "--skip-upstream-generate",
        action="store_true",
        help="Do not rerun the upstream rag_doc_builder generator.",
    )
    parser.add_argument(
        "--skip-generated-sync",
        action="store_true",
        help="Do not copy generated docs into this repo after the upstream refresh.",
    )
    parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Do not rebuild any local retrieval indexes.",
    )
    parser.add_argument(
        "--skip-narratives",
        action="store_true",
        help="Do not re-extract module-level narrative docstrings.",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Do not rebuild the local retrieval graph artifact.",
    )
    parser.add_argument(
        "--skip-precondition-graph",
        action="store_true",
        help="Do not rebuild the local precondition graph artifact (used by code-plan codegen).",
    )
    parser.add_argument(
        "--index-mode",
        choices=[*INDEX_MODES, "all"],
        default="both",
        help="Which local index variant to rebuild.",
    )
    parser.add_argument(
        "--include-generated-api-md",
        dest="include_generated_api_md",
        action="store_true",
        default=None,
        help="Force API_REFERENCE.md into generated/both index builds.",
    )
    parser.add_argument(
        "--no-generated-api-md",
        dest="include_generated_api_md",
        action="store_false",
        help="Skip API_REFERENCE.md even if the profile would normally include it.",
    )
    parser.add_argument(
        "--no-dense",
        action="store_true",
        help="Build lexical-only indexes.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    profile_id = args.profile or None
    profile = load_knowledge_base_profile(profile_id)
    include_generated_api_md = (
        profile.include_generated_api_md_by_default
        if args.include_generated_api_md is None
        else bool(args.include_generated_api_md)
    )

    print(f"Knowledge profile: {profile.id} ({profile.label})")
    print(f"Upstream root: {profile.upstream_root_path}")
    print(f"Local symbols: {profile.local_generated_symbols_path}")
    print(f"Local API markdown: {profile.local_generated_api_md_path}")
    print(f"Local narratives: {profile.local_narratives_path}")
    print(f"Local graph: {profile.local_graph_resolved_path}")
    print(f"Local precondition graph: {profile.local_precondition_graph_resolved_path}")
    print(f"Default index: {profile.local_index_resolved_path(profile.default_source_mode)}")

    if not args.skip_upstream_generate:
        _refresh_upstream_generated_docs(profile_id)

    if not args.skip_generated_sync:
        _sync_local_generated_docs(profile_id)

    if not args.skip_narratives:
        _extract_narratives(profile_id)

    if not args.skip_graph:
        _build_graph_artifact(profile_id)

    if not args.skip_precondition_graph:
        _build_precondition_graph_artifact(profile_id)

    if not args.skip_index:
        modes = INDEX_MODES if args.index_mode == "all" else (args.index_mode,)
        for mode in modes:
            _build_index_for_mode(
                mode,
                include_generated_api_md=include_generated_api_md,
                no_dense=args.no_dense,
                profile_id=profile_id,
            )

    print("\nReview these only if the upstream API changed structurally:")
    print("  - config/task_workflows.json")
    print("  - docs/SOURCE_BACKED_WORKFLOWS.md")
    print("  - config/workflow_retrieval_cases.json")
    print("  - config/workflow_routing_cases.json")
    print("  - docs/BENCHMARKS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
