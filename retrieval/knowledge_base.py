#!/usr/bin/env python3
"""Shared knowledge-base profile and path helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_KNOWLEDGE_BASE_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "knowledge_base.json"
PROFILE_ENV_VAR = "ORE_ASSISTANT_KB_PROFILE"
INDEX_MODES = ("pdf", "generated", "both")


@dataclass(frozen=True)
class KnowledgeBaseProfile:
    id: str
    label: str
    description: str
    upstream_root: str
    upstream_input_root: str
    upstream_generator_script: str
    upstream_generated_dir: str
    local_generated_symbols: str
    local_generated_api_md: str
    local_narratives: str
    local_pdf_paths: tuple[str, ...]
    local_index_paths: dict[str, str]
    local_graph_path: str
    local_precondition_graph_path: str
    local_reports_dir: str
    default_source_mode: str = "both"
    include_generated_api_md_by_default: bool = True

    @property
    def repo_root(self) -> Path:
        return DEFAULT_KNOWLEDGE_BASE_CONFIG_PATH.parent.parent

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.repo_root / path
        return path.resolve()

    @property
    def upstream_root_path(self) -> Path:
        return self.resolve_path(self.upstream_root)

    @property
    def upstream_input_root_path(self) -> Path:
        return self.resolve_path(self.upstream_input_root)

    @property
    def upstream_generator_script_path(self) -> Path:
        return self.resolve_path(self.upstream_generator_script)

    @property
    def upstream_generated_dir_path(self) -> Path:
        return self.resolve_path(self.upstream_generated_dir)

    @property
    def local_generated_symbols_path(self) -> Path:
        return self.resolve_path(self.local_generated_symbols)

    @property
    def local_generated_api_md_path(self) -> Path:
        return self.resolve_path(self.local_generated_api_md)

    @property
    def local_narratives_path(self) -> Path:
        return self.resolve_path(self.local_narratives)

    @property
    def local_pdf_resolved_paths(self) -> tuple[Path, ...]:
        return tuple(self.resolve_path(path) for path in self.local_pdf_paths)

    @property
    def local_reports_dir_path(self) -> Path:
        return self.resolve_path(self.local_reports_dir)

    def local_index_path(self, mode: str | None = None) -> str:
        index_mode = str(mode or self.default_source_mode).strip().lower()
        if index_mode not in self.local_index_paths:
            raise KeyError(f"Unknown index mode: {index_mode}")
        return self.local_index_paths[index_mode]

    def local_index_resolved_path(self, mode: str | None = None) -> Path:
        return self.resolve_path(self.local_index_path(mode))

    @property
    def local_graph_resolved_path(self) -> Path:
        return self.resolve_path(self.local_graph_path)

    @property
    def local_precondition_graph_resolved_path(self) -> Path:
        return self.resolve_path(self.local_precondition_graph_path)


def _coerce_string(value: object) -> str:
    return str(value or "").strip()


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    items: list[str] = []
    for item in value:
        text = _coerce_string(item)
        if text:
            items.append(text)
    return tuple(items)


def _coerce_index_paths(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, str] = {}
    for key in INDEX_MODES:
        text = _coerce_string(value.get(key))
        if text:
            out[key] = text
    return out


@lru_cache(maxsize=4)
def load_knowledge_base_config(path: str | Path | None = None) -> dict[str, object]:
    config_path = Path(path).expanduser() if path else DEFAULT_KNOWLEDGE_BASE_CONFIG_PATH
    return json.loads(config_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def load_knowledge_base_profile(
    profile_id: str | None = None,
    *,
    config_path: str | Path | None = None,
) -> KnowledgeBaseProfile:
    payload = load_knowledge_base_config(config_path)
    requested_id = _coerce_string(profile_id or os.getenv(PROFILE_ENV_VAR) or payload.get("active_profile"))
    for item in payload.get("profiles", []):
        if not isinstance(item, dict):
            continue
        item_id = _coerce_string(item.get("id"))
        if item_id != requested_id:
            continue
        index_paths = _coerce_index_paths(item.get("local_index_paths"))
        if not index_paths:
            raise RuntimeError(f"Knowledge profile `{item_id}` is missing local_index_paths.")
        default_source_mode = _coerce_string(item.get("default_source_mode")) or "both"
        if default_source_mode not in INDEX_MODES:
            raise RuntimeError(
                f"Knowledge profile `{item_id}` has invalid default_source_mode `{default_source_mode}`."
            )
        return KnowledgeBaseProfile(
            id=item_id,
            label=_coerce_string(item.get("label")) or item_id,
            description=_coerce_string(item.get("description")),
            upstream_root=_coerce_string(item.get("upstream_root")),
            upstream_input_root=_coerce_string(item.get("upstream_input_root")),
            upstream_generator_script=_coerce_string(item.get("upstream_generator_script")),
            upstream_generated_dir=_coerce_string(item.get("upstream_generated_dir")),
            local_generated_symbols=_coerce_string(item.get("local_generated_symbols")),
            local_generated_api_md=_coerce_string(item.get("local_generated_api_md")),
            local_narratives=(
                _coerce_string(item.get("local_narratives"))
                or "generated/module_narratives.jsonl"
            ),
            local_pdf_paths=_coerce_string_tuple(item.get("local_pdf_paths")),
            local_index_paths=index_paths,
            local_graph_path=_coerce_string(item.get("local_graph_path")) or ".rag/ore_algebra_graph.json",
            local_precondition_graph_path=(
                _coerce_string(item.get("local_precondition_graph_path"))
                or ".rag/ore_algebra_precondition_graph.json"
            ),
            local_reports_dir=_coerce_string(item.get("local_reports_dir")) or "generated",
            default_source_mode=default_source_mode,
            include_generated_api_md_by_default=bool(item.get("include_generated_api_md_by_default", True)),
        )
    raise RuntimeError(f"Unknown knowledge-base profile: {requested_id or '<empty>'}")


def default_generated_symbols_path() -> str:
    return load_knowledge_base_profile().local_generated_symbols


def default_generated_api_md_path() -> str:
    return load_knowledge_base_profile().local_generated_api_md


def default_narratives_path() -> str:
    return load_knowledge_base_profile().local_narratives


def default_pdf_inputs() -> tuple[str, ...]:
    return load_knowledge_base_profile().local_pdf_paths


def default_index_path_for_mode(mode: str | None = None) -> str:
    return load_knowledge_base_profile().local_index_path(mode)


def default_reports_dir() -> str:
    return load_knowledge_base_profile().local_reports_dir


def default_graph_path() -> str:
    return load_knowledge_base_profile().local_graph_path


def default_precondition_graph_path() -> str:
    return load_knowledge_base_profile().local_precondition_graph_path
