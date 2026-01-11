#!/usr/bin/env python3
"""Combine SA-Co/VEval JSONs into a noun-phrase-only JSON.

This script scans VEval-style JSONs (e.g. `saco_veval_*_{val,test}.json`) and
extracts only the noun-phrase text annotations.

It prefers reading from top-level `video_np_pairs` entries (which include the
`noun_phrase` field). If missing, it falls back to extracting `noun_phrase`
from `annotations`.

Output is intentionally minimal: a single JSON with a sorted, unique list of
noun phrases plus basic stats.

Usage:
  python combine_veval_noun_phrases_only.py \
    --input-dir data/sa-v-text/sa-co-veval \
    --output data/sa-v-text/sa-co-veval/saco_veval_noun_phrases.json

Notes:
- The output does NOT include video names, category names, or IDs.
- Normalization is conservative: trim + collapse whitespace.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def _normalize_phrase(text: str) -> str:
    return " ".join(text.strip().split())


@dataclass(frozen=True)
class SourceStats:
    file: str
    num_phrases_seen: int
    num_unique_phrases: int


def _iter_input_files(input_dir: Path, pattern: str, output_path: Optional[Path]) -> List[Path]:
    files = sorted(p for p in input_dir.glob(pattern) if p.is_file())
    if output_path is not None:
        files = [p for p in files if p.resolve() != output_path.resolve()]

    # Keep it simple and safe: only consume the original split JSONs.
    # This avoids accidentally re-feeding already-combined outputs.
    def _is_split_file(p: Path) -> bool:
        name = p.name
        if name.startswith("combine_"):
            return False
        if name.endswith("_np_combined.json"):
            return False
        if name.endswith("_noun_phrases.json"):
            return False
        return name.endswith("_val.json") or name.endswith("_test.json")

    return [p for p in files if _is_split_file(p)]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object at top-level")
    return data


def _extract_phrases_from_video_np_pairs(data: Dict[str, Any]) -> List[str]:
    pairs = data.get("video_np_pairs")
    if not isinstance(pairs, list):
        return []

    out: List[str] = []
    for item in pairs:
        if not isinstance(item, dict):
            continue
        raw = item.get("noun_phrase")
        if not isinstance(raw, str) or not raw.strip():
            continue
        out.append(_normalize_phrase(raw))
    return out


def _extract_phrases_from_annotations(data: Dict[str, Any]) -> List[str]:
    anns = data.get("annotations")
    if not isinstance(anns, list):
        return []

    out: List[str] = []
    for item in anns:
        if not isinstance(item, dict):
            continue
        raw = item.get("noun_phrase")
        if not isinstance(raw, str) or not raw.strip():
            continue
        out.append(_normalize_phrase(raw))
    return out


def combine_noun_phrases(input_files: Iterable[Path]) -> Dict[str, Any]:
    phrases_all: Set[str] = set()
    sources: List[SourceStats] = []

    total_seen = 0

    for path in input_files:
        data = _load_json(path)

        phrases = _extract_phrases_from_video_np_pairs(data)
        if not phrases:
            phrases = _extract_phrases_from_annotations(data)

        if not phrases:
            raise ValueError(
                f"{path}: could not find any noun phrases in 'video_np_pairs' or 'annotations'"
            )

        total_seen += len(phrases)
        unique_here = set(phrases)
        phrases_all.update(unique_here)

        sources.append(
            SourceStats(
                file=path.name,
                num_phrases_seen=len(phrases),
                num_unique_phrases=len(unique_here),
            )
        )

    noun_phrases = sorted(phrases_all)

    return {
        "info": {
            "version": "v1",
            "description": "Combined SA-Co/VEval noun phrases (NP-only)",
            "sources": [
                {
                    "file": s.file,
                    "num_phrases_seen": s.num_phrases_seen,
                    "num_unique_phrases": s.num_unique_phrases,
                }
                for s in sources
            ],
        },
        "noun_phrases": noun_phrases,
        "stats": {
            "num_sources": len(sources),
            "num_phrases_seen_total": total_seen,
            "num_unique_noun_phrases": len(noun_phrases),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing saco_veval_*.json files.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="saco_veval_*.json",
        help="Glob pattern for input JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "saco_veval_noun_phrases.json",
        help="Output JSON path.",
    )
    args = parser.parse_args()

    input_files = _iter_input_files(args.input_dir, args.pattern, args.output)
    if not input_files:
        raise SystemExit(f"No input files matched {args.pattern!r} in {args.input_dir}")

    combined = combine_noun_phrases(input_files)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(combined, f)

    stats = combined.get("stats", {})
    print(
        "Wrote",
        str(args.output),
        "| sources:",
        stats.get("num_sources"),
        "| seen:",
        stats.get("num_phrases_seen_total"),
        "| unique:",
        stats.get("num_unique_noun_phrases"),
    )


if __name__ == "__main__":
    main()
