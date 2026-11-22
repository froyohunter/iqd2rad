"""Run offline IQD measurement on an explicitly aligned embedding archive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from .core import analyze_sweep, load_embeddings


def _csv_ints(text: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in text.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True, help="NPZ with query_embeddings, candidate_embeddings, correct_indices")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON sweep manifest; explicit flags override its analysis fields")
    parser.add_argument("--dims", type=_csv_ints, help="Explicit prefix dimensions, e.g. 128,256,512,2048; no native dimension is appended")
    parser.add_argument("--bits", type=_csv_ints, help="Bit widths, default 1,2,4,8,16")
    parser.add_argument("--k", type=int, help="Recall@k and density neighborhood, default 10")
    parser.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=None,
                        help="Average forward/reverse metrics for a bijective pair pool; default false")
    parser.add_argument("--block-size", type=int, default=512, help="Similarity rows per CPU block")
    parser.add_argument("--plots", action="store_true", help="Write correlation heatmaps (requires matplotlib)")
    parser.add_argument("--overwrite", action="store_true", help="Replace analysis outputs in an existing result directory")
    return parser


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> Path:
    manifest = {}
    if args.config:
        manifest = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("The IQD config must be a JSON object")
        if manifest.get("quantization", "iqd_fixed_symmetric_grid") != "iqd_fixed_symmetric_grid":
            raise ValueError("Expected quantization='iqd_fixed_symmetric_grid'")
    dims = args.dims if args.dims is not None else manifest.get("dims")
    if dims is None:
        raise ValueError("Specify --dims or a config containing an explicit dims list")
    bits = args.bits if args.bits is not None else manifest.get("bits", [1, 2, 4, 8, 16])
    k = args.k if args.k is not None else manifest.get("k", 10)
    symmetric = args.symmetric if args.symmetric is not None else manifest.get("symmetric", False)
    if not isinstance(symmetric, bool):
        raise ValueError("Config symmetric must be a JSON boolean")
    data = load_embeddings(args.embeddings)
    native_dim = manifest.get("native_dimension")
    if native_dim is not None and native_dim != data.query_embeddings.shape[1]:
        raise ValueError(f"Config expects native_dimension={native_dim}, archive has {data.query_embeddings.shape[1]}")
    expected_count = manifest.get("expected_query_count")
    if expected_count is not None and expected_count != len(data.query_embeddings):
        raise ValueError(f"Config expects {expected_count} queries, archive has {len(data.query_embeddings)}")
    summary_path = args.output_dir / "summary.json"
    if not args.overwrite and (summary_path.exists() or (args.output_dir.exists() and any(args.output_dir.glob("samples_d*_b*.npz")))):
        raise FileExistsError("Analysis outputs already exist; choose a new output directory or use --overwrite")
    plot = None
    if args.plots:
        # Check optional dependencies before writing partial analysis outputs.
        from .plotting import plot_correlations
        plot = plot_correlations
    provenance = manifest.get("encoding_provenance_status")
    if provenance == "unverified":
        print("Warning: this manifest's embedding-export provenance has not been verified; the run is an analysis, not a certified paper reproduction.", file=sys.stderr)
    summary, samples = analyze_sweep(data, dims, bits, k=k, symmetric=symmetric, block_size=args.block_size)
    summary["source"] = {"file": args.embeddings.name, "sha256": _file_hash(args.embeddings)}
    if manifest:
        summary["manifest"] = manifest
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, arrays in samples.items():
        np.savez_compressed(args.output_dir / f"samples_{key}.npz", **arrays)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if plot is not None:
        plot(summary, args.output_dir / "predictor_correlations.png")
    return summary_path


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = run(args)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        parser.error(str(exc))
    print(f"IQD results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
