"""Encode public retrieval manifests into the shared NumPy exchange format."""

import argparse
import json
from pathlib import Path

import numpy as np


def read_manifest(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate IDs in manifest: {path}")
    return rows, ids


def model_input(row, root: Path, *, candidate=False):
    result = {key: row[key] for key in ("text", "instruction", "fps", "max_frames") if key in row}
    def resolve(value):
        p = Path(value).expanduser()
        return str(p if p.is_absolute() else root / p)

    if row.get("image") is not None:
        result["image"] = resolve(row["image"])
    if row.get("frames") is not None:
        result["video"] = [resolve(p) for p in row["frames"]]
    elif row.get("video") is not None:
        value = row["video"]
        result["video"] = [resolve(p) for p in value] if isinstance(value, list) else resolve(value)
    if candidate and ("image" in result or "video" in result) and result.get("text"):
        raise ValueError("Visual candidate manifests must omit caption text during retrieval evaluation")
    if not any(key in result for key in ("text", "image", "video")):
        raise ValueError(f"Input {row.get('id')} has no text/image/video")
    return result


def correct_indices(queries, candidate_ids):
    lookup = {name: index for index, name in enumerate(candidate_ids)}
    rows = []
    for query in queries:
        positive = query.get("positive_ids")
        if not isinstance(positive, list) or not positive:
            raise ValueError(f"Query {query['id']} needs a nonempty positive_ids list")
        try:
            rows.append(sorted({lookup[str(name)] for name in positive}))
        except KeyError as exc:
            raise ValueError(f"Query {query['id']} has a positive missing from the gallery: {exc}") from exc
    width = max(map(len, rows))
    if width == 1:
        return np.asarray([row[0] for row in rows], dtype=np.int64)
    result = np.full((len(rows), width), -1, dtype=np.int64)
    for i, row in enumerate(rows):
        result[i, :len(row)] = row
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True, help="Output .npz file; metadata is saved beside it")
    parser.add_argument("--base-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--revision", help="Pin the exact base model Hub revision")
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--processor-source")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--max-frames", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if output.suffix != ".npz":
        parser.error("--output must end in .npz")
    queries, qids = read_manifest(args.queries)
    candidates, cids = read_manifest(args.candidates)
    gold = correct_indices(queries, cids)
    root = Path(args.data_root).expanduser().resolve()
    qinputs = [model_input(row, root) for row in queries]
    cinputs = [model_input(row, root, candidate=True) for row in candidates]
    try:
        from .qwen import QwenEncoder
    except ImportError as exc:
        raise SystemExit("Qwen encoding needs the train extra: pip install -e '.[train]'") from exc
    encoder = QwenEncoder(
        args.base_model, revision=args.revision, adapter=args.adapter,
        adapter_revision=args.adapter_revision, processor_source=args.processor_source,
        device=args.device, torch_dtype=args.dtype, max_frames=args.max_frames,
        max_length=args.max_length, local_files_only=args.local_files_only,
    )
    # Main retrieval SQ is fitted to raw embedding prefixes. Normalization before
    # calibration changes per-channel ranges; normalize only inside each protocol.
    q = encoder.encode(qinputs, batch_size=args.batch_size, normalize=False)
    c = encoder.encode(cinputs, batch_size=args.batch_size, normalize=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, query_embeddings=q, candidate_embeddings=c,
                        correct_indices=gold, query_ids=np.asarray(qids), candidate_ids=np.asarray(cids))
    meta = {"base_model": args.base_model, "requested_revision": args.revision,
            "resolved_revision": getattr(encoder.model.config, "_commit_hash", None),
            "adapter": args.adapter, "adapter_revision": args.adapter_revision,
            "num_queries": len(q), "num_candidates": len(c), "dimension": q.shape[1],
            "normalized": False, "max_frames": args.max_frames, "max_length": args.max_length,
            "candidate_captions": False, "format_version": 1}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Saved {len(q)} queries and {len(c)} candidates to {output}")


if __name__ == "__main__":
    main()
