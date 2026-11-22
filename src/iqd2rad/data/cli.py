"""Prepare local annotation files; validate splits; sample local video frames."""

import argparse
import json
from pathlib import Path

from .frames import extract_frames
from .manifests import check_disjoint, load_ids, load_jsonl, validate_retrieval
from .prepare import DATASETS, prepare_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Convert local annotations to portable manifests")
    prepare.add_argument("--config", type=Path, help="JSON preparation config; CLI flags override its values")
    prepare.add_argument("--dataset", choices=DATASETS)
    prepare.add_argument("--annotations")
    prepare.add_argument("--raw-root")
    prepare.add_argument("--output-dir")
    prepare.add_argument("--split")
    prepare.add_argument("--annotation-format", choices=["auto", "native", "mmeb", "anymodal-flickr30k"])
    prepare.add_argument("--mode", choices=["train", "retrieval"])
    prepare.add_argument("--caption-mode", choices=["auto", "all", "first", "paragraph"])
    prepare.add_argument("--query-instruction", help="Explicit retrieval instruction for native annotations")
    for flag in ("split-ids", "exclude-ids", "query-ids", "candidate-ids", "media-map", "frames-subdir"):
        prepare.add_argument(f"--{flag}")
    prepare.add_argument("--num-frames", type=int)
    prepare.add_argument("--expected-queries", type=int)
    prepare.add_argument("--expected-candidates", type=int)
    prepare.add_argument("--include-restval", action=argparse.BooleanOptionalAction, default=None)
    splits = commands.add_parser("check-splits", help="Check media-ID disjointness")
    splits.add_argument("--split", action="append", required=True, metavar="NAME=IDS_PATH")
    frames = commands.add_parser("extract-frames", help="Sample a locally acquired video with ffmpeg")
    frames.add_argument("--video", required=True)
    frames.add_argument("--output-dir", required=True)
    frames.add_argument("--num-frames", type=int, default=64)
    validate = commands.add_parser("validate", help="Validate prepared retrieval manifests and local media")
    validate.add_argument("--queries", required=True)
    validate.add_argument("--candidates", required=True)
    validate.add_argument("--raw-root", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        options = json.loads(args.config.read_text()) if args.config else {}
        if not isinstance(options, dict):
            parser.error("Preparation config must be a JSON object")
        options.update({key: value for key, value in vars(args).items()
                        if key not in {"command", "config"} and value is not None})
        missing = [key for key in ("dataset", "annotations", "raw_root", "output_dir") if not options.get(key)]
        if missing:
            parser.error(f"Missing preparation options: {', '.join(missing)}")
        result = prepare_dataset(**options)
    elif args.command == "check-splits":
        named = {}
        for definition in args.split:
            if "=" not in definition:
                parser.error("--split must have the form NAME=IDS_PATH")
            name, path = definition.split("=", 1)
            if not name or name in named:
                parser.error("Split names must be nonempty and unique")
            named[name] = load_ids(path)
        result = check_disjoint(named)
    elif args.command == "extract-frames":
        paths = extract_frames(args.video, args.output_dir, args.num_frames)
        result = {"frames": len(paths), "output_dir": args.output_dir}
    else:
        queries, candidates = load_jsonl(args.queries), load_jsonl(args.candidates)
        validate_retrieval(queries, candidates, args.raw_root)
        result = {"queries": len(queries), "candidates": len(candidates), "valid": True}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
