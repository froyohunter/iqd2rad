"""Evaluate public embedding archives without loading a model."""

import argparse
import json
from pathlib import Path

from .core import evaluate_embeddings, load_embeddings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON configuration; explicit flags override values")
    parser.add_argument("--embeddings", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dims", nargs="+", type=int)
    parser.add_argument("--backends", nargs="+",
                        choices=["float", "sign", "sq2", "sq4", "sq8"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--bootstrap", type=int, metavar="DRAWS")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    options = {"dims": None, "backends": ["float", "sign", "sq2", "sq4", "sq8"],
               "batch_size": 256, "bootstrap": 0, "seed": 42}
    if args.config:
        config = json.loads(args.config.read_text())
        allowed = set(options) | {"embeddings", "output"}
        if not isinstance(config, dict) or set(config) - allowed:
            parser.error("Evaluation config has unknown options or is not an object")
        options.update(config)
    options.update({key: value for key, value in vars(args).items() if key != "config" and value is not None})
    if not options.get("embeddings") or not options.get("output"):
        parser.error("--embeddings and --output are required (or supply them in --config)")
    arrays = load_embeddings(options.pop("embeddings"))
    output = Path(options.pop("output"))
    result = evaluate_embeddings(arrays["query_embeddings"], arrays["candidate_embeddings"],
                                 arrays["correct_indices"], **options)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Evaluated {result['queries']} queries against {result['candidates']} candidates; {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
