"""Create reproducible synthetic, non-diagonal retrieval inputs for smoke tests."""
import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(42)
    candidates = rng.normal(size=(48, 16)).astype(np.float32)
    gold = rng.permutation(48)
    queries = candidates[gold] + rng.normal(0, 0.8, size=(48, 16)).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, query_embeddings=queries,
                        candidate_embeddings=candidates, correct_indices=gold,
                        query_ids=np.asarray([f"query-{i}" for i in range(48)]),
                        candidate_ids=np.asarray([f"candidate-{i}" for i in range(48)]))
    print(args.output)


if __name__ == "__main__":
    main()
