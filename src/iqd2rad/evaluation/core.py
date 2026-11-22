"""Public, NumPy-only retrieval evaluation. All reported metrics use [0, 1]."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape) == 0:
        raise ValueError(f"{name} must be a nonempty two-dimensional array")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def normalize(array: np.ndarray) -> np.ndarray:
    """L2 normalization with a deterministic zero-vector convention."""
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-12)


@dataclass(frozen=True)
class CandidateScalarQuantizer:
    """One affine grid per channel, fitted once using candidate vectors only.

    Transform clips out-of-range queries to the fitted range. Constant channels
    reconstruct to their candidate value. The input is the unnormalized prefix
    embedding; reconstruction is L2-normalized only when computing cosine scores.
    """

    bits: int
    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def fit(cls, candidates: np.ndarray, bits: int) -> "CandidateScalarQuantizer":
        if bits not in (2, 4, 8):
            raise ValueError("Scalar quantization supports 2, 4, or 8 bits")
        candidates = _matrix(candidates, "candidate_embeddings")
        minimum = candidates.min(axis=0).copy()
        maximum = candidates.max(axis=0).copy()
        minimum.setflags(write=False)
        maximum.setflags(write=False)
        return cls(bits, minimum, maximum)

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        embeddings = _matrix(embeddings, "embeddings")
        if embeddings.shape[1] != len(self.minimum):
            raise ValueError("Embedding dimension differs from fitted quantizer")
        levels = (1 << self.bits) - 1
        span = self.maximum - self.minimum
        scale = span / levels
        scale = np.where(scale < 1e-12, 1.0, scale)
        clipped = np.clip(embeddings, self.minimum, self.maximum)
        codes = np.rint((clipped - self.minimum) / scale)
        codes = np.clip(codes, 0, levels).astype(np.int16)
        return codes.astype(np.float32) * scale + self.minimum


def validate_correct_indices(correct_indices, n_queries: int, n_candidates: int) -> np.ndarray:
    indices = np.asarray(correct_indices)
    if indices.ndim == 1:
        indices = indices[:, None]
    if indices.ndim != 2 or indices.shape[0] != n_queries or indices.shape[1] == 0:
        raise ValueError("correct_indices must have shape (queries,) or (queries, positives)")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("correct_indices must be integers; use -1 only for padding")
    if np.any(indices < -1) or np.any(indices >= n_candidates):
        raise ValueError("correct_indices references a candidate outside the archive")
    if np.any(np.all(indices == -1, axis=1)):
        raise ValueError("Every query must have at least one positive candidate")
    for row in indices:
        valid = row[row >= 0]
        if len(np.unique(valid)) != len(valid):
            raise ValueError("A query contains duplicate positive candidate indices")
    return indices.astype(np.int64)


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    """Read a non-pickled NPZ archive; ground truth is mandatory, never diagonal."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"query_embeddings", "candidate_embeddings", "correct_indices"}
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Missing NPZ arrays: {', '.join(sorted(missing))}")
        result = {key: archive[key] for key in required}
        for key in ("query_ids", "candidate_ids"):
            if key in archive.files:
                result[key] = archive[key]
    q = _matrix(result["query_embeddings"], "query_embeddings")
    c = _matrix(result["candidate_embeddings"], "candidate_embeddings")
    if q.shape[1] != c.shape[1]:
        raise ValueError("Query and candidate embedding dimensions differ")
    validate_correct_indices(result["correct_indices"], len(q), len(c))
    for key, length in (("query_ids", len(q)), ("candidate_ids", len(c))):
        if key not in result:
            continue
        ids = result[key]
        if ids.ndim != 1 or len(ids) != length or ids.dtype.kind != "U":
            raise ValueError(f"{key} must be a one-dimensional Unicode array of length {length}")
        if len(set(ids.tolist())) != length or any(not value for value in ids):
            raise ValueError(f"{key} must be nonempty and unique")
    result["query_embeddings"], result["candidate_embeddings"] = q, c
    return result


def per_query_metrics(ranked: np.ndarray, correct_indices: np.ndarray) -> dict[str, np.ndarray]:
    """Hit-rate Recall@K and binary-relevance NDCG@10 for each query."""
    positives = np.asarray(correct_indices)
    if positives.ndim == 1:
        positives = positives[:, None]
    hits = np.any(ranked[:, :, None] == positives[:, None, :], axis=2)
    result = {f"R@{k}": hits[:, :k].any(axis=1).astype(float) for k in (1, 5, 10)}
    depth = min(10, ranked.shape[1])
    discounts = 1.0 / np.log2(np.arange(depth) + 2)
    dcg = (hits[:, :depth] * discounts).sum(axis=1)
    n_relevant = np.minimum((positives >= 0).sum(axis=1), depth)
    ideal = np.array([discounts[:n].sum() for n in n_relevant])
    result["NDCG@10"] = dcg / ideal
    return result


def _bootstrap(values: dict[str, np.ndarray], draws: int, seed: int) -> dict:
    if draws <= 0:
        raise ValueError("Bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    keys = list(values)
    data = np.stack([values[k] for k in keys], axis=1)
    means = np.empty((draws, len(keys)), dtype=float)
    # One draw at a time avoids allocating draws * n_queries index arrays.
    for index in range(draws):
        sample = rng.integers(0, len(data), size=len(data))
        means[index] = data[sample].mean(axis=0)
    return {key: [float(v) for v in np.quantile(means[:, col], [0.025, 0.975])]
            for col, key in enumerate(keys)}


def evaluate_embeddings(query_embeddings: np.ndarray, candidate_embeddings: np.ndarray,
                        correct_indices: np.ndarray, dims: Iterable[int] | None = None,
                        backends: Iterable[str] = ("float", "sign", "sq2", "sq4", "sq8"),
                        batch_size: int = 256, bootstrap: int = 0, seed: int = 42) -> dict:
    q = _matrix(query_embeddings, "query_embeddings")
    c = _matrix(candidate_embeddings, "candidate_embeddings")
    if q.shape[1] != c.shape[1]:
        raise ValueError("Query and candidate embedding dimensions differ")
    gt = validate_correct_indices(correct_indices, len(q), len(c))
    dims = list(dims if dims is not None else [q.shape[1]])
    backends = list(backends)
    if not dims or any(not isinstance(d, (int, np.integer)) or d < 1 or d > q.shape[1] for d in dims):
        raise ValueError("Each requested dimension must be in [1, embedding dimension]")
    if not backends or any(b not in {"float", "sign", "sq2", "sq4", "sq8"} for b in backends):
        raise ValueError("Backends must be chosen from float, sign, sq2, sq4, sq8")
    if batch_size < 1 or bootstrap < 0:
        raise ValueError("batch_size must be positive and bootstrap nonnegative")
    rows = []
    for dim in dict.fromkeys(dims):
        q_prefix, c_prefix = q[:, :dim], c[:, :dim]
        for backend in dict.fromkeys(backends):
            if backend == "float":
                q_repr, c_repr = normalize(q_prefix), normalize(c_prefix)
            elif backend == "sign":
                # Zero maps to +1: exactly one bit per coordinate, no ternary zero.
                q_repr = np.where(q_prefix >= 0, 1.0, -1.0).astype(np.float32)
                c_repr = np.where(c_prefix >= 0, 1.0, -1.0).astype(np.float32)
            else:
                quantizer = CandidateScalarQuantizer.fit(c_prefix, int(backend[2:]))
                q_repr = normalize(quantizer.transform(q_prefix))
                c_repr = normalize(quantizer.transform(c_prefix))
            parts = {key: [] for key in ("R@1", "R@5", "R@10", "NDCG@10")}
            for start in range(0, len(q), batch_size):
                scores = q_repr[start:start + batch_size] @ c_repr.T
                # Preserve the original NumPy quicksort convention: sign hashes
                # frequently tie, so a different sorting algorithm changes metrics.
                ranked = np.argsort(-scores, axis=1, kind="quicksort")[:, :min(10, len(c))]
                metrics = per_query_metrics(ranked, gt[start:start + batch_size])
                for key, value in metrics.items():
                    parts[key].append(value)
            values = {key: np.concatenate(value) for key, value in parts.items()}
            row = {"dim": int(dim), "backend": backend,
                   "metrics": {key: float(value.mean()) for key, value in values.items()}}
            if bootstrap:
                row["bootstrap_ci95"] = _bootstrap(values, bootstrap, seed)
            rows.append(row)
    return {"schema_version": 1, "protocol": "candidate-only-affine-sq",
            "metric_units": "fraction", "queries": len(q), "candidates": len(c),
            "embedding_dim": q.shape[1], "tie_break": "numpy_quicksort",
            "bootstrap": {"draws": bootstrap, "seed": seed, "unit": "query"}, "results": rows}
