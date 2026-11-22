"""IQD definitions extracted from the research implementation.

Derived from ``scripts/finetune/validate_iqd.py`` and
``scripts/finetune/validate_iqd_gt.py`` of HiQ, under Apache-2.0.
The original candidate-side density/LID, fixed-grid quantization, rank ties,
and loss-gap definitions are retained. Ground truth is always explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.stats import spearmanr


PREDICTORS = (
    "quant_error", "density", "iqd_combined", "margin_float",
    "margin_erosion", "lid", "inv_margin_float", "focal_weight", "online_diff",
)
STRATIFIERS = (
    "iqd_combined", "margin_erosion", "focal_weight", "online_diff",
    "inv_margin_float", "lid",
)
QUANTIZATION = {
    "name": "iqd_fixed_symmetric_grid",
    "input_normalization": "L2 per vector after prefix truncation",
    "range": [-1.0, 1.0],
    "one_bit": "numpy.sign; exact zeros remain zero",
    "multi_bit": "2**bits uniformly spaced levels, numpy.round",
    "similarity": "cosine after L2-normalizing quantized vectors",
    "calibration": "none",
}


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2 or not all(value.shape):
        raise ValueError(f"{name} must be a non-empty 2D array")
    if value.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    with np.errstate(over="ignore", invalid="ignore"):
        value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite float32 values")
    if np.any(np.linalg.norm(value, axis=1) == 0):
        raise ValueError(f"{name} contains a zero-length embedding")
    return value


def _ids(value: np.ndarray | None, n: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    value = np.asarray(value)
    if value.shape != (n,) or value.dtype.kind not in "US":
        raise ValueError(f"{name} must be a length-{n} string array (not object dtype)")
    value = value.astype(str)
    if len(np.unique(value)) != n:
        raise ValueError(f"{name} must be unique")
    return value


@dataclass
class EmbeddingData:
    query_embeddings: np.ndarray
    candidate_embeddings: np.ndarray
    correct_indices: np.ndarray
    query_ids: np.ndarray | None = None
    candidate_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.query_embeddings = _matrix(self.query_embeddings, "query_embeddings")
        self.candidate_embeddings = _matrix(self.candidate_embeddings, "candidate_embeddings")
        n, d = self.query_embeddings.shape
        m, cd = self.candidate_embeddings.shape
        if d != cd:
            raise ValueError("Query and candidate embedding dimensions must match")
        if m < 2:
            raise ValueError("IQD requires at least two candidates")
        gold = np.asarray(self.correct_indices)
        if gold.shape != (n,) or gold.dtype.kind not in "iu":
            raise ValueError("correct_indices must be one integer candidate index per query")
        if np.any(gold < 0) or np.any(gold >= m):
            raise ValueError("Missing or out-of-range ground truth in correct_indices")
        self.correct_indices = gold.astype(np.int64)
        self.query_ids = _ids(self.query_ids, n, "query_ids")
        self.candidate_ids = _ids(self.candidate_ids, m, "candidate_ids")


def load_embeddings(path: str | Path) -> EmbeddingData:
    """Load an NPZ without executing pickle content or inferring diagonal GT."""
    required = ("query_embeddings", "candidate_embeddings", "correct_indices")
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in required if name not in archive]
        if missing:
            raise ValueError(f"Embedding NPZ is missing: {', '.join(missing)}")
        return EmbeddingData(
            **{name: archive[name] for name in required},
            query_ids=archive["query_ids"] if "query_ids" in archive else None,
            candidate_ids=archive["candidate_ids"] if "candidate_ids" in archive else None,
        )


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def uniform_quantize(x: np.ndarray, bits: int) -> np.ndarray:
    """IQD's fixed grid; it does not fit per-vector or per-channel extrema."""
    if isinstance(bits, bool) or not isinstance(bits, (int, np.integer)) or not 1 <= bits <= 16:
        raise ValueError("bits must be an integer in [1, 16]")
    x = l2_normalize(np.asarray(x, dtype=np.float32))
    if bits == 1:
        return np.sign(x)
    levels = (1 << int(bits)) - 1
    return np.round(np.clip((x + 1.0) / 2.0, 0.0, 1.0) * levels) / levels * 2.0 - 1.0


def _candidate_geometry(candidates: np.ndarray, k: int, block_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Density and clipped LID on the candidate pool, excluding self."""
    z = l2_normalize(candidates)
    n = len(z)
    density_k, lid_k = min(k, n - 1), min(20, n - 1)
    density = np.empty(n, dtype=np.float32)
    lid = np.empty(n, dtype=np.float32)
    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        sim = z[start:stop] @ z.T
        sim[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        nearest = np.partition(-sim, kth=density_k - 1, axis=1)[:, :density_k]
        density[start:stop] = -nearest.mean(axis=1)
        distances = np.sqrt(np.maximum(2.0 - 2.0 * sim, 0.0))
        nearest_d = np.sort(np.partition(distances, kth=lid_k - 1, axis=1)[:, :lid_k], axis=1)
        ratios = np.clip(nearest_d / np.clip(nearest_d[:, -1:], 1e-12, None), 1e-12, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            estimates = -lid_k / np.log(ratios).sum(axis=1)
        # Preserve the research estimator, including its [0, 1000] clipping.
        lid[start:stop] = np.clip(estimates, 0.0, 1000.0)
    return density, lid


def _directional_metrics(
    qf: np.ndarray, cf: np.ndarray, qq: np.ndarray, cq: np.ndarray,
    correct: np.ndarray, k: int, block_size: int,
) -> dict[str, np.ndarray]:
    keys = ("rank_float", "rank_quant", "hit_float", "hit_quant", "margin_float", "margin_quant", "focal_weight", "online_diff")
    result = {name: np.empty(len(qf), dtype=np.float32) for name in keys}
    for start in range(0, len(qf), block_size):
        stop = min(start + block_size, len(qf))
        gold = correct[start:stop]
        rows = np.arange(stop - start)
        sf, sq = qf[start:stop] @ cf.T, qq[start:stop] @ cq.T
        for suffix, sims in (("float", sf), ("quant", sq)):
            positive = sims[rows, gold]
            result[f"rank_{suffix}"][start:stop] = (sims > positive[:, None]).sum(axis=1)
            # Match the historical top-k membership rule, including its tie behavior.
            topk = np.argpartition(-sims, kth=min(k, sims.shape[1] - 1), axis=1)[:, :k]
            result[f"hit_{suffix}"][start:stop] = (topk == gold[:, None]).any(axis=1)
            negative = sims.copy()
            negative[rows, gold] = -np.inf
            result[f"margin_{suffix}"][start:stop] = positive - negative.max(axis=1)
        lf, lq = sf / 0.07, sq / 0.07
        lf -= lf.max(axis=1, keepdims=True)
        lq -= lq.max(axis=1, keepdims=True)
        logpf = lf[rows, gold] - np.log(np.exp(lf).sum(axis=1))
        logpq = lq[rows, gold] - np.log(np.exp(lq).sum(axis=1))
        result["online_diff"][start:stop] = logpf - logpq
        result["focal_weight"][start:stop] = (1.0 - np.exp(logpq)) ** 2.0
    result["rank_drop"] = result["rank_quant"] - result["rank_float"]
    result["recall_drop"] = result["hit_float"] * (1.0 - result["hit_quant"])
    return result


def correlation_summary(values: np.ndarray, target: np.ndarray, min_samples: int = 10) -> dict:
    """Spearman over finite pairs, with explicit undefined-correlation states."""
    values, target = np.asarray(values), np.asarray(target)
    valid = np.isfinite(values) & np.isfinite(target)
    x, y = values[valid], target[valid]
    result = {"n_valid": int(valid.sum()), "spearman_rho": None, "p_value": None}
    if len(x) < min_samples:
        result["status"] = "insufficient_samples"
    elif np.std(y) < 1e-8:
        result["status"] = "constant_rank_drop"
    elif np.ptp(x) == 0:
        result["status"] = "constant_predictor"
    else:
        rho, p = spearmanr(x, y)
        if np.isfinite(rho) and np.isfinite(p):
            result.update(spearman_rho=float(rho), p_value=float(p), status="ok")
        else:
            result["status"] = "undefined"
    finite_values = values[np.isfinite(values)]
    result["mean"] = float(finite_values.mean()) if len(finite_values) else None
    result["std"] = float(finite_values.std()) if len(finite_values) else None
    return result


def stratify(values: np.ndarray, samples: dict[str, np.ndarray]) -> dict:
    """Value-based thirds; ties need not produce equally sized groups."""
    valid = np.isfinite(values) & np.isfinite(samples["rank_drop"]) & np.isfinite(samples["recall_drop"])
    if not valid.any():
        return {"thresholds": None, "n_valid": 0, "groups": {}}
    lower, upper = np.quantile(values[valid], [1 / 3, 2 / 3])
    groups = {}
    for name, condition in (
        ("easy", values <= lower),
        ("medium", (values > lower) & (values <= upper)),
        ("hard", values > upper),
    ):
        mask = valid & condition
        count = int(mask.sum())
        groups[name] = {
            "n": count,
            "recall_drop_rate": float(samples["recall_drop"][mask].mean()) if count else None,
            "mean_rank_drop": float(samples["rank_drop"][mask].mean()) if count else None,
            "recall_float": float(samples["hit_float"][mask].mean()) if count else None,
            "recall_quant": float(samples["hit_quant"][mask].mean()) if count else None,
        }
    return {"thresholds": [float(lower), float(upper)], "n_valid": int(valid.sum()), "groups": groups}


def analyze_setting(
    data: EmbeddingData, dim: int, bits: int, *, k: int = 10,
    symmetric: bool = False, block_size: int = 512,
) -> tuple[dict, dict[str, np.ndarray]]:
    """Measure one explicit dimension/bit cell, aligned to the query order."""
    if isinstance(dim, bool) or not isinstance(dim, (int, np.integer)) or not 1 <= dim <= data.query_embeddings.shape[1]:
        raise ValueError("dim must be an integer within the embedding dimension")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    q, c = data.query_embeddings[:, :dim], data.candidate_embeddings[:, :dim]
    if np.any(np.linalg.norm(q, axis=1) == 0) or np.any(np.linalg.norm(c, axis=1) == 0):
        raise ValueError(f"A prefix embedding has zero norm at dim={dim}")
    if symmetric and (len(q) != len(c) or len(np.unique(data.correct_indices)) != len(c)):
        raise ValueError("Symmetric IQD requires a bijective one-to-one query/candidate pairing")
    qf, cf = l2_normalize(q), l2_normalize(c)
    qraw, craw = uniform_quantize(q, bits), uniform_quantize(c, bits)
    qq, cq = l2_normalize(qraw), l2_normalize(craw)
    samples = _directional_metrics(qf, cf, qq, cq, data.correct_indices, k, block_size)
    if symmetric:
        inverse = np.empty(len(c), dtype=np.int64)
        inverse[data.correct_indices] = np.arange(len(q))
        reverse = _directional_metrics(cf, qf, cq, qq, inverse, k, block_size)
        for name in samples:
            samples[name] = (samples[name] + reverse[name][data.correct_indices]) / 2.0
    density, lid = _candidate_geometry(c, k, block_size)
    samples["quant_error"] = np.linalg.norm(qf - qraw, axis=1)
    samples["density"] = density[data.correct_indices]
    samples["lid"] = lid[data.correct_indices]
    samples["iqd_combined"] = samples["quant_error"] * samples["density"]
    samples["margin_erosion"] = samples["margin_float"] - samples["margin_quant"]
    samples["inv_margin_float"] = 1.0 / np.clip(samples["margin_float"], 1e-6, None)
    measures = {name: correlation_summary(samples[name], samples["rank_drop"]) for name in PREDICTORS}
    valid_rhos = {name: entry["spearman_rho"] for name, entry in measures.items() if entry["spearman_rho"] is not None}
    best = max(valid_rhos, key=lambda name: abs(valid_rhos[name])) if valid_rhos else None
    summary = {
        "dim": int(dim), "bits": int(bits), "n_queries": len(q), "n_candidates": len(c),
        "k": k, "effective_k": min(k, len(c)), "symmetric": symmetric,
        "density_k": min(k, len(c) - 1), "lid_k": min(20, len(c) - 1),
        "mean_rank_drop": float(samples["rank_drop"].mean()),
        "recall_drop_rate": float(samples["recall_drop"].mean()),
        "measures": measures, "best_predictor": best,
        "best_abs_rho": abs(valid_rhos[best]) if best is not None else None,
        "stratified": {name: stratify(samples[name], samples) for name in STRATIFIERS},
    }
    samples["correct_indices"] = data.correct_indices.copy()
    if data.query_ids is not None:
        samples["query_ids"] = data.query_ids.copy()
    if data.candidate_ids is not None:
        samples["candidate_ids"] = data.candidate_ids.copy()
    return summary, samples


def _grid(values: Iterable[int], name: str, upper: int) -> list[int]:
    result = list(values)
    if not result or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or not 1 <= v <= upper for v in result):
        raise ValueError(f"{name} must be a non-empty list of integers in [1, {upper}]")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} must not contain duplicates")
    return [int(v) for v in result]


def analyze_sweep(
    data: EmbeddingData, dims: Iterable[int], bits: Iterable[int], *,
    k: int = 10, symmetric: bool = False, block_size: int = 512,
) -> tuple[dict, dict[str, dict[str, np.ndarray]]]:
    """Run only the requested cells; native dimension is never appended."""
    dims = _grid(dims, "dims", data.query_embeddings.shape[1])
    bits = _grid(bits, "bits", 16)
    cells, per_sample = [], {}
    for dim in dims:
        for bit in bits:
            cell, samples = analyze_setting(data, dim, bit, k=k, symmetric=symmetric, block_size=block_size)
            key = f"d{dim}_b{bit}"
            cell["samples_file"] = f"samples_{key}.npz"
            cells.append(cell)
            per_sample[key] = samples
    summary = {
        "schema_version": 1, "quantization": QUANTIZATION.copy(),
        "ground_truth": "explicit single-positive correct_indices",
        "rank_definition": "zero-based count of candidate scores strictly greater than the gold score",
        "recall_definition": "float top-k hit lost after quantization, not signed recall difference",
        "topk_ties": "numpy.argpartition membership; candidate order matters at tied boundary",
        "correlation_min_samples": 10, "contrastive_temperature": 0.07, "focal_gamma": 2.0,
        "numeric_dtype": "float32", "dims": dims, "bits": bits,
        "k": k, "symmetric": symmetric, "n_queries": len(data.query_embeddings),
        "n_candidates": len(data.candidate_embeddings), "native_dimension": data.query_embeddings.shape[1],
        "settings_tested": len(cells), "per_setting": cells,
    }
    return summary, per_sample
