"""Scientific invariants for the offline IQD protocol; no model or data download."""

import json
from pathlib import Path

import numpy as np
import pytest

from iqd2rad.analysis import EmbeddingData, analyze_setting, analyze_sweep, load_embeddings
from iqd2rad.analysis.cli import main
from iqd2rad.analysis.core import PREDICTORS, correlation_summary, stratify, uniform_quantize


def gallery(seed=11):
    rng = np.random.default_rng(seed)
    return EmbeddingData(
        rng.normal(size=(12, 8)), rng.normal(size=(18, 8)),
        np.array([7, 3, 15, 1, 7, 9, 0, 10, 12, 5, 17, 2]),
        np.array([f"q{i}" for i in range(12)]),
        np.array([f"c{i}" for i in range(18)]),
    )


def test_fixed_grid_is_not_data_calibrated_and_sign_keeps_zero():
    x = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(uniform_quantize(x, 1), [[1, 0, -1], [0, 1, 0]])
    np.testing.assert_allclose(uniform_quantize(x, 2), [[1, 1 / 3, -1], [1 / 3, 1, 1 / 3]], atol=1e-7)
    # A separate vector cannot change the fixed grid or the first vector's output.
    np.testing.assert_array_equal(uniform_quantize(x[:1], 2), uniform_quantize(x, 2)[:1])
    np.testing.assert_array_equal(uniform_quantize(x * 5, 4), uniform_quantize(x, 4))


def test_gallery_metrics_follow_gold_when_candidate_order_changes():
    data = gallery()
    order = np.random.default_rng(3).permutation(len(data.candidate_embeddings))
    inverse = np.argsort(order)
    moved = EmbeddingData(data.query_embeddings, data.candidate_embeddings[order],
                          inverse[data.correct_indices], data.query_ids, data.candidate_ids[order])
    _, original = analyze_setting(data, 8, 8, k=3)
    _, reordered = analyze_setting(moved, 8, 8, k=3)
    for key in (*PREDICTORS, "rank_drop", "recall_drop"):
        np.testing.assert_allclose(original[key], reordered[key], rtol=2e-5, atol=2e-6, err_msg=key)
    np.testing.assert_array_equal(data.candidate_ids[data.correct_indices], moved.candidate_ids[moved.correct_indices])


def test_candidate_geometry_is_gathered_at_each_querys_gold():
    candidates = np.array([[1, 0], [0.5, np.sqrt(3) / 2], [-1, 0], [0, -1]])
    data = EmbeddingData(np.array([[1, 0.2], [0.4, 1], [-0.8, 0.3]]), candidates, np.array([1, 2, 3]))
    _, samples = analyze_setting(data, 2, 8, k=1)
    np.testing.assert_allclose(samples["density"], [0.5, 0, 0], atol=1e-7)


def test_recall_drop_counts_losses_without_subtracting_improvements():
    data = EmbeddingData(
        np.array([[1, 0.1], [1, 0.1]]),
        np.array([[1, -0.01], [0.1, 1]]), np.array([0, 1]),
    )
    _, samples = analyze_setting(data, 2, 1, k=1)
    np.testing.assert_array_equal(samples["rank_float"], [0, 1])
    np.testing.assert_array_equal(samples["rank_quant"], [1, 0])
    np.testing.assert_array_equal(samples["rank_drop"], [1, -1])
    np.testing.assert_array_equal(samples["recall_drop"], [1, 0])


def test_symmetric_metrics_align_reverse_direction_using_inverse_gold():
    rng = np.random.default_rng(21)
    q, c = rng.normal(size=(12, 7)), rng.normal(size=(12, 7))
    gold = rng.permutation(12)
    data = EmbeddingData(q, c, gold)
    reverse_data = EmbeddingData(c, q, np.argsort(gold))
    _, forward = analyze_setting(data, 7, 2, k=2)
    _, reverse = analyze_setting(reverse_data, 7, 2, k=2)
    _, both = analyze_setting(data, 7, 2, k=2, symmetric=True)
    for key in ("rank_drop", "recall_drop", "online_diff", "focal_weight", "margin_float", "margin_erosion"):
        np.testing.assert_allclose(both[key], (forward[key] + reverse[key][gold]) / 2, atol=1e-6)
    # Query distortion and matched-candidate geometry remain one descriptor per pair.
    np.testing.assert_array_equal(both["quant_error"], forward["quant_error"])
    with pytest.raises(ValueError, match="bijective"):
        analyze_setting(gallery(), 8, 2, symmetric=True)


def test_spearman_filters_nonfinite_pairs_and_exposes_undefined_cases():
    x = np.r_[np.arange(12), np.nan, 99]
    y = np.r_[-np.arange(12), 2, np.inf]
    result = correlation_summary(x, y)
    assert result["status"] == "ok" and result["n_valid"] == 12
    assert result["spearman_rho"] == pytest.approx(-1)
    assert correlation_summary(np.ones(12), np.arange(12))["status"] == "constant_predictor"
    assert correlation_summary(np.arange(12), np.zeros(12))["status"] == "constant_rank_drop"
    assert correlation_summary(np.arange(9), np.arange(9))["status"] == "insufficient_samples"


def test_value_terciles_preserve_ties_and_report_empty_groups():
    values = np.r_[np.zeros(8), np.ones(4)]
    samples = {"rank_drop": np.arange(12), "recall_drop": np.zeros(12),
               "hit_float": np.ones(12), "hit_quant": np.ones(12)}
    result = stratify(values, samples)
    assert [result["groups"][key]["n"] for key in ("easy", "medium", "hard")] == [8, 0, 4]
    assert result["groups"]["medium"]["recall_drop_rate"] is None
    assert sum(group["n"] for group in result["groups"].values()) == result["n_valid"] == 12


@pytest.mark.parametrize("gold", [np.array([-1, 0]), np.array([0, 3]), np.array([0.0, 1.0]), np.array([[0], [1]])])
def test_invalid_or_missing_ground_truth_is_rejected(gold):
    with pytest.raises(ValueError, match="correct_indices|ground truth"):
        EmbeddingData(np.ones((2, 3)), np.ones((3, 3)), gold)


def test_npz_requires_explicit_gold_and_rejects_pickle_ids(tmp_path):
    path = tmp_path / "input.npz"
    np.savez(path, query_embeddings=np.ones((2, 3)), candidate_embeddings=np.ones((3, 3)))
    with pytest.raises(ValueError, match="correct_indices"):
        load_embeddings(path)
    np.savez(path, query_embeddings=np.ones((2, 3)), candidate_embeddings=np.ones((3, 3)),
             correct_indices=np.array([0, 1]), query_ids=np.array(["q0", "q1"], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        load_embeddings(path)


def test_sweep_has_exact_grid_and_all_nine_predictors():
    summary, samples = analyze_sweep(gallery(), [3, 5], [1, 4], k=3)
    assert summary["dims"] == [3, 5]
    assert summary["native_dimension"] == 8
    assert summary["settings_tested"] == len(samples) == 4
    assert all(set(cell["measures"]) == set(PREDICTORS) for cell in summary["per_setting"])
    json.dumps(summary, allow_nan=False)
    with pytest.raises(ValueError, match="dims"):
        analyze_sweep(gallery(), [3, 9], [2])


def test_cli_roundtrip_and_explicit_override(tmp_path):
    data = gallery()
    archive = tmp_path / "input.npz"
    np.savez(archive, **vars(data))
    config = tmp_path / "grid.json"
    config.write_text(json.dumps({"dims": [3, 5], "bits": [1, 4], "k": 3, "symmetric": False}))
    output = tmp_path / "results"
    args = ["--embeddings", str(archive), "--output-dir", str(output), "--config", str(config), "--dims", "5", "--bits", "4"]
    assert main(args) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["settings_tested"] == 1
    assert summary["dims"] == [5] and summary["bits"] == [4]
    assert len(summary["source"]["sha256"]) == 64
    with np.load(output / "samples_d5_b4.npz", allow_pickle=False) as result:
        np.testing.assert_array_equal(result["correct_indices"], data.correct_indices)
        np.testing.assert_array_equal(result["query_ids"], data.query_ids)
        assert result["rank_drop"].shape == (12,)
    with pytest.raises(SystemExit) as rejected:
        main(args)
    assert rejected.value.code == 2


def test_paper_manifests_define_exact_70_cell_grid():
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "iqd"
    grids = {path.stem: json.loads(path.read_text()) for path in config_dir.glob("*.json")}
    assert set(grids) == {"qwen_msrvtt", "qwen_mscoco", "clip_msrvtt", "clip_mscoco"}
    assert sum(len(grid["dims"]) * len(grid["bits"]) for grid in grids.values()) == 70
    assert grids["qwen_mscoco"]["dims"] == [128, 256, 512, 1024]
    assert grids["clip_msrvtt"]["encoding_provenance_status"] == "unverified"


def test_optional_plot_uses_measured_correlations(tmp_path):
    pytest.importorskip("matplotlib")
    from iqd2rad.analysis.plotting import plot_correlations
    summary, _ = analyze_sweep(gallery(), [4, 8], [1, 2], k=3)
    path = tmp_path / "correlations.png"
    plot_correlations(summary, path)
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
