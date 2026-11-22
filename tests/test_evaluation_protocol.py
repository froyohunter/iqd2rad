import json

import numpy as np
import pytest

from iqd2rad.evaluation.cli import main
from iqd2rad.evaluation.core import CandidateScalarQuantizer, evaluate_embeddings, load_embeddings


def test_candidate_grid_is_fixed_when_query_distribution_changes():
    candidates = np.array([[0, 4, -3], [3, 4, 3]], dtype=np.float32)
    quantizer = CandidateScalarQuantizer.fit(candidates, 2)
    query = np.array([[1.4, -100, 1.0]], dtype=np.float32)
    np.testing.assert_allclose(quantizer.transform(query), [[1, 4, 1]])
    combined = np.vstack([query, [1e10, 1e10, -1e10]])
    np.testing.assert_array_equal(quantizer.transform(query), quantizer.transform(combined)[:1])
    np.testing.assert_array_equal(quantizer.minimum, [0, 4, -3])
    np.testing.assert_array_equal(quantizer.maximum, [3, 4, 3])
    np.testing.assert_array_equal(quantizer.transform(combined)[1], [3, 4, -3])


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_affine_reconstruction_matches_research_numeric_convention(bits):
    candidates = np.array([[0, 1, -1], [1e-13, 1, 2], [0.8e-13, 1, 0]], dtype=np.float32)
    queries = np.array([[-10, 0, 9], [1e-13, 2, -8]], dtype=np.float32)
    minimum, maximum = candidates.min(0), candidates.max(0)
    scale = (maximum - minimum) / ((1 << bits) - 1)
    scale = np.where(scale < 1e-12, 1.0, scale)
    clipped = np.clip(queries, minimum, maximum)
    codes = np.rint((clipped - minimum) / scale)
    codes = np.clip(codes, 0, (1 << bits) - 1).astype(np.int16)
    expected = codes.astype(np.float32) * scale + minimum
    np.testing.assert_array_equal(CandidateScalarQuantizer.fit(candidates, bits).transform(queries), expected)


def test_ground_truth_tracks_shuffled_gallery_including_small_pool():
    query = np.array([[1, 0], [0, 1]], dtype=np.float32)
    candidates = query[::-1].copy()
    result = evaluate_embeddings(query, candidates, np.array([1, 0]), backends=["float"])
    assert result["results"][0]["metrics"] == {"R@1": 1., "R@5": 1., "R@10": 1., "NDCG@10": 1.}
    wrong = evaluate_embeddings(query, candidates, np.array([0, 1]), backends=["float"])
    assert wrong["results"][0]["metrics"]["R@1"] == 0


def test_multiple_relevant_candidates_have_normalized_ndcg():
    query = np.array([[1, 0]], dtype=np.float32)
    candidates = np.array([[1, 0], [0.8, 0.2], [-1, 0]], dtype=np.float32)
    result = evaluate_embeddings(query, candidates, np.array([[0, 1, -1]]), backends=["float"])
    assert result["results"][0]["metrics"]["NDCG@10"] == 1


def test_sign_ties_preserve_research_quicksort_order():
    # More than 16 equal scores exercises quicksort's non-stable tie handling.
    query = np.ones((1, 2), dtype=np.float32)
    candidates = np.ones((32, 2), dtype=np.float32)
    gold = np.array([15])
    ranking = np.argsort(-np.ones((1, 32)), axis=1)[0]
    expected = float(15 in ranking[:10])
    result = evaluate_embeddings(query, candidates, gold, backends=["sign"])
    assert result["tie_break"] == "numpy_quicksort"
    assert result["results"][0]["metrics"]["R@10"] == expected


@pytest.mark.parametrize("gold", [np.array([-1]), np.array([2]), np.array([0.0]), np.array([[0, 0]])])
def test_missing_invalid_and_duplicate_ground_truth_fail(gold):
    with pytest.raises(ValueError):
        evaluate_embeddings(np.ones((1, 2)), np.ones((2, 2)), gold)


def test_missing_archive_ground_truth_is_never_inferred(tmp_path):
    path = tmp_path / "no_gt.npz"
    np.savez(path, query_embeddings=np.ones((1, 2)), candidate_embeddings=np.ones((1, 2)))
    with pytest.raises(ValueError, match="correct_indices"):
        load_embeddings(path)


def test_cli_recipe_and_bootstrap_are_json_serializable(tmp_path):
    archive = tmp_path / "embeddings.npz"
    output = tmp_path / "result.json"
    np.savez(archive, query_embeddings=np.eye(2), candidate_embeddings=np.eye(2)[::-1],
             correct_indices=np.array([1, 0]), query_ids=np.array(["q1", "q2"]),
             candidate_ids=np.array(["c2", "c1"]))
    config = tmp_path / "eval.json"
    config.write_text(json.dumps({"embeddings": str(archive), "output": str(output),
                                  "backends": ["float"], "dims": [2]}))
    assert main(["--config", str(config), "--bootstrap", "50"]) == 0
    result = json.loads(output.read_text())
    assert result["metric_units"] == "fraction"
    assert result["results"][0]["bootstrap_ci95"]["R@1"] == [1., 1.]


def test_batching_does_not_change_candidate_fit_or_ranking():
    rng = np.random.default_rng(3)
    query, candidates = rng.normal(size=(11, 8)), rng.normal(size=(17, 8))
    gold = np.arange(11)
    a = evaluate_embeddings(query, candidates, gold, dims=[4, 8], batch_size=1)
    b = evaluate_embeddings(query, candidates, gold, dims=[4, 8], batch_size=9)
    assert a == b
