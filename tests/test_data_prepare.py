import json
from pathlib import Path

import pytest

from iqd2rad.data.cli import main
from iqd2rad.data.manifests import check_disjoint, load_jsonl, validate_retrieval
from iqd2rad.data.prepare import prepare_dataset


def annotation(tmp_path, value, name="annotations.json"):
    path = tmp_path / name
    path.write_text(json.dumps(value))
    return path


def media(tmp_path, *names):
    root = tmp_path / "raw"
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def test_karpathy_first_caption_train_and_restval(tmp_path):
    root = media(tmp_path, "train/a.jpg", "val/b.jpg", "val/c.jpg")
    path = annotation(tmp_path, {"images": [
        {"cocoid": 0, "filename": "a.jpg", "filepath": "train", "split": "train",
         "sentences": [{"raw": "first"}, {"raw": "second"}]},
        {"cocoid": 2, "filename": "b.jpg", "filepath": "val", "split": "restval", "sentences": ["rest"]},
        {"cocoid": 3, "filename": "c.jpg", "filepath": "val", "split": "test", "sentences": ["test"]}]})
    output = tmp_path / "prepared"
    result = prepare_dataset("mscoco", path, root, output, split="train", include_restval=True,
                             mode="train", caption_mode="first")
    rows = load_jsonl(output / "train.jsonl")
    assert result["candidates"] == 2
    assert [row["text"] for row in rows] == ["first", "rest"]
    assert rows[0]["id"].startswith("0:")
    assert all(not Path(row["image"]).is_absolute() for row in rows)


def test_official_coco_annotation_join_handles_nondiagonal_ids(tmp_path):
    root = media(tmp_path, "a.jpg", "b.jpg")
    path = annotation(tmp_path, {"images": [{"id": 9, "file_name": "b.jpg"}, {"id": 0, "file_name": "a.jpg"}],
                               "annotations": [{"id": 1, "image_id": 0, "caption": "A"},
                                               {"id": 2, "image_id": 9, "caption": "B"}]})
    output = tmp_path / "prepared"
    prepare_dataset("mscoco", path, root, output, caption_mode="first")
    queries, candidates = load_jsonl(output / "queries.jsonl"), load_jsonl(output / "candidates.jsonl")
    assert [row["positive_ids"] for row in queries] == [["9"], ["0"]]
    assert validate_retrieval(queries[::-1], candidates, root) == [[1], [0]]


def test_mmeb_preserves_full_gallery_and_positive_identity_without_caption_leakage(tmp_path):
    root = media(tmp_path, "a.jpg", "b.jpg", "negative.jpg")
    path = annotation(tmp_path, [
        {"id": "q_b", "qry_text": "B", "qry_inst": "Find it <|image_1|>",
         "tgt_img_path": ["b.jpg", "negative.jpg"], "tgt_text": ["leaked B", ""]},
        {"id": "q_a", "qry_text": "A", "tgt_img_path": ["a.jpg", "b.jpg"]}])
    output = tmp_path / "prepared"
    prepare_dataset("visualnews", path, root, output)
    queries, candidates = load_jsonl(output / "queries.jsonl"), load_jsonl(output / "candidates.jsonl")
    assert [c["id"] for c in candidates] == ["b.jpg", "negative.jpg", "a.jpg"]
    assert validate_retrieval(queries, candidates, root) == [[0], [2]]
    assert queries[0]["instruction"] == "Find it"
    assert all("text" not in c for c in candidates)


def test_msrvtt_official_json_filters_ids_without_changing_annotation_order(tmp_path):
    root = media(tmp_path, "frames/video0/frame_0.jpg", "frames/video1/frame_0.jpg")
    path = annotation(tmp_path, {"videos": [{"video_id": "video0", "split": "train"},
                                           {"video_id": "video1", "split": "train"}],
                               "sentences": [{"video_id": "video0", "caption": "zero", "sen_id": 0},
                                             {"video_id": "video1", "caption": "one", "sen_id": 1}]})
    ids = annotation(tmp_path, ["video1", "video0"], "ids.json")
    output = tmp_path / "prepared"
    prepare_dataset("msrvtt", path, root, output, split="train", split_ids=ids)
    assert [row["text"] for row in load_jsonl(output / "queries.jsonl")] == ["zero", "one"]


def test_msrvtt_official_csv(tmp_path):
    root = media(tmp_path, "frames/video7/frame_0.jpg")
    path = tmp_path / "test.csv"
    path.write_text('video_id,sentence\nvideo7,"An example, with a comma"\n')
    output = tmp_path / "prepared"
    prepare_dataset("msrvtt", path, root, output)
    assert load_jsonl(output / "queries.jsonl")[0]["text"] == "An example, with a comma"


def test_msvd_official_csv_filters_languages(tmp_path):
    root = media(tmp_path, "frames/youtube_0_10/frame_0.jpg")
    path = tmp_path / "msvd.csv"
    path.write_text("YouTubeID,Start,End,Language,Description\nyoutube,0,10,English,A dog runs\nyoutube,0,10,French,Un chien court\n")
    output = tmp_path / "prepared"
    prepare_dataset("msvd", path, root, output)
    queries = load_jsonl(output / "queries.jsonl")
    assert len(queries) == 1 and queries[0]["text"] == "A dog runs"


def test_didemo_groups_descriptions_into_one_query(tmp_path):
    root = media(tmp_path, "frames/clip/frame_0.jpg")
    path = annotation(tmp_path, [{"video": "clip.mp4", "description": "A person enters.", "times": [[0, 1]]},
                                {"video": "clip.mp4", "description": "They sit down.", "times": [[1, 2]]}])
    output = tmp_path / "prepared"
    prepare_dataset("didemo", path, root, output)
    assert load_jsonl(output / "queries.jsonl")[0]["text"] == "A person enters. They sit down."


def test_activitynet_official_json_with_explicit_frame_mapping(tmp_path):
    root = media(tmp_path, "frames/example/frame_0.jpg")
    path = annotation(tmp_path, {"v_example": {"sentences": ["First scene.", "Next scene."],
                                             "timestamps": [[0, 1], [1, 2]], "duration": 2}})
    mapping = annotation(tmp_path, {"v_example": {"frame_dir": "frames/example"}}, "map.json")
    output = tmp_path / "prepared"
    prepare_dataset("activitynet", path, root, output, media_map=mapping)
    assert load_jsonl(output / "queries.jsonl")[0]["text"] == "First scene. Next scene."
    assert load_jsonl(output / "candidates.jsonl")[0]["id"] == "v_example"


def test_flickr_token_file_all_captions_and_split(tmp_path):
    root = media(tmp_path, "123.jpg", "456.jpg")
    path = tmp_path / "captions.token"
    path.write_text("123.jpg#0\tCaption zero\n123.jpg#1\tCaption one\n456.jpg#0\tOther split\n")
    ids = annotation(tmp_path, ["123"], "ids.json")
    output = tmp_path / "prepared"
    prepare_dataset("flickr30k", path, root, output, split_ids=ids)
    assert len(load_jsonl(output / "queries.jsonl")) == 2
    assert len(load_jsonl(output / "candidates.jsonl")) == 1


def test_flickr_paper_recipe_uses_anymodal_caption_source_and_expected_counts(tmp_path):
    recipe_path = Path(__file__).parents[1] / "configs" / "data" / "flickr30k_test.json"
    recipe = json.loads(recipe_path.read_text())
    assert recipe["annotation_format"] == "anymodal-flickr30k"
    assert recipe["caption_mode"] == "first"
    assert recipe["expected_queries"] == recipe["expected_candidates"] == 1000
    root = media(tmp_path, "123.jpg", "456.jpg")
    path = annotation(tmp_path, [
        {"filename": "123.jpg", "alt_text": ["Actual exported query"],
         "sentences": ["Different Karpathy caption"]},
        {"filename": "456.jpg", "alt_text": ["Second exported query"]}])
    output = tmp_path / "prepared"
    recipe.update(annotations=path, raw_root=root, output_dir=output,
                  expected_queries=2, expected_candidates=2)
    prepare_dataset(**recipe)
    queries = load_jsonl(output / "queries.jsonl")
    assert [row["text"] for row in queries] == ["Actual exported query", "Second exported query"]
    assert len(load_jsonl(output / "candidates.jsonl")) == 2
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["annotation_format"] == "anymodal-flickr30k"


def test_flickr_paper_format_rejects_karpathy_substitution_and_wrong_counts(tmp_path):
    root = media(tmp_path, "123.jpg")
    native = annotation(tmp_path, [{"filename": "123.jpg", "sentences": ["Karpathy caption"]}])
    with pytest.raises(ValueError, match="alt_text"):
        prepare_dataset("flickr30k", native, root, tmp_path / "native",
                        annotation_format="anymodal-flickr30k", caption_mode="first")
    actual = annotation(tmp_path, [{"filename": "123.jpg", "alt_text": ["Exported query"]}], "actual.json")
    with pytest.raises(ValueError, match="Expected 1000 queries"):
        prepare_dataset("flickr30k", actual, root, tmp_path / "incomplete",
                        annotation_format="anymodal-flickr30k", caption_mode="first",
                        expected_queries=1000, expected_candidates=1000)
    assert not (tmp_path / "incomplete").exists()


def test_visualnews_native_format(tmp_path):
    root = media(tmp_path, "images/1.jpg")
    path = annotation(tmp_path, {"data": [{"id": 1, "image_path": "./images/1.jpg", "caption": "A news image", "split": "test"}]})
    output = tmp_path / "prepared"
    prepare_dataset("visualnews", path, root, output, split="test", annotation_format="native")
    assert load_jsonl(output / "candidates.jsonl") == [{"id": "1", "image": "images/1.jpg"}]


def test_overlap_missing_media_and_missing_positive_fail(tmp_path):
    with pytest.raises(ValueError, match="overlap"):
        check_disjoint({"train": ["same"], "test": ["same"]})
    root = media(tmp_path, "present.jpg")
    path = annotation(tmp_path, [{"id": "x", "filename": "missing.jpg", "caption": "A"}])
    with pytest.raises(FileNotFoundError):
        prepare_dataset("flickr30k", path, root, tmp_path / "prepared")
    with pytest.raises(ValueError, match="missing candidate"):
        validate_retrieval([{"id": "q", "text": "A", "positive_ids": ["absent"]}],
                           [{"id": "present", "image": "present.jpg"}], root)


def test_media_cannot_escape_root(tmp_path):
    root = media(tmp_path, "present.jpg")
    path = annotation(tmp_path, [{"id": "x", "filename": "../outside.jpg", "caption": "A"}])
    with pytest.raises(ValueError, match="relative"):
        prepare_dataset("flickr30k", path, root, tmp_path / "prepared")


def test_data_cli_overrides_recipe_and_rejects_overwrite(tmp_path):
    root = media(tmp_path, "a.jpg")
    path = annotation(tmp_path, [{"id": 1, "filename": "a.jpg", "caption": "A"}])
    output = tmp_path / "prepared"
    config = annotation(tmp_path, {"dataset": "flickr30k", "annotations": str(path),
                                   "raw_root": str(root), "output_dir": str(output)}, "config.json")
    assert main(["prepare", "--config", str(config), "--mode", "train"]) == 0
    assert (output / "train.jsonl").is_file()
    with pytest.raises(FileExistsError):
        main(["prepare", "--config", str(config), "--mode", "train"])


def test_bundled_msrvtt_splits_are_complete_and_disjoint():
    split_dir = Path(__file__).parents[1] / "configs" / "data" / "splits"
    train = (split_dir / "msrvtt_train_ids.txt").read_text().splitlines()
    test = (split_dir / "msrvtt_test_ids.txt").read_text().splitlines()
    assert check_disjoint({"train": train, "test": test}) == {"train": 7010, "test": 1000}
