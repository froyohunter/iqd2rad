"""Convert documented local annotation formats into portable JSONL manifests."""

import csv
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path

from .manifests import (check_disjoint, load_ids, load_jsonl, relative_path,
                        validate_retrieval, write_jsonl)

DATASETS = ("msrvtt", "mscoco", "msvd", "didemo", "flickr30k", "activitynet", "visualnews")
VIDEO_DATASETS = {"msrvtt", "msvd", "didemo", "activitynet"}
QUERY_INSTRUCTIONS = {
    "msrvtt": "Find a video that contains the following visual content.",
    "msvd": "Find the video snippet that corresponds to the given summary.",
    "didemo": "Find a video that includes the following described scenes.",
    "activitynet": "Find a video that matches the given description.",
    "mscoco": "Retrieve the image that best matches the given text description.",
    "flickr30k": "Retrieve the image that best matches the given text description.",
    "visualnews": "Retrieve the image that best matches the given text description.",
}


def _read(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        return load_jsonl(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if path.suffix.lower() in {".txt", ".token"}:
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            image_caption, text = line.split("\t", 1)
            filename, caption_id = image_caption.rsplit("#", 1)
            rows.append({"id": Path(filename).stem, "filename": filename,
                         "caption": text, "caption_id": caption_id})
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("records", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError("Expected a list of annotation records (or records/data wrapper)")


def _text(value):
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("raw", "caption", "sentence", "text", "description"):
            if isinstance(value.get(key), str):
                return value[key].strip()
        if isinstance(value.get("tokens"), list):
            return " ".join(value["tokens"]).strip()
    raise ValueError("A caption must be a string or a sentence record with raw/text/caption/tokens")


def _caption_list(value):
    values = value if isinstance(value, list) else [value]
    result = []
    for index, item in enumerate(values):
        text = _text(item)
        if not text:
            raise ValueError("Empty caption in annotation file")
        source_id = item.get("sentid", item.get("id", index)) if isinstance(item, dict) else index
        result.append({"key": str(source_id), "text": text})
    if not result:
        raise ValueError("A media item has no captions")
    return result


def _first(row, names):
    for name in names:
        if row.get(name) is not None:
            return row[name]
    return None


def _image_records(payload):
    """COCO captions, Karpathy COCO/Flickr, Flickr token files, VisualNews JSON."""
    if isinstance(payload, dict) and isinstance(payload.get("images"), list):
        annotations = {}
        for annotation in payload.get("annotations", []):
            media_id = str(annotation["image_id"])
            annotations.setdefault(media_id, []).append(
                {"id": annotation["id"], "caption": annotation["caption"]})
        rows = payload["images"]
    else:
        annotations, rows = {}, _rows(payload)
    grouped = OrderedDict()
    for row in rows:
        filename = _first(row, ("file_name", "filename", "image_path", "image"))
        if not isinstance(filename, str):
            raise ValueError("Image annotation requires file_name, filename, or image_path")
        if row.get("filepath"):
            filename = str(Path(row["filepath"]) / filename)
        source_id = _first(row, ("cocoid", "id", "imgid"))
        media_id = str(source_id if source_id is not None else Path(filename).stem)
        value = _first(row, ("sentences", "captions", "alt_text", "caption"))
        if value is None:
            value = annotations.get(media_id)
        if value is None:
            raise ValueError(f"Image {media_id} has no associated captions")
        captions = _caption_list(value)
        if "caption_id" in row and len(captions) == 1:
            captions[0]["key"] = str(row["caption_id"])
        record = {"id": media_id, "image": relative_path(filename),
                  "captions": captions, "split": row.get("split")}
        if media_id in grouped:
            old = grouped[media_id]
            if old["image"] != record["image"] or old["split"] != record["split"]:
                raise ValueError(f"Conflicting metadata for image {media_id}")
            old["captions"].extend(captions)
        else:
            grouped[media_id] = record
    return list(grouped.values())


def _anymodal_flickr_records(payload):
    """Use the actual AnyModal caption field, not an assumed Karpathy caption."""
    rows = _rows(payload)
    converted = []
    for row in rows:
        if not isinstance(row.get("filename"), str) or "alt_text" not in row:
            raise ValueError("AnyModal Flickr30k metadata requires filename and alt_text for every image")
        converted.append({"filename": row["filename"], "alt_text": row["alt_text"],
                          **{key: row[key] for key in ("id", "split") if key in row}})
    return _image_records(converted)


def _video_records(payload, dataset):
    """MSR-VTT sentences, MSVD captions/CSV, DiDeMo descriptions, ActivityNet JSON."""
    videos = {}
    if isinstance(payload, dict) and isinstance(payload.get("sentences"), list):
        videos = {str(row["video_id"]): row for row in payload.get("videos", [])}
        rows = payload["sentences"]
    elif isinstance(payload, dict) and not any(k in payload for k in ("records", "data")):
        rows = []
        for media_id, value in payload.items():
            if isinstance(value, dict):
                rows.append({**value, "video_id": media_id})
            elif isinstance(value, list):
                rows.append({"video_id": media_id, "captions": value})
            else:
                raise ValueError("Video-ID mapping values must contain caption records")
    else:
        rows = _rows(payload)
    grouped = OrderedDict()
    for row in rows:
        # The official MSVD multilingual CSV includes non-English descriptions.
        language = row.get("Language", "English").strip().lower()
        if dataset == "msvd" and language not in ("english", "en"):
            continue
        video_name = _first(row, ("video", "video_path", "filename"))
        media_id = _first(row, ("video_id", "id", "VideoID"))
        if media_id is None and row.get("YouTubeID"):
            if "Start" not in row or "End" not in row:
                raise ValueError("MSVD YouTubeID CSV requires Start and End clip times")
            media_id = f"{row['YouTubeID']}_{row['Start']}_{row['End']}"
        if media_id is None and isinstance(video_name, str):
            media_id = Path(video_name).stem
        if media_id is None:
            raise ValueError("Video annotation has no video_id or video filename")
        media_id = str(media_id)
        value = _first(row, ("sentences", "captions", "caption", "sentence", "description", "Description"))
        if value is None:
            raise ValueError(f"Video {media_id} has no caption field")
        captions = _caption_list(value)
        if "sen_id" in row and len(captions) == 1:
            captions[0]["key"] = str(row["sen_id"])
        record = {"id": media_id, "captions": captions,
                  "split": row.get("split", videos.get(media_id, {}).get("split"))}
        if isinstance(video_name, str):
            record["video"] = relative_path(video_name)
        if row.get("frames") is not None:
            record["frames"] = row["frames"]
        if media_id in grouped:
            old = grouped[media_id]
            if old["split"] != record["split"]:
                raise ValueError(f"Conflicting split labels for video {media_id}")
            old["captions"].extend(captions)
        else:
            grouped[media_id] = record
    return list(grouped.values())


def _natural_key(path):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", str(path))]


def _media(record, root, frames_subdir, num_frames, media_map):
    media_id = record["id"]
    override = media_map.get(media_id, {})
    if isinstance(override, str):
        override = {"image": override} if "image" in record else {"frame_dir": override}
    if not isinstance(override, dict):
        raise ValueError("media_map values must be paths or objects with image/frames/frame_dir")
    if "image" in record:
        return {"id": media_id, "image": relative_path(override.get("image", record["image"]))}
    if "frames" in override or "frames" in record:
        frames = override.get("frames", record.get("frames"))
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"Video {media_id} needs a nonempty frames list")
        frames = [relative_path(value) for value in frames]
    else:
        frame_dir = override.get("frame_dir")
        if frame_dir is None:
            frame_dir = str(Path(frames_subdir) / media_id)
        folder = root / relative_path(frame_dir)
        frames = [str(path.relative_to(root)) for path in sorted(folder.glob("*"), key=_natural_key)
                  if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
        if not frames:
            raise FileNotFoundError(f"No frames for video {media_id}; prepare frames or supply media_map")
    if len(frames) > num_frames:
        positions = [int(index * (len(frames) - 1) / (num_frames - 1)) for index in range(num_frames)] \
            if num_frames > 1 else [len(frames) // 2]
        frames = [frames[index] for index in positions]
    return {"id": media_id, "frames": frames}


def _selected(records, split, include_restval, split_ids, exclude_ids):
    if split is not None:
        labels = {split, "restval"} if split == "train" and include_restval else {split}
        # A split-specific annotation file may have no split field. In that case
        # the user-supplied filename/split is recorded but no split is invented.
        tagged = [record for record in records if record.get("split") is not None]
        if tagged:
            if len(tagged) != len(records):
                raise ValueError("Only some records have split labels; supply a consistently split annotation file")
            records = [record for record in records if record["split"] in labels]
    indexed = {record["id"]: record for record in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate media identities")
    ids = load_ids(split_ids) if split_ids else list(indexed)
    if split_ids:
        missing = set(ids) - indexed.keys()
        if missing:
            raise ValueError(f"Split IDs absent from selected annotation split: {sorted(missing)[:10]}")
    if exclude_ids:
        check_disjoint({"selected": ids, "excluded": load_ids(exclude_ids)})
    # Split lists define membership, just as the original dataset filter did.
    # Explicit query_ids/candidate_ids, separately, can define archive ordering.
    allowed = set(ids)
    return [record for record in records if record["id"] in allowed]


def _restrict(records, ids_path, kind):
    if not ids_path:
        return records
    ids = load_ids(ids_path)
    indexed = {record["id"]: record for record in records}
    missing = set(ids) - indexed.keys()
    if missing:
        raise ValueError(f"Requested {kind} IDs absent from annotations: {sorted(missing)[:10]}")
    return [indexed[value] for value in ids]


def _mmeb(payload, dataset, root, split, split_ids, exclude_ids):
    rows = _rows(payload)
    if split is not None and any("split" in row for row in rows):
        rows = [row for row in rows if row.get("split") == split]
    allowed = set(load_ids(split_ids)) if split_ids else None
    candidates, queries = OrderedDict(), []
    seen_positive_ids = set()
    for index, row in enumerate(rows):
        paths = row.get("tgt_img_path")
        if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
            raise ValueError("MMEB rows require nonempty tgt_img_path arrays")
        paths = [relative_path(path) for path in paths]
        positive_id = paths[0]
        if allowed is not None and positive_id not in allowed:
            continue
        seen_positive_ids.add(positive_id)
        for candidate_index, path in enumerate(paths):
            candidate = {"id": path, "image": path}
            if path in candidates and candidates[path] != candidate:
                raise ValueError(f"Conflicting MMEB candidate records for {path}")
            candidates[path] = candidate
        instruction = row.get("qry_inst", QUERY_INSTRUCTIONS[dataset]).replace("<|image_1|>", "").strip()
        queries.append({"id": str(row.get("id", f"{dataset}:q:{index}")),
                        "text": _text(row["qry_text"]), "instruction": instruction,
                        "positive_ids": [positive_id]})
    if allowed is not None and allowed - seen_positive_ids:
        raise ValueError("Some requested split IDs have no positive MMEB query")
    if exclude_ids:
        check_disjoint({"gallery": list(candidates), "excluded": load_ids(exclude_ids)})
    return queries, list(candidates.values())


def prepare_dataset(dataset, annotations, raw_root, output_dir, *, split=None,
                    annotation_format="auto", mode="retrieval", caption_mode="auto",
                    split_ids=None, exclude_ids=None, query_ids=None, candidate_ids=None,
                    media_map=None, frames_subdir="frames", num_frames=64,
                    include_restval=False, query_instruction=None,
                    expected_queries=None, expected_candidates=None):
    """Convert annotations locally; no raw-data downloading, copying, or fallback filtering."""
    dataset = dataset.lower().replace("-", "")
    if dataset not in DATASETS:
        raise ValueError(f"Supported datasets: {', '.join(DATASETS)}")
    if mode not in {"retrieval", "train"} or caption_mode not in {"auto", "all", "first", "paragraph"}:
        raise ValueError("Invalid mode or caption_mode")
    if annotation_format not in {"auto", "native", "mmeb", "anymodal-flickr30k"}:
        raise ValueError("Unsupported annotation_format")
    if annotation_format == "anymodal-flickr30k" and dataset != "flickr30k":
        raise ValueError("anymodal-flickr30k format is specific to Flickr30k")
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    root, output = Path(raw_root).resolve(), Path(output_dir)
    if not root.is_dir():
        raise FileNotFoundError("raw_root must be an existing local directory")
    payload = _read(annotations)
    mapping = json.loads(Path(media_map).read_text()) if media_map else {}
    if not isinstance(mapping, dict):
        raise ValueError("media_map must be a JSON object keyed by media ID")
    list_rows = payload if isinstance(payload, list) else payload.get("records", payload.get("data", []))
    is_mmeb = annotation_format == "mmeb" or (annotation_format == "auto" and bool(list_rows)
                                             and "qry_text" in list_rows[0] and "tgt_img_path" in list_rows[0])
    if is_mmeb:
        if dataset not in {"mscoco", "flickr30k", "visualnews"} or mode != "retrieval":
            raise ValueError("MMEB annotations are supported for image retrieval evaluation only")
        queries, candidates = _mmeb(payload, dataset, root, split, split_ids, exclude_ids)
        used_caption_mode = "metadata-queries"
    else:
        if annotation_format == "anymodal-flickr30k":
            records = _anymodal_flickr_records(payload)
        else:
            records = _video_records(payload, dataset) if dataset in VIDEO_DATASETS else _image_records(payload)
        records = _selected(records, split, include_restval, split_ids, exclude_ids)
        if not records:
            raise ValueError("No records remain after split selection")
        used_caption_mode = caption_mode
        if used_caption_mode == "auto":
            used_caption_mode = "all" if mode == "train" or dataset not in VIDEO_DATASETS else \
                ("paragraph" if dataset in {"didemo", "activitynet"} else "first")
        candidates, queries = [], []
        for record in records:
            candidates.append(_media(record, root, frames_subdir, num_frames, mapping))
            captions = record["captions"]
            if used_caption_mode == "first":
                captions = captions[:1]
            elif used_caption_mode == "paragraph":
                captions = [{"key": "paragraph", "text": " ".join(cap["text"] for cap in captions)}]
            for index, caption in enumerate(captions):
                # Prefixing with the media ID makes native caption IDs globally
                # unique; enumeration also handles caption lists without IDs.
                queries.append({"id": f"{record['id']}:caption:{caption['key']}:{index}",
                                "text": caption["text"], "instruction": QUERY_INSTRUCTIONS[dataset],
                                "positive_ids": [record["id"]]})
    if query_instruction is not None:
        if not isinstance(query_instruction, str) or not query_instruction.strip():
            raise ValueError("query_instruction must be a nonempty string")
        for query in queries:
            query["instruction"] = query_instruction.strip()
    queries = _restrict(queries, query_ids, "query")
    candidates = _restrict(candidates, candidate_ids, "candidate")
    validate_retrieval(queries, candidates, root)
    for name, expected, actual in (("queries", expected_queries, len(queries)),
                                  ("candidates", expected_candidates, len(candidates))):
        if expected is not None:
            if not isinstance(expected, int) or isinstance(expected, bool) or expected < 1:
                raise ValueError(f"expected_{name} must be a positive integer")
            if actual != expected:
                raise ValueError(f"Expected {expected} {name}, got {actual}; verify annotation source and split")
    # Refuse to overwrite a prepared split: callers choose a new destination.
    filenames = ["train.jsonl", "metadata.json"] if mode == "train" else \
        ["queries.jsonl", "candidates.jsonl", "metadata.json"]
    if any((output / filename).exists() for filename in filenames):
        raise FileExistsError("Prepared output already exists; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    if mode == "train":
        candidate_map = {candidate["id"]: candidate for candidate in candidates}
        training = []
        for query in queries:
            candidate = candidate_map[query["positive_ids"][0]]
            training.append({"id": query["id"], "text": query["text"],
                             **{key: candidate[key] for key in ("image", "frames") if key in candidate}})
        write_jsonl(output / "train.jsonl", training)
    else:
        write_jsonl(output / "queries.jsonl", queries)
        write_jsonl(output / "candidates.jsonl", candidates)
    digest = hashlib.sha256(Path(annotations).read_bytes()).hexdigest()
    metadata = {"schema_version": 1, "dataset": dataset, "split": split, "mode": mode,
                "annotation_format": "mmeb" if is_mmeb else
                    ("anymodal-flickr30k" if annotation_format == "anymodal-flickr30k" else "native"),
                "annotation_sha256": digest,
                "queries": len(queries), "candidates": len(candidates),
                "caption_mode": used_caption_mode, "num_frames_max": num_frames,
                "include_restval": include_restval, "explicit_split_ids": bool(split_ids),
                "explicit_query_ids": bool(query_ids), "explicit_candidate_ids": bool(candidate_ids),
                "query_instruction_override": query_instruction,
                "expected_queries": expected_queries, "expected_candidates": expected_candidates,
                "paths_relative_to": "user-supplied raw_root"}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
