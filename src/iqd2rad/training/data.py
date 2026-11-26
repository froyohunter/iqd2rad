"""JSONL image/caption or frame-sequence/caption training pairs."""

import json
from pathlib import Path

import numpy as np


class ManifestDataset:
    """A map-style dataset without requiring torch merely to inspect a manifest."""

    def __init__(self, manifest, data_root=".", modality=None, max_frames=64):
        self.manifest = Path(manifest).expanduser().resolve()
        self.data_root = Path(data_root).expanduser().resolve()
        self.max_frames = max_frames
        self.records = []
        with self.manifest.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("each record must be an object")
                    if not isinstance(record.get("id"), (str, int)) or not str(record["id"]):
                        raise ValueError("a nonempty id is required")
                    if not isinstance(record.get("text"), str) or not record["text"].strip():
                        raise ValueError("nonempty text is required")
                    has_image, has_frames = "image" in record, "frames" in record
                    if has_image == has_frames:
                        raise ValueError("provide exactly one of image or frames")
                    if has_image:
                        if not isinstance(record["image"], str) or not record["image"]:
                            raise ValueError("image must be a nonempty local path")
                        if modality == "video":
                            raise ValueError("video recipe requires frames")
                    else:
                        frames = record["frames"]
                        if not isinstance(frames, list) or not frames or not all(isinstance(x, str) and x for x in frames):
                            raise ValueError("frames must be a nonempty list of local paths")
                        if modality == "image":
                            raise ValueError("image recipe requires image")
                    self.records.append(record)
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"{self.manifest}:{line_number}: {exc}") from exc
        if not self.records:
            raise ValueError(f"No training pairs in {self.manifest}")

    def __len__(self):
        return len(self.records)

    def _path(self, value):
        if "://" in value:
            raise ValueError("Manifest media must use local paths, not remote URLs")
        path = Path(value).expanduser()
        path = path if path.is_absolute() else self.data_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing manifest media: {path}")
        return str(path)

    def __getitem__(self, index):
        source = self.records[index]
        result = {"id": str(source["id"]), "text": source["text"]}
        if "image" in source:
            result["image"] = self._path(source["image"])
        else:
            frames = source["frames"]
            if len(frames) > self.max_frames:
                positions = np.linspace(0, len(frames) - 1, self.max_frames, dtype=int)
                frames = [frames[index] for index in positions]
            result["video"] = [self._path(path) for path in frames]
        return result


def collate_pairs(records):
    """Leave variable-size images/videos for the shared Qwen processor."""
    return records


def model_inputs(records, visual_input_includes_text=True):
    visuals, texts = [], []
    for record in records:
        visual = {key: record[key] for key in ("image", "video") if key in record}
        if visual_input_includes_text:
            visual["text"] = record["text"]
        visuals.append(visual)
        texts.append({"text": record["text"]})
    return visuals, texts
