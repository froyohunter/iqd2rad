"""Validation shared by local dataset conversion and embedding export."""

import json
from pathlib import Path, PurePosixPath


def load_jsonl(path):
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(record)
    return records


def relative_path(value):
    value = str(value).replace("\\", "/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or ":" in value:
        raise ValueError("Media paths must be relative, without parent traversal or URI schemes")
    if str(path) == ".":
        raise ValueError("Media path must name a file")
    return str(path)


def validate_media(record, raw_root=None):
    if ("image" in record) == ("frames" in record):
        raise ValueError("A media record must contain exactly one of image or frames")
    paths = [record["image"]] if "image" in record else record["frames"]
    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) for p in paths):
        raise ValueError("frames must be a nonempty list of relative paths")
    root = Path(raw_root).resolve() if raw_root is not None else None
    for name in paths:
        rel = relative_path(name)
        if root is not None:
            resolved = (root / rel).resolve()
            if not resolved.is_relative_to(root):
                raise ValueError("A media symlink escapes raw_root")
            if not resolved.is_file():
                raise FileNotFoundError(f"Missing media relative to raw_root: {rel}")


def _id_map(records, kind):
    ids = {}
    for index, row in enumerate(records):
        value = row.get("id")
        if not isinstance(value, str) or not value:
            raise ValueError(f"Each {kind} id must be a nonempty string")
        if value in ids:
            raise ValueError(f"Duplicate {kind} id: {value}")
        ids[value] = index
    if not ids:
        raise ValueError(f"No {kind} records")
    return ids


def validate_retrieval(queries, candidates, raw_root=None):
    """Validate identity-based ground truth; return candidate indices per query."""
    _id_map(queries, "query")
    candidate_map = _id_map(candidates, "candidate")
    for candidate in candidates:
        validate_media(candidate, raw_root)
    indices = []
    for query in queries:
        if not isinstance(query.get("text"), str) or not query["text"].strip():
            raise ValueError(f"Empty query text: {query['id']}")
        positives = query.get("positive_ids")
        if not isinstance(positives, list) or not positives or any(not isinstance(v, str) for v in positives):
            raise ValueError(f"Query {query['id']} needs nonempty string positive_ids")
        if len(set(positives)) != len(positives):
            raise ValueError(f"Duplicate positive_ids for query {query['id']}")
        missing = set(positives) - candidate_map.keys()
        if missing:
            raise ValueError(f"Query {query['id']} references missing candidate IDs: {sorted(missing)}")
        indices.append([candidate_map[value] for value in positives])
    return indices


def load_ids(path):
    """Read explicit ordered IDs from a JSON list or newline-delimited text."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        values = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(values, list):
            raise ValueError("Split IDs must be a JSON list or a text file")
    else:
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(value) for value in values]
    if not ids or len(set(ids)) != len(ids) or any(not value for value in ids):
        raise ValueError("Split IDs must be nonempty and unique")
    return ids


def check_disjoint(named_splits):
    """Reject media identity overlap across named train/validation/test lists."""
    seen = {}
    for name, ids in named_splits.items():
        values = list(ids)
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate IDs in split {name}")
        for value in values:
            if value in seen:
                raise ValueError(f"Split overlap: {value} occurs in {seen[value]} and {name}")
            seen[value] = name
    return {name: len(ids) for name, ids in named_splits.items()}


def write_jsonl(path, records):
    path = Path(path)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in records),
                    encoding="utf-8")
