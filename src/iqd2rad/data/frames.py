"""Deterministic frame sampling from locally acquired videos."""

import json
import shutil
import subprocess
from pathlib import Path


def extract_frames(video, output_dir, num_frames=64):
    """Sample uniformly spaced decoded frame indices, including both endpoints.

    Extraction uses a temporary sibling directory; a partial failure cannot leave
    a directory that looks complete. Existing output directories are rejected.
    """
    import tempfile

    video, output = Path(video), Path(output_dir)
    if not video.is_file():
        raise FileNotFoundError(video)
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    if output.exists():
        raise FileExistsError(f"Frame directory already exists: {output}")
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("Install ffmpeg and ffprobe to extract frames; pre-extracted frames also work")
    info = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                           "-show_entries", "stream=nb_read_frames", "-of", "json", str(video)],
                          capture_output=True, text=True, check=True)
    streams = json.loads(info.stdout).get("streams", [])
    if not streams:
        raise ValueError("Video has no decodable video stream")
    count = int(streams[0]["nb_read_frames"])
    if count < 1:
        raise ValueError("Video has no decodable frames")
    take = min(num_frames, count)
    indices = [int(index * (count - 1) / (take - 1)) for index in range(take)] \
        if take > 1 else [0]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".frames-", dir=output.parent) as temporary:
        staging = Path(temporary)
        selection = "+".join(f"eq(n\\,{index})" for index in indices)
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video),
                        "-map", "0:v:0", "-vf", f"select={selection}", "-vsync", "0",
                        "-q:v", "2", "-start_number", "0", str(staging / "frame_%06d.jpg")],
                       check=True, capture_output=True)
        decoded = sorted(staging.glob("frame_*.jpg"))
        if len(decoded) != take or any(path.stat().st_size == 0 for path in decoded):
            raise RuntimeError("ffmpeg did not return the requested decoded frame sample")
        staging.rename(output)
    return sorted(output.glob("frame_*.jpg"))
