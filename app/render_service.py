from __future__ import annotations

import math
import os
import shutil
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import lameenc
import numpy as np

from .schemas import Arrangement, RenderResponse


SAMPLE_RATE = 22_050
TTL_HOURS = 24
RUNTIME_DIR = Path(os.getenv("SHANYIN_RUNTIME_DIR", "/app/runtime"))
EXPORT_DIR = RUNTIME_DIR / "exports"


def render_arrangement(arrangement: Arrangement, accompaniment_gain: float, public_base_url: str) -> RenderResponse:
    cleanup_expired()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    audio = np.zeros(int((arrangement.duration + 1.5) * SAMPLE_RATE), dtype=np.float32)
    for event in arrangement.accompaniment:
        gain = event.velocity * (0.46 if event.role == "melody" else 0.24 * accompaniment_gain)
        _mix_piano_note(audio, event.start, event.duration, event.midi, gain)
    peak = float(np.max(np.abs(audio))) if audio.size else 0
    if peak > 0.94:
        audio *= 0.94 / peak
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(3)
    mp3 = encoder.encode(pcm) + encoder.flush()
    object_name = f"shanyin-exports/{datetime.now(UTC):%Y/%m/%d}/{uuid.uuid4().hex}.mp3"
    path = EXPORT_DIR / (Path(object_name).name)
    path.write_bytes(mp3)
    expires = datetime.now(UTC) + timedelta(hours=TTL_HOURS)
    cloud = _upload_cos(path, object_name)
    if cloud:
        path.unlink(missing_ok=True)
        return RenderResponse(fileId=cloud, expiresAt=expires.isoformat())
    base = public_base_url.rstrip("/")
    return RenderResponse(downloadUrl=f"{base}/api/v1/render/files/{path.stem}", expiresAt=expires.isoformat())


def resolve_local_file(file_id: str) -> Path | None:
    if not file_id.isalnum() or len(file_id) != 32:
        return None
    path = EXPORT_DIR / f"{file_id}.mp3"
    return path if path.exists() and time.time() - path.stat().st_mtime < TTL_HOURS * 3600 else None


def cleanup_expired() -> int:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    removed = 0
    cutoff = time.time() - TTL_HOURS * 3600
    for path in EXPORT_DIR.glob("*.mp3"):
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def _mix_piano_note(target: np.ndarray, start: float, duration: float, midi: int, gain: float) -> None:
    note_seconds = min(8.0, max(0.08, duration + 0.65))
    count = int(note_seconds * SAMPLE_RATE)
    offset = int(start * SAMPLE_RATE)
    if offset >= target.size:
        return
    count = min(count, target.size - offset)
    t = np.arange(count, dtype=np.float32) / SAMPLE_RATE
    frequency = 440.0 * (2.0 ** ((midi - 69) / 12.0))
    envelope = np.minimum(1.0, t / 0.012) * np.exp(-t * (2.2 / max(0.25, duration)))
    tone = (
        np.sin(2 * math.pi * frequency * t)
        + 0.34 * np.sin(2 * math.pi * frequency * 2 * t)
        + 0.13 * np.sin(2 * math.pi * frequency * 3 * t)
    ) / 1.47
    target[offset : offset + count] += tone * envelope * gain


def _upload_cos(path: Path, object_name: str) -> str | None:
    required = ["TENCENTCLOUD_SECRET_ID", "TENCENTCLOUD_SECRET_KEY", "COS_REGION", "COS_BUCKET", "CLOUDBASE_ENV_ID"]
    if not all(os.getenv(name, "").strip() for name in required):
        return None
    from qcloud_cos import CosConfig, CosS3Client

    token = os.getenv("TENCENTCLOUD_SESSION_TOKEN", "") or None
    config = CosConfig(
        Region=os.environ["COS_REGION"],
        SecretId=os.environ["TENCENTCLOUD_SECRET_ID"],
        SecretKey=os.environ["TENCENTCLOUD_SECRET_KEY"],
        Token=token,
        Scheme="https",
    )
    client = CosS3Client(config)
    with path.open("rb") as handle:
        client.put_object(Bucket=os.environ["COS_BUCKET"], Body=handle, Key=object_name, ContentType="audio/mpeg")
    file_id_bucket = os.getenv("CLOUDBASE_FILEID_BUCKET", os.environ["COS_BUCKET"])
    return f"cloud://{os.environ['CLOUDBASE_ENV_ID']}.{file_id_bucket}/{object_name}"


def purge_runtime() -> None:
    """仅供测试清理明确的运行目录。"""
    if EXPORT_DIR.exists():
        shutil.rmtree(EXPORT_DIR)
