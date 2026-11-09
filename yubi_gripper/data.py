"""Load YUBI gripper encoder trajectories from mcap bags or LeRobot datasets.

Data conventions (from airoa-org/yubi-sw and airoa-org/rebake):

* ``encoder_node`` publishes ``sensor_msgs/JointState`` at ~100 Hz on
  ``/yubi/gripper/{left,right}/joint_state`` (one joint each) and on
  ``/joint_states`` (both joints, only when both sides are alive).
  ``position`` is in radians, 0 = fully closed (``min_raw`` offset captured
  in the device-setup GUI), increasing as the gripper opens. The URDF limit
  of the finger joint is 0.00 .. 0.94 rad.
* Joint names: ``left_right_finger_joint`` / ``right_right_finger_joint``.
* rebake (``config/robot_model/yubi.yaml``) writes these as
  ``observation.joint_states`` and ``action.joint_states`` (= state shifted
  by one step), shape ``[2]`` ordered ``[left, right]``, at 30 fps.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

SIDES = ("left", "right")
JOINT_NAMES = {
    "left": "left_right_finger_joint",
    "right": "right_right_finger_joint",
}
SIDE_TOPICS = {
    "left": "/yubi/gripper/left/joint_state",
    "right": "/yubi/gripper/right/joint_state",
}
COMBINED_TOPIC = "/joint_states"
DEFAULT_LEROBOT_KEY = "observation.joint_states"


@dataclass
class Trajectory:
    """Gripper angle trajectory: ``t`` [s, starts at 0] and ``q`` [rad]."""

    t: np.ndarray
    q: np.ndarray
    source: str
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.t = np.asarray(self.t, dtype=float)
        self.q = np.asarray(self.q, dtype=float)
        if self.t.shape != self.q.shape or self.t.ndim != 1:
            raise ValueError("t and q must be 1-D arrays of equal length")
        if len(self.t) == 0:
            raise ValueError(f"no gripper samples found in {self.source}")
        order = np.argsort(self.t, kind="stable")
        self.t = self.t[order] - self.t[order][0]
        self.q = self.q[order]

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    @property
    def rate_hz(self) -> float:
        if len(self.t) < 2 or self.duration <= 0:
            return float("nan")
        return (len(self.t) - 1) / self.duration

    def trimmed(self, start: float = 0.0, end: Optional[float] = None) -> "Trajectory":
        end = self.duration if end is None else end
        mask = (self.t >= start) & (self.t <= end)
        return Trajectory(self.t[mask], self.q[mask], self.source, dict(self.meta))

    def summary(self) -> str:
        return (
            f"{self.source}: {len(self.t)} samples, {self.duration:.2f} s, "
            f"~{self.rate_hz:.1f} Hz, q=[{self.q.min():.3f}, {self.q.max():.3f}] rad"
        )


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(path: str) -> str:
    """Return ``"mcap"`` or ``"lerobot"`` for ``path``."""
    if os.path.isfile(path):
        if path.endswith(".mcap"):
            return "mcap"
        if path.endswith(".parquet"):
            return "lerobot"
        raise ValueError(f"cannot infer format of file {path!r}")
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, "meta", "info.json")):
            return "lerobot"
        if _mcap_files(path):
            return "mcap"
    raise ValueError(f"{path!r} is neither an mcap bag nor a LeRobot dataset")


def load_trajectory(
    path: str,
    side: str = "left",
    fmt: str = "auto",
    topic: Optional[str] = None,
    key: str = DEFAULT_LEROBOT_KEY,
    episode: int = 0,
) -> Trajectory:
    """Load one side's gripper trajectory from ``path``."""
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    if fmt == "auto":
        fmt = detect_format(path)
    if fmt == "mcap":
        return load_mcap(path, side=side, topic=topic)
    if fmt == "lerobot":
        return load_lerobot(path, side=side, key=key, episode=episode)
    raise ValueError(f"unknown format {fmt!r}")


# ---------------------------------------------------------------------------
# mcap (rosbag2 / yubi-core recordings)
# ---------------------------------------------------------------------------

def _mcap_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(os.path.join(path, "*.mcap")))


def list_mcap_topics(path: str) -> Dict[str, Dict[str, object]]:
    """Return ``{topic: {"schema": ..., "count": ...}}`` for an mcap bag."""
    from mcap.reader import make_reader

    topics: Dict[str, Dict[str, object]] = {}
    for f in _mcap_files(path):
        with open(f, "rb") as fp:
            reader = make_reader(fp)
            summary = reader.get_summary()
            if summary is None:
                continue
            counts = summary.statistics.channel_message_counts if summary.statistics else {}
            for cid, ch in summary.channels.items():
                schema = summary.schemas.get(ch.schema_id)
                entry = topics.setdefault(
                    ch.topic, {"schema": schema.name if schema else "?", "count": 0}
                )
                entry["count"] = int(entry["count"]) + int(counts.get(cid, 0))
    return topics


def _stamp_seconds(msg, log_time_ns: int) -> float:
    header = getattr(msg, "header", None)
    if header is not None:
        st = header.stamp
        sec = getattr(st, "sec", 0) + getattr(st, "nanosec", 0) * 1e-9
        if sec > 0:
            return sec
    return log_time_ns * 1e-9


def load_mcap(path: str, side: str = "left", topic: Optional[str] = None) -> Trajectory:
    """Read ``sensor_msgs/JointState`` gripper angles from an mcap bag.

    ``path`` may be a single ``.mcap`` file or a rosbag2 directory.
    By default the per-side topic is used; if it is absent, ``/joint_states``
    is used and the joint is picked by name.
    """
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    files = _mcap_files(path)
    if not files:
        raise FileNotFoundError(f"no .mcap files in {path!r}")

    available = list_mcap_topics(path)
    if topic is None:
        candidates = [SIDE_TOPICS[side], COMBINED_TOPIC]
        topic = next((c for c in candidates if c in available), None)
        if topic is None:
            raise ValueError(
                f"none of {candidates} found in {path!r}. Available topics: "
                f"{sorted(available)}. Use --topic to choose one."
            )
    elif topic not in available:
        raise ValueError(f"topic {topic!r} not in {path!r}: {sorted(available)}")

    joint = JOINT_NAMES[side]
    ts: List[float] = []
    qs: List[float] = []
    for f in files:
        with open(f, "rb") as fp:
            reader = make_reader(fp, decoder_factories=[DecoderFactory()])
            for _schema, _channel, message, msg in reader.iter_decoded_messages(topics=[topic]):
                q = _extract_position(msg, joint)
                if q is None:
                    continue
                ts.append(_stamp_seconds(msg, message.log_time))
                qs.append(q)

    return Trajectory(
        np.array(ts), np.array(qs), source=f"mcap:{topic}",
        meta={"path": path, "topic": topic, "joint": joint},
    )


def _extract_position(msg, joint: str) -> Optional[float]:
    if hasattr(msg, "position"):
        positions = list(msg.position)
        names = list(getattr(msg, "name", []) or [])
        if not positions:
            return None
        if names and joint in names:
            return float(positions[names.index(joint)])
        if len(positions) == 1:
            return float(positions[0])
        return None
    if hasattr(msg, "data"):  # std_msgs/Float32/Float64
        return float(msg.data)
    return None


# ---------------------------------------------------------------------------
# LeRobot (rebake output, v2.1; v3-style chunked files also accepted)
# ---------------------------------------------------------------------------

def _read_info(root: str) -> Dict[str, object]:
    info_path = os.path.join(root, "meta", "info.json")
    if os.path.isfile(info_path):
        with open(info_path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return {}


def _find_dataset_root(parquet_path: str) -> Optional[str]:
    d = os.path.dirname(os.path.abspath(parquet_path))
    for _ in range(4):
        if os.path.isfile(os.path.join(d, "meta", "info.json")):
            return d
        d = os.path.dirname(d)
    return None


def _episode_parquet(root: str, info: Dict[str, object], episode: int) -> str:
    template = info.get("data_path")
    chunks_size = int(info.get("chunks_size", 1000) or 1000)
    if isinstance(template, str) and "episode_index" in template:
        p = os.path.join(
            root,
            template.format(episode_chunk=episode // chunks_size, episode_index=episode),
        )
        if os.path.isfile(p):
            return p
    hits = sorted(glob.glob(os.path.join(root, "data", "**", f"episode_{episode:06d}.parquet"),
                            recursive=True))
    if hits:
        return hits[0]
    files = sorted(glob.glob(os.path.join(root, "data", "**", "*.parquet"), recursive=True))
    if files:  # v3 style: multiple episodes per file, filtered by episode_index
        return files[0] if len(files) == 1 else ",".join(files)
    raise FileNotFoundError(f"episode {episode} not found under {root}/data")


def feature_names(info: Dict[str, object], key: str) -> Optional[List[str]]:
    feat = (info.get("features") or {}).get(key) if info else None
    if not feat:
        return None
    names = feat.get("names")
    if isinstance(names, dict):  # e.g. {"motors": [...]}
        names = next(iter(names.values()), None)
    return list(names) if isinstance(names, (list, tuple)) else None


def list_lerobot_keys(path: str) -> Dict[str, object]:
    root = path if os.path.isdir(path) else _find_dataset_root(path)
    info = _read_info(root) if root else {}
    return {
        "fps": info.get("fps"),
        "total_episodes": info.get("total_episodes"),
        "features": {
            k: {"shape": v.get("shape"), "names": feature_names(info, k)}
            for k, v in (info.get("features") or {}).items()
            if v.get("dtype") not in ("video", "image")
        },
    }


def load_lerobot(
    path: str, side: str = "left", key: str = DEFAULT_LEROBOT_KEY, episode: int = 0
) -> Trajectory:
    """Read the gripper column of one episode of a LeRobot dataset."""
    import pyarrow.parquet as pq
    import pyarrow as pa

    if os.path.isdir(path):
        root = path
        info = _read_info(root)
        parquet = _episode_parquet(root, info, episode)
    else:
        parquet = path
        root = _find_dataset_root(path)
        info = _read_info(root) if root else {}

    tables = [pq.read_table(p) for p in parquet.split(",")]
    table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
    cols = table.column_names
    if "episode_index" in cols:
        ep = np.asarray(table.column("episode_index").to_pylist())
        if np.unique(ep).size > 1 or os.path.isdir(path):
            mask = ep == episode
            if not mask.any():
                raise ValueError(f"episode_index {episode} not present in {parquet}")
            table = table.filter(pa.array(mask))

    if key not in cols:
        cand = [c for c in cols if "joint" in c or "gripper" in c]
        raise ValueError(f"column {key!r} not found. Candidates: {cand or cols}")

    values = table.column(key).to_pylist()
    names = feature_names(info, key)
    joint = JOINT_NAMES[side]
    if names and joint in names:
        idx = names.index(joint)
    else:
        idx = SIDES.index(side)
    q = np.array([v[idx] if isinstance(v, (list, tuple)) else v for v in values], dtype=float)

    fps = float(info.get("fps", 30) or 30)
    if "timestamp" in cols:
        t = np.asarray(table.column("timestamp").to_pylist(), dtype=float)
    elif "frame_index" in cols:
        t = np.asarray(table.column("frame_index").to_pylist(), dtype=float) / fps
    else:
        t = np.arange(len(q)) / fps

    return Trajectory(
        t, q, source=f"lerobot:{key}[{idx}]",
        meta={"path": parquet, "key": key, "index": idx, "fps": fps, "episode": episode},
    )


def resample(traj: Trajectory, rate_hz: float) -> Trajectory:
    """Zero-order-hold resample (useful to send commands at a fixed rate)."""
    t_new = np.arange(0.0, traj.duration + 1e-9, 1.0 / rate_hz)
    idx = np.clip(np.searchsorted(traj.t, t_new, side="right") - 1, 0, len(traj.t) - 1)
    return Trajectory(t_new, traj.q[idx], traj.source, dict(traj.meta))


def available_sources(path: str) -> Sequence[str]:
    fmt = detect_format(path)
    if fmt == "mcap":
        return sorted(list_mcap_topics(path))
    return sorted(list_lerobot_keys(path)["features"])
