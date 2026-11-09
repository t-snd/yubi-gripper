import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from yubi_gripper import cli
from yubi_gripper import data as D
from yubi_gripper.mapping import GripperMapping

JOINT_STATE_DEF = """std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def _q(t):
    return 0.47 + 0.47 * np.sin(2 * np.pi * 0.5 * t)


@pytest.fixture
def bag(tmp_path):
    from mcap_ros2.writer import Writer

    path = tmp_path / "episode.mcap"
    with open(path, "wb") as fp:
        w = Writer(fp)
        schema = w.register_msgdef("sensor_msgs/msg/JointState", JOINT_STATE_DEF)
        t0 = 1_700_000_000.0
        for i in range(200):  # 100 Hz, 2 s
            t = i / 100.0
            for side, sign in (("left", 1.0), ("right", -1.0)):
                stamp = t0 + t
                msg = {
                    "header": {"stamp": {"sec": int(stamp), "nanosec": int((stamp % 1) * 1e9)},
                               "frame_id": ""},
                    "name": [D.JOINT_NAMES[side]],
                    "position": [float(_q(t)) if sign > 0 else float(0.94 - _q(t))],
                    "velocity": [], "effort": [],
                }
                w.write_message(D.SIDE_TOPICS[side], schema, msg,
                                log_time=int(stamp * 1e9), publish_time=int(stamp * 1e9))
        w.finish()
    return str(path)


@pytest.fixture
def lerobot(tmp_path):
    root = tmp_path / "lerobot_yubi"
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v2.1", "fps": 30, "total_episodes": 2, "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "features": {
            "observation.joint_states": {"dtype": "float32", "shape": [2],
                                         "names": ["left_right_finger_joint",
                                                   "right_right_finger_joint"]},
            "action.joint_states": {"dtype": "float32", "shape": [2],
                                    "names": ["left_right_finger_joint",
                                              "right_right_finger_joint"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    for ep in range(2):
        t = np.arange(60) / 30.0
        left = _q(t) * (ep + 1) / 2
        right = 0.94 - _q(t)
        table = pa.table({
            "observation.joint_states": [[float(a), float(b)] for a, b in zip(left, right)],
            "action.joint_states": [[float(a), float(b)] for a, b in zip(left, right)],
            "timestamp": t.astype(np.float32),
            "frame_index": np.arange(60),
            "episode_index": np.full(60, ep),
        })
        pq.write_table(table, root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
    return str(root)


def test_mcap_loader(bag):
    assert D.detect_format(bag) == "mcap"
    left = D.load_trajectory(bag, side="left")
    right = D.load_trajectory(bag, side="right")
    assert len(left.t) == 200
    assert left.t[0] == 0.0
    assert abs(left.rate_hz - 100.0) < 1.0
    np.testing.assert_allclose(left.q, _q(left.t), atol=1e-6)
    np.testing.assert_allclose(right.q, 0.94 - _q(right.t), atol=1e-6)
    assert left.meta["topic"] == "/yubi/gripper/left/joint_state"


def test_mcap_directory(bag):
    traj = D.load_trajectory(os.path.dirname(bag), side="left")
    assert len(traj.t) == 200


def test_lerobot_loader(lerobot):
    assert D.detect_format(lerobot) == "lerobot"
    ep1 = D.load_trajectory(lerobot, side="left", episode=1)
    assert len(ep1.t) == 60 and abs(ep1.rate_hz - 30.0) < 0.5
    np.testing.assert_allclose(ep1.q, _q(ep1.t), atol=1e-5)
    right = D.load_trajectory(lerobot, side="right", key="action.joint_states")
    np.testing.assert_allclose(right.q, 0.94 - _q(right.t), atol=1e-5)
    parquet = os.path.join(lerobot, "data", "chunk-000", "episode_000000.parquet")
    direct = D.load_trajectory(parquet, side="left")
    np.testing.assert_allclose(direct.q, _q(direct.t) / 2, atol=1e-5)


def test_mapping():
    m = GripperMapping(motor_closed_tick=2000, motor_open_tick=3000)
    assert m.to_tick(0.0) == 2000
    assert m.to_tick(0.94) == 3000
    assert m.to_tick(0.47) == 2500
    assert m.to_tick(-0.5) == 2000 and m.to_tick(5.0) == 3000  # clipped
    np.testing.assert_array_equal(m.to_tick(np.array([0.0, 0.94])), [2000, 3000])
    assert abs(m.tick_to_rad(2500) - 0.47) < 1e-9
    rev = GripperMapping(motor_closed_tick=3000, motor_open_tick=2000)
    assert rev.to_tick(0.94) == 2000


def test_resample(bag):
    traj = D.resample(D.load_trajectory(bag), 50.0)
    assert abs(traj.rate_hz - 50.0) < 0.5


class FakeMotor:
    """Simulated XM460 that follows the goal with a first-order lag."""

    def __init__(self):
        self.pos = 2048
        self.goal = 2048
        self.goals = []
        self.torque_on = False
        self.model_number = 1200

    def torque(self, on):
        self.torque_on = on

    def set_operating_mode(self, mode):
        assert not self.torque_on

    def current_limit(self):
        return 1000

    def position_limits(self):
        return 0, 4095

    def set_goal_current(self, v):
        self.goal_current = v

    def set_profile(self, v, a=0):
        pass

    def set_goal_position(self, tick):
        assert self.torque_on
        self.goal = tick
        self.goals.append(tick)

    def present_position(self):
        self.pos += int(round(0.7 * (self.goal - self.pos)))
        return self.pos

    def present_current(self):
        return 10

    def wait_until(self, tick, tol=20, timeout=5.0):
        self.pos = tick
        return tick

    def hardware_error(self):
        return 0

    def temperature(self):
        return 30

    def close(self):
        pass


def test_replay_with_fake_motor(bag, tmp_path, monkeypatch):
    motor = FakeMotor()
    monkeypatch.setattr(cli, "open_motor", lambda cfg: motor)
    log = tmp_path / "log.csv"
    rc = cli.main(["replay", bag, "--side", "left", "--yes", "--speed", "20",
                   "--motor-closed-tick", "2000", "--motor-open-tick", "3000",
                   "--log", str(log)])
    assert rc == 0
    assert len(motor.goals) == 1 + 200  # start position + all samples
    assert min(motor.goals) >= 2000 and max(motor.goals) <= 3000
    assert not motor.torque_on  # released at the end
    assert motor.goal_current == 300
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 201


def test_dry_run_lerobot(lerobot, tmp_path):
    log = tmp_path / "dry.csv"
    assert cli.main(["replay", lerobot, "--episode", "1", "--side", "right",
                     "--dry-run", "--log", str(log)]) == 0
    assert len(log.read_text().strip().splitlines()) == 61


def test_info(bag, lerobot, capsys):
    assert cli.main(["info", bag]) == 0
    assert cli.main(["info", lerobot]) == 0
    out = capsys.readouterr().out
    assert "/yubi/gripper/left/joint_state" in out
    assert "observation.joint_states" in out
