"""Command line interface: ``yubi-gripper {info,calibrate,goto,replay}``.

Typical flow::

    yubi-gripper info  episode.mcap                 # what is in the data?
    yubi-gripper calibrate                          # measure closed/open ticks
    yubi-gripper replay episode.mcap --dry-run      # check the command, no motor
    yubi-gripper replay episode.mcap --side left    # move the gripper
    yubi-gripper replay lerobot_yubi/ --episode 3 --side right
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import signal
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from . import data as D
from .mapping import GripperMapping, TICKS_PER_REV

DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = {
    "dynamixel": {
        "port": "/dev/ttyUSB0",
        "baudrate": 57600,
        "id": 1,
        "operating_mode": 5,
        "goal_current": 300,
        "approach_profile_velocity": 30,
        "replay_profile_velocity": 0,
        "profile_acceleration": 0,
    },
    "mapping": {
        "encoder_closed_rad": 0.0,
        "encoder_open_rad": 0.94,
        "motor_closed_tick": 2048,
        "motor_open_tick": 3072,
        "clip": True,
    },
}
DEFAULT_CONFIG_PATH = os.path.join("config", "gripper.yaml")


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def load_config(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is None and os.path.isfile(DEFAULT_CONFIG_PATH):
        path = DEFAULT_CONFIG_PATH
    if path:
        import yaml

        with open(path, "r", encoding="utf-8") as fp:
            user = yaml.safe_load(fp) or {}
        for section, values in user.items():
            cfg.setdefault(section, {}).update(values or {})
        cfg["_path"] = path  # type: ignore[assignment]
    return cfg


def apply_overrides(cfg: Dict[str, Dict[str, Any]], args: argparse.Namespace) -> None:
    dxl = cfg["dynamixel"]
    for name in ("port", "baudrate", "id", "operating_mode", "goal_current"):
        v = getattr(args, name if name != "id" else "dxl_id", None)
        if v is not None:
            dxl[name] = v
    if getattr(args, "profile_velocity", None) is not None:
        dxl["replay_profile_velocity"] = args.profile_velocity
    m = cfg["mapping"]
    for name in ("encoder_closed_rad", "encoder_open_rad", "motor_closed_tick", "motor_open_tick"):
        v = getattr(args, name, None)
        if v is not None:
            m[name] = v
    if getattr(args, "no_clip", False):
        m["clip"] = False


def make_mapping(cfg: Dict[str, Dict[str, Any]]) -> GripperMapping:
    m = cfg["mapping"]
    return GripperMapping(
        motor_closed_tick=int(m["motor_closed_tick"]),
        motor_open_tick=int(m["motor_open_tick"]),
        encoder_closed_rad=float(m["encoder_closed_rad"]),
        encoder_open_rad=float(m["encoder_open_rad"]),
        clip=bool(m.get("clip", True)),
    )


def open_motor(cfg: Dict[str, Dict[str, Any]]):
    from .dxl import XM460

    d = cfg["dynamixel"]
    motor = XM460(d["port"], int(d["baudrate"]), int(d["id"])).open()
    print(f"[dxl] connected ID {d['id']} on {d['port']} @ {d['baudrate']} "
          f"(model number {motor.model_number})")
    return motor


def setup_motor(motor, cfg: Dict[str, Dict[str, Any]], mapping: GripperMapping) -> None:
    from . import dxl as X

    d = cfg["dynamixel"]
    motor.torque(False)
    mode = int(d["operating_mode"])
    motor.set_operating_mode(mode)
    lo, hi = X.POSITION_RANGE.get(mode, (None, None))
    if lo is not None:
        for name, tick in (("closed", mapping.motor_closed_tick), ("open", mapping.motor_open_tick)):
            if not lo <= tick <= hi:
                raise SystemExit(f"motor_{name}_tick={tick} is outside mode {mode} range [{lo}, {hi}]")
    if mode == X.MODE_POSITION:
        plo, phi = motor.position_limits()
        for name, tick in (("closed", mapping.motor_closed_tick), ("open", mapping.motor_open_tick)):
            if not plo <= tick <= phi:
                print(f"[warn] motor_{name}_tick={tick} is outside the motor's "
                      f"Min/Max Position Limit [{plo}, {phi}] - it will be clamped by the motor")
    if mode == X.MODE_CURRENT_BASED_POSITION:
        limit = motor.current_limit()
        gc = min(int(d["goal_current"]), limit)
        motor.set_goal_current(gc)
        print(f"[dxl] mode 5 (current-based position), goal current {gc} "
              f"(~{gc * X.CURRENT_UNIT_MA:.0f} mA, limit {limit})")
    else:
        print(f"[dxl] operating mode {mode}")
    motor.set_profile(int(d["approach_profile_velocity"]), int(d["profile_acceleration"]))
    motor.torque(True)


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> int:
    fmt = D.detect_format(args.data) if args.format == "auto" else args.format
    print(f"format: {fmt}")
    if fmt == "mcap":
        topics = D.list_mcap_topics(args.data)
        for name, t in sorted(topics.items()):
            mark = "  <- gripper" if name in D.SIDE_TOPICS.values() or name == D.COMBINED_TOPIC else ""
            print(f"  {name:55s} {t['schema']:35s} {t['count']:>8}{mark}")
    else:
        info = D.list_lerobot_keys(args.data)
        print(f"  fps={info['fps']} total_episodes={info['total_episodes']}")
        for k, v in info["features"].items():
            mark = "  <- gripper" if k.endswith("joint_states") else ""
            print(f"  {k:55s} shape={v['shape']} names={v['names']}{mark}")
    print()
    for side in D.SIDES:
        try:
            traj = D.load_trajectory(args.data, side=side, fmt=fmt, topic=args.topic,
                                     key=args.key, episode=args.episode)
            print(f"[{side}] {traj.summary()}")
        except Exception as e:  # noqa: BLE001 - informative only
            print(f"[{side}] not available: {e}")
    return 0


# ---------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------

def cmd_calibrate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    motor = open_motor(cfg)
    try:
        motor.torque(False)
        print("Torque OFF. Move the robot-side gripper by hand.\n")
        captured: Dict[str, int] = {}
        for label in ("closed", "open"):
            done = threading.Event()

            def wait_enter() -> None:
                try:
                    input()
                except EOFError:
                    pass
                done.set()

            print(f"Move to fully {label.upper()} and press Enter.")
            threading.Thread(target=wait_enter, daemon=True).start()
            pos = motor.present_position()
            while not done.is_set():
                pos = motor.present_position()
                deg = pos * 360.0 / TICKS_PER_REV
                sys.stdout.write(f"\r  present position: {pos:7d} tick ({deg:8.2f} deg)   ")
                sys.stdout.flush()
                time.sleep(0.05)
            captured[label] = pos
            print(f"\n  -> motor_{label}_tick = {pos}\n")
    finally:
        motor.close()

    stroke = abs(captured["open"] - captured["closed"]) * 360.0 / TICKS_PER_REV
    print(f"stroke: {stroke:.1f} deg")
    snippet = (
        "mapping:\n"
        f"  encoder_closed_rad: {cfg['mapping']['encoder_closed_rad']}\n"
        f"  encoder_open_rad: {cfg['mapping']['encoder_open_rad']}\n"
        f"  motor_closed_tick: {captured['closed']}\n"
        f"  motor_open_tick: {captured['open']}\n"
        f"  clip: true\n"
    )
    print("\nPut this into config/gripper.yaml:\n")
    print(snippet)
    if args.write:
        import yaml

        target = args.write
        out = {k: v for k, v in load_config(args.config).items() if not k.startswith("_")}
        out["dynamixel"].update({k: cfg["dynamixel"][k] for k in ("port", "baudrate", "id")})
        out["mapping"]["motor_closed_tick"] = captured["closed"]
        out["mapping"]["motor_open_tick"] = captured["open"]
        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        with open(target, "w", encoding="utf-8") as fp:
            yaml.safe_dump(out, fp, sort_keys=False)
        print(f"written to {target}")
    return 0


# ---------------------------------------------------------------------------
# goto
# ---------------------------------------------------------------------------

def cmd_goto(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    mapping = make_mapping(cfg)
    if args.tick is not None:
        tick = args.tick
    elif args.rad is not None:
        tick = mapping.to_tick(args.rad)
    else:
        tick = mapping.to_tick(mapping.encoder_open_rad if args.state == "open"
                               else mapping.encoder_closed_rad)
    motor = open_motor(cfg)
    try:
        setup_motor(motor, cfg, mapping)
        motor.set_goal_position(tick)
        pos = motor.wait_until(tick, tol=args.tolerance, timeout=args.timeout)
        print(f"goal {tick}, present {pos}, current {motor.present_current()}")
        if args.hold:
            input("Holding. Press Enter to release torque...")
    finally:
        try:
            motor.torque(False)
        finally:
            motor.close()
    return 0


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def _load_for_replay(args: argparse.Namespace) -> D.Trajectory:
    traj = D.load_trajectory(args.data, side=args.side, fmt=args.format, topic=args.topic,
                             key=args.key, episode=args.episode)
    print(traj.summary())
    if args.start or args.end is not None:
        traj = traj.trimmed(args.start, args.end)
        print(f"trimmed to [{args.start}, {args.end}] -> {len(traj.t)} samples")
    if args.rate:
        traj = D.resample(traj, args.rate)
        print(f"resampled to {args.rate} Hz -> {len(traj.t)} samples")
    return traj


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"log written to {path}")


def _plot(path: str, rows: List[Dict[str, Any]], title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping plot")
        return
    t = np.array([r["t_cmd"] for r in rows])
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axes[0].plot(t, [r["cmd_rad"] for r in rows], label="recorded encoder")
    if rows and rows[0].get("present_rad") not in (None, ""):
        axes[0].plot(t, [r["present_rad"] for r in rows], label="motor (encoder-equiv.)")
    axes[0].set_ylabel("angle [rad]")
    axes[0].legend()
    axes[1].plot(t, [r["cmd_tick"] for r in rows], label="goal")
    if rows and rows[0].get("present_tick") not in (None, ""):
        axes[1].plot(t, [r["present_tick"] for r in rows], label="present")
    axes[1].set_ylabel("position [tick]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"plot written to {path}")


def _report(rows: List[Dict[str, Any]], mapping: GripperMapping) -> None:
    err = np.array([r["present_tick"] - r["cmd_tick"] for r in rows
                    if r.get("present_tick") not in (None, "")], dtype=float)
    if err.size == 0:
        return
    rad_per_tick = abs(mapping.encoder_open_rad - mapping.encoder_closed_rad) / abs(
        mapping.motor_open_tick - mapping.motor_closed_tick)
    dt = np.diff([r["t_actual"] for r in rows])
    print("\n=== result ===")
    print(f"samples sent      : {len(rows)}")
    if dt.size:
        print(f"command period    : mean {dt.mean() * 1e3:.1f} ms, max {dt.max() * 1e3:.1f} ms")
    print(f"tracking error    : RMS {np.sqrt(np.mean(err ** 2)):.1f} tick, "
          f"max {np.abs(err).max():.0f} tick "
          f"(~{np.sqrt(np.mean(err ** 2)) * rad_per_tick:.3f} / "
          f"{np.abs(err).max() * rad_per_tick:.3f} rad encoder-equiv.)")
    cur = [r["current"] for r in rows if r.get("current") not in (None, "")]
    if cur:
        print(f"current           : max |{max(abs(c) for c in cur)}| (x2.69 mA)")


def cmd_replay(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    apply_overrides(cfg, args)
    mapping = make_mapping(cfg)
    traj = _load_for_replay(args)
    ticks = mapping.to_tick(traj.q)
    ticks = np.atleast_1d(ticks)

    outside = np.mean((traj.q < min(mapping.encoder_closed_rad, mapping.encoder_open_rad) - 1e-3)
                      | (traj.q > max(mapping.encoder_closed_rad, mapping.encoder_open_rad) + 1e-3))
    print(f"mapping: {mapping.encoder_closed_rad:.3f}..{mapping.encoder_open_rad:.3f} rad -> "
          f"{mapping.motor_closed_tick}..{mapping.motor_open_tick} tick "
          f"(stroke {mapping.motor_stroke_deg:.1f} deg); goal range {ticks.min()}..{ticks.max()}")
    if outside > 0:
        print(f"[warn] {outside * 100:.1f}% of samples are outside the encoder range "
              f"({'clipped' if mapping.clip else 'NOT clipped'}). "
              "Check encoder_closed_rad / encoder_open_rad.")

    stem = os.path.splitext(os.path.basename(os.path.normpath(args.data)))[0]
    log_path = args.log or os.path.join("logs", f"replay_{stem}_{args.side}_{time.strftime('%Y%m%d_%H%M%S')}.csv")

    if args.dry_run:
        rows = [{"t_cmd": float(t), "t_actual": float(t), "cmd_rad": float(q), "cmd_tick": int(k),
                 "present_tick": "", "present_rad": "", "current": ""}
                for t, q, k in zip(traj.t, traj.q, ticks)]
        _write_csv(log_path, rows)
        if args.plot:
            _plot(args.plot, rows, f"{traj.source} ({args.side}) dry-run")
        print("dry-run: motor not touched")
        return 0

    motor = open_motor(cfg)
    stop = {"flag": False}

    def on_sigint(_sig, _frm):
        stop["flag"] = True

    prev = signal.signal(signal.SIGINT, on_sigint)
    rows: List[Dict[str, Any]] = []
    try:
        setup_motor(motor, cfg, mapping)
        first = int(ticks[0])
        print(f"moving to start position {first} ...")
        motor.set_goal_position(first)
        pos = motor.wait_until(first, tol=args.tolerance, timeout=args.timeout)
        print(f"  present {pos}")
        if not args.yes:
            input(f"Press Enter to start replay ({traj.duration / args.speed:.1f} s, "
                  f"speed x{args.speed}). Ctrl+C to abort during replay.")
        d = cfg["dynamixel"]
        motor.set_profile(int(d["replay_profile_velocity"]), int(d["profile_acceleration"]))

        for loop in range(args.loops):
            t0 = time.perf_counter()
            for t, q, k in zip(traj.t, traj.q, ticks):
                if stop["flag"]:
                    break
                target = t0 + t / args.speed
                while True:
                    now = time.perf_counter()
                    if now >= target:
                        break
                    time.sleep(min(target - now, 0.002))
                motor.set_goal_position(int(k))
                row = {"t_cmd": float(t) / args.speed + loop * traj.duration / args.speed,
                       "t_actual": time.perf_counter() - t0 + loop * traj.duration / args.speed,
                       "cmd_rad": float(q), "cmd_tick": int(k)}
                if not args.no_feedback:
                    p = motor.present_position()
                    row.update(present_tick=p, present_rad=float(mapping.tick_to_rad(p)),
                               current=motor.present_current())
                else:
                    row.update(present_tick="", present_rad="", current="")
                rows.append(row)
                if args.verbose:
                    print(f"t={row['t_actual']:7.3f} cmd={q:6.3f} rad goal={k:6d} "
                          f"present={row['present_tick']}")
            if stop["flag"]:
                print("\naborted by user")
                break

        hw = motor.hardware_error()
        if hw:
            print(f"[warn] hardware error status: 0x{hw:02x}")
        print(f"temperature: {motor.temperature()} C")
    finally:
        signal.signal(signal.SIGINT, prev)
        try:
            if args.hold and not stop["flag"]:
                input("Holding last position. Press Enter to release torque...")
            motor.torque(False)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to disable torque: {e}")
        motor.close()

    _write_csv(log_path, rows)
    _report(rows, mapping)
    if args.plot:
        _plot(args.plot, rows, f"{traj.source} ({args.side})")
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def _add_dxl_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("dynamixel / mapping (override config)")
    g.add_argument("--config", help=f"YAML config (default: {DEFAULT_CONFIG_PATH} if present)")
    g.add_argument("--port")
    g.add_argument("--baudrate", type=int)
    g.add_argument("--id", dest="dxl_id", type=int)
    g.add_argument("--operating-mode", type=int, choices=[3, 4, 5])
    g.add_argument("--goal-current", type=int, help="mode 5 only, unit 2.69 mA")
    g.add_argument("--profile-velocity", type=int, help="replay profile velocity (0=unlimited)")
    g.add_argument("--encoder-closed-rad", type=float)
    g.add_argument("--encoder-open-rad", type=float)
    g.add_argument("--motor-closed-tick", type=int)
    g.add_argument("--motor-open-tick", type=int)
    g.add_argument("--no-clip", action="store_true", help="do not clip to calibrated range")


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("data", help=".mcap file, rosbag2 dir, LeRobot dataset dir or episode .parquet")
    p.add_argument("--format", choices=["auto", "mcap", "lerobot"], default="auto")
    p.add_argument("--topic", help="mcap topic (default: /yubi/gripper/<side>/joint_state)")
    p.add_argument("--key", default=D.DEFAULT_LEROBOT_KEY,
                   help="LeRobot column (observation.joint_states or action.joint_states)")
    p.add_argument("--episode", type=int, default=0, help="LeRobot episode index")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="yubi-gripper", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="show topics/features and gripper ranges in the data")
    _add_data_args(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("calibrate", help="torque off and capture closed/open motor ticks")
    _add_dxl_args(p)
    p.add_argument("--write", metavar="YAML", help="save a config with the captured ticks")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("goto", help="move to open/closed/angle once (motor check)")
    _add_dxl_args(p)
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--state", choices=["open", "closed"], default="closed")
    grp.add_argument("--rad", type=float, help="encoder-equivalent angle")
    grp.add_argument("--tick", type=int, help="raw goal position")
    p.add_argument("--tolerance", type=int, default=20)
    p.add_argument("--timeout", type=float, default=5.0)
    p.add_argument("--hold", action="store_true", help="keep torque until Enter")
    p.set_defaults(func=cmd_goto)

    p = sub.add_parser("replay", help="send recorded encoder angles as goal positions")
    _add_data_args(p)
    _add_dxl_args(p)
    p.add_argument("--side", choices=list(D.SIDES), default="left")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    p.add_argument("--start", type=float, default=0.0, help="start time [s]")
    p.add_argument("--end", type=float, help="end time [s]")
    p.add_argument("--rate", type=float, help="resample to fixed rate [Hz] (zero-order hold)")
    p.add_argument("--loops", type=int, default=1)
    p.add_argument("--dry-run", action="store_true", help="do not touch the motor")
    p.add_argument("--yes", "-y", action="store_true", help="do not wait for Enter before replay")
    p.add_argument("--hold", action="store_true", help="keep torque at the end until Enter")
    p.add_argument("--no-feedback", action="store_true", help="skip present position reads")
    p.add_argument("--tolerance", type=int, default=20, help="start-position tolerance [tick]")
    p.add_argument("--timeout", type=float, default=5.0, help="start-position timeout [s]")
    p.add_argument("--log", help="CSV log path (default logs/replay_*.csv)")
    p.add_argument("--plot", metavar="PNG", help="save command vs. present plot")
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(func=cmd_replay)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "speed", 1.0) <= 0:
        raise SystemExit("--speed must be > 0")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
