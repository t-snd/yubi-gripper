"""Minimal Dynamixel XM460-W350-R driver (Protocol 2.0) on top of dynamixel_sdk.

Control table addresses (X-series, see ROBOTIS e-Manual for XM460-W350):
https://emanual.robotis.com/docs/en/dxl/x/xm540-w270/ (same X-series layout)
"""

from __future__ import annotations

import time
from typing import Optional

# EEPROM
ADDR_MODEL_NUMBER = 0        # 2 bytes
ADDR_OPERATING_MODE = 11     # 1 byte
ADDR_CURRENT_LIMIT = 38      # 2 bytes
ADDR_MAX_POSITION_LIMIT = 48 # 4 bytes
ADDR_MIN_POSITION_LIMIT = 52 # 4 bytes
# RAM
ADDR_TORQUE_ENABLE = 64      # 1 byte
ADDR_HARDWARE_ERROR = 70     # 1 byte
ADDR_GOAL_CURRENT = 102      # 2 bytes
ADDR_PROFILE_ACCELERATION = 108  # 4 bytes
ADDR_PROFILE_VELOCITY = 112  # 4 bytes
ADDR_GOAL_POSITION = 116     # 4 bytes
ADDR_PRESENT_CURRENT = 126   # 2 bytes
ADDR_PRESENT_POSITION = 132  # 4 bytes
ADDR_PRESENT_TEMPERATURE = 146  # 1 byte

MODE_POSITION = 3
MODE_EXTENDED_POSITION = 4
MODE_CURRENT_BASED_POSITION = 5

POSITION_RANGE = {
    MODE_POSITION: (0, 4095),
    MODE_EXTENDED_POSITION: (-1_048_575, 1_048_575),
    MODE_CURRENT_BASED_POSITION: (-1_048_575, 1_048_575),
}

CURRENT_UNIT_MA = 2.69          # Goal/Present Current unit
PROFILE_VELOCITY_UNIT_RPM = 0.229


def _signed(value: int, bits: int) -> int:
    if value & (1 << (bits - 1)):
        value -= 1 << bits
    return value


class DynamixelError(RuntimeError):
    pass


class XM460:
    def __init__(self, port: str, baudrate: int = 57600, dxl_id: int = 1):
        from dynamixel_sdk import PacketHandler, PortHandler

        self.port_name = port
        self.baudrate = baudrate
        self.id = dxl_id
        self.port = PortHandler(port)
        self.packet = PacketHandler(2.0)
        self._opened = False

    # -- connection ---------------------------------------------------------
    def open(self) -> "XM460":
        if not self.port.openPort():
            raise DynamixelError(f"failed to open {self.port_name}")
        if not self.port.setBaudRate(self.baudrate):
            raise DynamixelError(f"failed to set baudrate {self.baudrate}")
        self._opened = True
        model, comm, err = self.packet.ping(self.port, self.id)
        self._check(comm, err, f"ping ID {self.id}")
        self.model_number = model
        return self

    def close(self) -> None:
        if self._opened:
            self.port.closePort()
            self._opened = False

    def __enter__(self) -> "XM460":
        return self.open()

    def __exit__(self, *exc) -> None:
        try:
            self.torque(False)
        except Exception:
            pass
        self.close()

    # -- low level ----------------------------------------------------------
    def _check(self, comm: int, err: int, what: str) -> None:
        from dynamixel_sdk import COMM_SUCCESS

        if comm != COMM_SUCCESS:
            raise DynamixelError(f"{what}: {self.packet.getTxRxResult(comm)}")
        if err != 0:
            # Hardware-alert bit (0x80) is reported here; surface it but keep a
            # readable message.
            raise DynamixelError(f"{what}: {self.packet.getRxPacketError(err)}")

    def _w1(self, addr: int, v: int, what: str) -> None:
        comm, err = self.packet.write1ByteTxRx(self.port, self.id, addr, int(v))
        self._check(comm, err, what)

    def _w2(self, addr: int, v: int, what: str) -> None:
        comm, err = self.packet.write2ByteTxRx(self.port, self.id, addr, int(v) & 0xFFFF)
        self._check(comm, err, what)

    def _w4(self, addr: int, v: int, what: str) -> None:
        comm, err = self.packet.write4ByteTxRx(self.port, self.id, addr, int(v) & 0xFFFFFFFF)
        self._check(comm, err, what)

    def _r1(self, addr: int, what: str) -> int:
        v, comm, err = self.packet.read1ByteTxRx(self.port, self.id, addr)
        self._check(comm, err, what)
        return v

    def _r2(self, addr: int, what: str) -> int:
        v, comm, err = self.packet.read2ByteTxRx(self.port, self.id, addr)
        self._check(comm, err, what)
        return v

    def _r4(self, addr: int, what: str) -> int:
        v, comm, err = self.packet.read4ByteTxRx(self.port, self.id, addr)
        self._check(comm, err, what)
        return v

    # -- high level ---------------------------------------------------------
    def torque(self, on: bool) -> None:
        self._w1(ADDR_TORQUE_ENABLE, 1 if on else 0, "torque enable")

    def operating_mode(self) -> int:
        return self._r1(ADDR_OPERATING_MODE, "read operating mode")

    def set_operating_mode(self, mode: int) -> None:
        """Requires torque off (EEPROM)."""
        if self.operating_mode() != mode:
            self._w1(ADDR_OPERATING_MODE, mode, "operating mode")

    def current_limit(self) -> int:
        return self._r2(ADDR_CURRENT_LIMIT, "read current limit")

    def position_limits(self) -> tuple:
        lo = _signed(self._r4(ADDR_MIN_POSITION_LIMIT, "read min pos limit"), 32)
        hi = _signed(self._r4(ADDR_MAX_POSITION_LIMIT, "read max pos limit"), 32)
        return lo, hi

    def set_goal_current(self, value: int) -> None:
        self._w2(ADDR_GOAL_CURRENT, value, "goal current")

    def set_profile(self, velocity: int, acceleration: int = 0) -> None:
        self._w4(ADDR_PROFILE_ACCELERATION, acceleration, "profile acceleration")
        self._w4(ADDR_PROFILE_VELOCITY, velocity, "profile velocity")

    def set_goal_position(self, tick: int) -> None:
        self._w4(ADDR_GOAL_POSITION, tick, "goal position")

    def present_position(self) -> int:
        return _signed(self._r4(ADDR_PRESENT_POSITION, "present position"), 32)

    def present_current(self) -> int:
        return _signed(self._r2(ADDR_PRESENT_CURRENT, "present current"), 16)

    def temperature(self) -> int:
        return self._r1(ADDR_PRESENT_TEMPERATURE, "temperature")

    def hardware_error(self) -> int:
        return self._r1(ADDR_HARDWARE_ERROR, "hardware error status")

    def wait_until(self, tick: int, tol: int = 20, timeout: float = 5.0) -> Optional[int]:
        """Wait until present position is within ``tol`` of ``tick``."""
        t0 = time.monotonic()
        pos = None
        while time.monotonic() - t0 < timeout:
            pos = self.present_position()
            if abs(pos - tick) <= tol:
                return pos
            time.sleep(0.02)
        return pos
