"""KEY10 车头差速。子机只上报相对角，差速在这里算。

正 z = 车头相对按下时往左偏，回的差速为负。
死区内不再推，只保留已经记下的歪轮差速。差速封顶与子机执行层一致。
"""

from __future__ import annotations


class ClimbTrim:
    DZ = 0.6
    K = 6.0
    KI = 4.0
    ILIM = 12.0
    SLOW = 16.0

    def __init__(self) -> None:
        self.i = 0.0
        self.last_t = None

    def reset(self) -> None:
        self.i = 0.0
        self.last_t = None

    def step(self, z_deg: float, now: float) -> float:
        dt = 0.05
        if self.last_t is not None:
            dt = now - self.last_t
        if dt <= 0.0 or dt > 0.5:
            self.i = 0.0
            dt = 0.05
        self.last_t = now

        if z_deg > self.DZ:
            err = self.DZ - z_deg
        elif z_deg < -self.DZ:
            err = -self.DZ - z_deg
        else:
            err = 0.0
        if err != 0.0:
            self.i += err * self.KI * dt
            if self.i > self.ILIM:
                self.i = self.ILIM
            elif self.i < -self.ILIM:
                self.i = -self.ILIM
        steer = self.K * err + self.i
        if steer > self.SLOW:
            steer = self.SLOW
        elif steer < -self.SLOW:
            steer = -self.SLOW
        return steer
