from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .schemas import Vector3


@dataclass(frozen=True)
class FrameTransform:
    source_frame: str
    target_frame: str
    translation: Vector3 = (0.0, 0.0, 0.0)
    rotation_rpy: Vector3 = (0.0, 0.0, 0.0)

    def apply(self, point: Sequence[float]) -> Vector3:
        px, py, pz = (float(value) for value in point)
        rx, ry, rz = self.rotation_rpy
        rotated = _rotate_xyz((px, py, pz), rx, ry, rz)
        tx, ty, tz = self.translation
        return (rotated[0] + tx, rotated[1] + ty, rotated[2] + tz)

    def inverse(self) -> "FrameTransform":
        rx, ry, rz = self.rotation_rpy
        inverse_rotation = (-rx, -ry, -rz)
        inverted_translation = _rotate_xyz((-self.translation[0], -self.translation[1], -self.translation[2]), *inverse_rotation)
        return FrameTransform(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            translation=inverted_translation,
            rotation_rpy=inverse_rotation,
        )



def _rotate_xyz(point: Vector3, roll: float, pitch: float, yaw: float) -> Vector3:
    x, y, z = point
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    y1 = y * cr - z * sr
    z1 = y * sr + z * cr
    x1 = x

    x2 = x1 * cp + z1 * sp
    z2 = -x1 * sp + z1 * cp
    y2 = y1

    x3 = x2 * cy - y2 * sy
    y3 = x2 * sy + y2 * cy
    return (x3, y3, z2)
