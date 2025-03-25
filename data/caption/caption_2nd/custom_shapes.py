from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal, randint, uniform
from scipy.integrate import quad

from common.args import rule_args
from data.rule.utils import distance_2points, distance_point_to_line, polar_angle


@dataclass
class GSRule(ABC):
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        # __json__ = to_dict
        pass

    @abstractmethod
    def get_bbox(self) -> list[tuple[float, float]]:
        # return bounding box in form of xyxy
        pass

    @abstractmethod
    def get_area(self) -> float:
        pass

    @abstractmethod
    def get_centroid(self) -> tuple[float, float]:
        pass

    @staticmethod
    def bbox_from_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        min_x = min(x for x, y in points)
        max_x = max(x for x, y in points)
        min_y = min(y for x, y in points)
        max_y = max(y for x, y in points)
        return [(min_x, max_y), (max_x, min_y)]

    @staticmethod
    def getDim(point, center):
        x, y = point
        cx, cy = center
        if x > cx and y > cy:
            return 1
        elif x < cx and y > cy:
            return 2
        elif x < cx and y < cy:
            return 3
        elif x > cx and y < cy:
            return 4
        raise ValueError("getDim bug!")


@dataclass
class Ellipse(GSRule):
    support_point: tuple[float, float]
    center: tuple[float, float]
    major_axis: float = 0
    minor_axis: float = 0
    rotation: float = 0  # e.g., pi/3
    special_info: str = ""
    fill_mode: Literal["no", "white", "black"] = "no"

    def __post_init__(self):
        # Calculate points on the ellipse in counterclockwise order
        data_points = 30
        dim = GSRule.getDim(self.support_point, self.center)
        t_values = np.linspace(0 + (dim - 1) * (1 / 2 * np.pi), dim * 1 / 2 * np.pi, data_points)

        x_coords = (
            self.center[0]
            + 0.5 * self.major_axis * np.cos(t_values) * np.cos(self.rotation)
            - 0.5 * self.minor_axis * np.sin(t_values) * np.sin(self.rotation)
        )
        y_coords = (
            self.center[1]
            + 0.5 * self.major_axis * np.cos(t_values) * np.sin(self.rotation)
            + 0.5 * self.minor_axis * np.sin(t_values) * np.cos(self.rotation)
        )

        self.curve_points = np.column_stack((x_coords, y_coords))
        self.curve_points_recommended = self.curve_points

    def adjust_curve_points(self):
        self.__post_init__()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "ellipse"} | asdict(self)

    def get_bbox(self) -> list[tuple[float, float]]:
        h, k = self.center
        a = 0.5 * self.major_axis
        b = 0.5 * self.minor_axis

        # Rotation matrix
        cos_r = np.cos(self.rotation)
        sin_r = np.sin(self.rotation)

        # Corner points in the local (rotated) coordinate system
        points = [
            (a * cos_r, a * sin_r),  # (a, 0) rotated
            (-a * cos_r, -a * sin_r),  # (-a, 0) rotated
            (b * -sin_r, b * cos_r),  # (0, b) rotated
            (-b * -sin_r, -b * cos_r),  # (0, -b) rotated
        ]
        transformed_points = [(h + x, k + y) for x, y in points]
        return self.bbox_from_points(transformed_points)

    def get_area(self) -> float:
        return np.pi * self.major_axis * self.minor_axis / 4

    def get_centroid(self) -> tuple[float, float]:
        return self.center


@dataclass
class Fusiform(GSRule):
    # use symetric parabolas to generate a fusiform
    # y = 4*p * (x-x_0)^2 + c
    support_point: tuple[float, float]
    focal_length: float = 0.0  # p
    x_offset: float = 0.0  # x_0
    y_offset: float = 0.0  # c
    y_symmetric_axis: float = 0.0

    # add sine wave
    sin_params: list[float] = field(default_factory=list)

    x_start: float = 0.0
    x_end: float = 1.0
    center: tuple[float, float] = field(init=False)
    ratio: float = field(init=False)
    special_info: str = ""
    fill_mode: Literal["no", "white", "black"] = "no"

    def __post_init__(self):
        self.center = (self.x_offset, self.y_symmetric_axis)

        data_points = 30
        dim = GSRule.getDim(self.support_point, self.center)
        if dim == 1 or dim == 4:
            x = np.linspace(0.5, 1, data_points)
        else:
            x = np.linspace(0, 0.5, data_points)
        epsilon, omega, phi = self.sin_params
        sin_wave = epsilon * np.sin(omega * x + phi)
        y1 = 4 * self.focal_length * (x - self.x_offset) ** 2 + self.y_offset + sin_wave
        y2 = 2 * self.y_symmetric_axis - y1

        if dim == 1 or dim == 2:
            self.curve_points = np.column_stack([x[::-1], y2[::-1]])
        else:
            self.curve_points = np.column_stack([x, y1])

        self.curve_points_recommended = self.curve_points

        delta_y = np.abs(y2 - y1)
        min_delta = np.min(delta_y)
        if min_delta > 1e-2:  # no intersection
            self.ratio = float("inf")
            return

        left_intersection = np.argmin(delta_y[: data_points // 2])
        right_intersection = np.argmin(delta_y[data_points // 2 :]) + data_points // 2

        self.x_start = x[left_intersection]
        self.x_end = x[right_intersection]
        self.width = abs(self.x_end - self.x_start)

        self.height = 2 * (self.y_symmetric_axis - self.y_offset - sin_wave.min())
        self.ratio = self.width / self.height if self.height != 0 else float("inf")

    def adjust_curve_points(self):
        self.__post_init__()

    def to_dict(self) -> dict[str, Any]:
        return {"type": "fusiform_1"} | asdict(self)

    def get_bbox(self) -> list[tuple[float, float]]:
        y_max = self.y_symmetric_axis + 0.5 * self.height
        y_min = self.y_symmetric_axis - 0.5 * self.height
        return [(self.x_start, y_max), (self.x_end, y_min)]

    def get_area(self) -> float:
        raise NotImplementedError()

    def get_centroid(self) -> tuple[float, float]:
        raise NotImplementedError()

    def is_closed(self) -> bool:
        return self.ratio < 1e3


@dataclass
class Fusiform_2(GSRule):
    # use symetric parabola-like curves to generate a fusiform
    # x = 4*p * (y - y_0) ^ m + x_0 => y = ((x-x_0) / 4*p) ** (1/m) + y_0
    focal_length: float = 0.0  # p
    x_offset: float = 0.0  # x_0
    y_offset: float = 0.0  # y_0
    power: float = 0.0  # m
    x_symmetric_axis: float = 0.0

    # add sine wave
    sin_params: list[float] = field(default_factory=list)

    x_start: float = 0.0
    x_end: float = 1.0
    center: tuple[float, float] = field(init=False)
    ratio: float = field(init=False)
    special_info: str = ""
    fill_mode: Literal["no", "white", "black"] = "no"

    def __post_init__(self, first_init=True):
        self.center = (self.x_symmetric_axis, self.y_offset)

        assert self.x_offset < self.x_symmetric_axis
        data_points = 1000
        self.x_start = self.x_offset
        self.x_end = 2 * self.x_symmetric_axis - self.x_offset
        if self.x_start == 0.0 or self.x_end > 1.0:
            self.ratio = float("inf")
            return

        self.width = self.x_end - self.x_start
        if first_init:  # adjust omega
            self.sin_params[1] = self.sin_params[1] / self.width

        x = np.linspace(self.x_start, self.x_end, data_points)
        x_left = x[: data_points // 2]

        y_left = ((x_left - self.x_offset) / (4 * self.focal_length)) ** (1 / self.power) + self.y_offset
        y_right = np.flip(y_left)

        epsilon, omega, phi = self.sin_params
        sin_wave = epsilon * np.sin(omega * (x - self.x_start) + phi)
        y1 = np.concatenate([y_left, y_right]) + sin_wave
        y2 = 2 * self.y_offset - y1

        curve_points_upper = np.column_stack([x[::-1], y1[::-1]])  # counterclockwise
        curve_points_lower = np.column_stack([x, y2])
        self.curve_points = np.concatenate([curve_points_upper, curve_points_lower])

        self.height = max(y1) - min(y2)
        self.ratio = self.width / self.height if self.height != 0 else float("inf")

    def adjust_curve_points(self):
        self.__post_init__(first_init=False)

    def to_dict(self) -> dict[str, Any]:
        return {"type": "fusiform_2"} | asdict(self)

    def get_bbox(self) -> list[tuple[float, float]]:
        y_max = self.y_offset + 0.5 * self.height
        y_min = self.y_offset - 0.5 * self.height
        return [(self.x_start, y_max), (self.x_end, y_min)]

    def get_area(self) -> float:
        raise NotImplementedError()

    def get_centroid(self) -> tuple[float, float]:
        raise NotImplementedError()

    def is_closed(self) -> bool:
        return self.ratio < 1e3

    def get_point(self, theta: float) -> tuple[float, float]:
        theta = theta % (2 * np.pi)
        n = len(self.curve_points)
        if np.isclose(theta, 0.5 * np.pi):
            return tuple(self.curve_points[n // 4])
        elif np.isclose(theta, 1.5 * np.pi):
            return tuple(self.curve_points[int(3 * n // 4)])

        slope = np.tan(theta)
        intercept = self.center[1] - slope * self.center[0]
        line = (slope, intercept)

        quarter_size = len(self.curve_points) // 4
        quarter_idx = theta // (0.5 * np.pi)  # i_th curve
        start_idx = int(quarter_size * quarter_idx)
        end_idx = int(quarter_size * (quarter_idx + 1))
        distances = [distance_point_to_line(point, line) for point in self.curve_points[start_idx:end_idx]]
        min_distance_idx = np.argmin(distances)
        point = self.curve_points[start_idx + min_distance_idx]

        return tuple(point)


@dataclass
class Curve:
    """Cubic Bézier curve for controllable shapes."""

    control_points: list[tuple[float, float]]
    special_info: str = ""
    fill_mode: Literal["no", "white", "black"] = "no"

    def __post_init__(self):
        self.num_points = 30
        # Ensure there are exactly 4 control points for a cubic curve
        assert len(self.control_points) == 4, "A cubic Bézier curve requires exactly 4 control points."

        # Unpack control points
        p0, p1, p2, p3 = self.control_points

        # Precompute curve points
        self.curve_points = self._compute_curve_points(p0, p1, p2, p3)

    def to_dict(self):
        return {"type": "curves"} | asdict(self)

    def _compute_curve_points(self, p0, p1, p2, p3):
        """Computes points along the cubic Bézier curve"""
        curve_points = []
        t_values = np.linspace(0, 1, self.num_points)

        for t in t_values:
            one_minus_t = 1 - t
            point = (
                one_minus_t**3 * np.array(p0)
                + 3 * one_minus_t**2 * t * np.array(p1)
                + 3 * one_minus_t * t**2 * np.array(p2)
                + t**3 * np.array(p3)
            )
            curve_points.append(tuple(point))

        return curve_points

    def plot_curve(self, figure_id=0):
        plt.figure(figure_id)
        # Separate the points into x and y components
        curve_points = np.array(self.curve_points)
        x_vals = curve_points[:, 0]
        y_vals = curve_points[:, 1]
        plt.plot(x_vals, y_vals)

        # Plot the control points
        control_x_vals, control_y_vals = zip(*self.control_points)
        plt.plot(control_x_vals, control_y_vals, "ro--")


@dataclass
class CustomedShape(GSRule):
    """Customed shape composed of 4 bezier curves"""

    curves: list[Curve]
    support_point: tuple[float, float]

    vertices: list[tuple[float, float]] = field(init=False)
    center: tuple[float, float] = field(init=False)
    ratio: float = field(init=False)
    special_info: str = ""
    fill_mode: Literal["no", "white", "black"] = "no"

    def __post_init__(self):
        # Verify that the shape is closed
        assert len(self.curves) == 4, "CustomedShape has to consist 4 curves"
        for i in range(4):
            assert (
                self.curves[i].control_points[-1] == self.curves[(i + 1) % 4].control_points[0]
            ), f"Curve {i + 1} does not connect to Curve {i}"

        self.vertices = [curve.control_points[0] for curve in self.curves]
        self.curve_points = np.concatenate([curve.curve_points for curve in self.curves], axis=0)

        center = np.mean(self.curve_points, axis=0)
        self.center = tuple(center)
        dim = GSRule.getDim(self.support_point, center)
        self.curve_points_recommended = self.curves[dim - 1].curve_points

        self.width = abs(self.vertices[0][0] - self.vertices[2][0])
        self.height = abs(self.vertices[1][1] - self.vertices[3][1])
        self.ratio = self.width / self.height if self.height != 0 else float("inf")

    def adjust_curve_points(self):
        self.__post_init__()

    def to_dict(self) -> dict[str, Any]:
        all_dict = asdict(self)

        control_points_list = []
        for curve_dict in all_dict["curves"]:
            control_points = curve_dict["control_points"]
            control_points_list.append(control_points)

        new_dict = {k: v for k, v in asdict(self).items() if "curves" not in k}
        return {"type": "curves"} | {"control_points": control_points_list} | new_dict

    def get_bbox(self) -> list[tuple[float, float]]:
        x_min = self.vertices[2][0]
        y_max = self.vertices[1][1]
        x_max = self.vertices[0][0]
        y_min = self.vertices[3][1]
        return [(x_min, y_max), (x_max, y_min)]

    def get_area(self) -> float:
        raise NotImplementedError()

    def get_centroid(self) -> tuple[float, float]:
        raise NotImplementedError()

    def get_point(self, theta: float) -> tuple[float, float]:
        theta = theta % (2 * np.pi)
        if np.isclose(theta, 0.5 * np.pi):
            return self.vertices[1]
        elif np.isclose(theta, 1.5 * np.pi):
            return self.vertices[3]
        else:
            slope = np.tan(theta)
            intercept = self.center[1] - slope * self.center[0]
            line = (slope, intercept)

        quarter_size = len(self.curve_points) // 4
        quarter_idx = theta // (0.5 * np.pi)  # i_th curve
        start_idx = int(quarter_size * quarter_idx)
        end_idx = int(quarter_size * (quarter_idx + 1))
        distances = [distance_point_to_line(point, line) for point in self.curve_points[start_idx:end_idx]]
        min_distance_idx = np.argmin(distances)
        point = self.curve_points[start_idx + min_distance_idx]

        return tuple(point)
