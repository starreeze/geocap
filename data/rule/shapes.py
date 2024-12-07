from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal, randint, uniform

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
    def from_dict(data: dict[str, Any]) -> "GSRule":
        shape_type = data.get("type", "")
        if shape_type == "polygon":
            return Polygon(
                points=data["points"], special_info=data.get("special_info", ""), fill_mode=data.get("fill_mode", "no")
            )
        elif shape_type in ["line", "segment", "ray"]:
            return Line(type=shape_type, points=data["points"])
        elif shape_type == "ellipse":
            return Ellipse(
                center=data["center"],
                major_axis=data.get("major_axis", 0),
                minor_axis=data.get("minor_axis", 0),
                rotation=data.get("rotation", 0),
                special_info=data.get("special_info", ""),
                fill_mode=data.get("fill_mode", "no"),
            )
        elif shape_type == "spiral":
            return Spiral(
                center=data["center"],
                initial_radius=data["initial_radius"],
                growth_rate=data["growth_rate"],
                sin_params=data["sin_params"],
                max_theta=data["max_theta"],
            )
        raise ValueError(f"Unknown shape type: {shape_type}")


@dataclass
class Polygon(GSRule):
    points: list[tuple[float, float]]
    special_info: str = ""
    fill_mode: Literal["no", "white", "black"] = "no"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "polygon"} | asdict(self)

    def get_bbox(self) -> list[tuple[float, float]]:
        return self.bbox_from_points(self.points)

    def get_area(self) -> float:
        n = len(self.points)
        area = 0.0
        for i in range(n):
            x_i, y_i = self.points[i]
            x_next, y_next = self.points[(i + 1) % n]
            area += x_i * y_next - x_next * y_i
        area = abs(area) / 2.0
        return area

    def get_centroid(self) -> tuple[float, float]:
        A = self.get_area()
        C_x = 0
        C_y = 0
        n = len(self.points)
        for i in range(n):
            x_i, y_i = self.points[i]
            x_next, y_next = self.points[(i + 1) % n]
            common_term = x_i * y_next - x_next * y_i
            C_x += (x_i + x_next) * common_term
            C_y += (y_i + y_next) * common_term
        C_x /= 6 * A
        C_y /= 6 * A
        return C_x, C_y

    def normalize_points(self):
        min_x = min(x for x, y in self.points)
        min_y = min(y for x, y in self.points)
        max_x = max(x for x, y in self.points)
        max_y = max(y for x, y in self.points)
        max_range = max(max_x - min_x, max_y - min_y) * 1.2
        if min_x < 0 or min_y < 0 or max_x > 1 or max_y > 1:
            self.points = [((x - min_x) / max_range, (y - min_y) / max_range) for x, y in self.points]

    def is_convex(self) -> bool:
        n = len(self.points)
        if n < 3:
            return False  # A polygon with fewer than 3 points is not a polygon

        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

        sign = None
        for i in range(n):
            o = self.points[i]
            a = self.points[(i + 1) % n]
            b = self.points[(i + 2) % n]
            cp = cross_product(o, a, b)
            if cp != 0:
                if sign is None:
                    sign = cp > 0
                elif sign != (cp > 0):
                    return False
        return True

    def check_angle(self, thres=0.15 * np.pi) -> bool:
        # Check if each angle is greater than thres
        def angle_between(v1, v2):
            return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

        angles = []
        for i in range(len(self.points)):
            v1 = np.array(self.points[i]) - np.array(self.points[i - 1])
            v2 = np.array(np.array(self.points[i]) - self.points[(i + 1) % len(self.points)])
            angles.append(angle_between(v1, v2))

        min_angle = min(angles)
        return min_angle > thres

    def to_simple_polygon(self):
        n = len(self.points)
        center = (
            sum([p[0] for p in self.points]) / n,
            sum([p[1] for p in self.points]) / n,
        )
        self.points.sort(key=lambda p: (polar_angle(center, p), -distance_2points(center, p)))

    def to_equilateral_triangle(self, side_len: float, rotation: float):
        self.special_info = "equilateral triangle"
        self.side_len = side_len

        x0, y0 = self.points[0]
        r = rotation
        x1 = x0 + side_len * np.cos(r)
        y1 = y0 + side_len * np.sin(r)
        x2 = x0 + side_len * np.cos(r + np.pi / 3)
        y2 = y0 + side_len * np.sin(r + np.pi / 3)
        self.points = [(x0, y0), (x1, y1), (x2, y2)]

    def to_rectangle(self, width: float, height: float, rotation: float):
        self.special_info += "rectangle"
        self.width = width
        self.height = height

        x0, y0 = self.points[0]
        r = rotation

        x1 = x0 + width * np.cos(r)
        y1 = y0 + width * np.sin(r)
        x2 = x1 - height * np.sin(r)
        y2 = y1 + height * np.cos(r)
        x3 = x0 - height * np.sin(r)
        y3 = y0 + height * np.cos(r)
        self.points = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]


@dataclass
class Line(GSRule):
    type: str  # line, segment, ray
    points: list[tuple[float, float]]  # two points determine the line

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def get_bbox(self) -> list[tuple[float, float]]:
        return self.bbox_from_points(self.points)

    def get_centroid(self) -> tuple[float, float]:
        x_coords = [point[0] for point in self.points]
        y_coords = [point[1] for point in self.points]
        centroid_x = sum(x_coords) / len(self.points)
        centroid_y = sum(y_coords) / len(self.points)
        return centroid_x, centroid_y

    def get_area(self) -> float:
        return 0

    def get_length(self) -> float:
        x1, y1 = self.points[0]
        x2, y2 = self.points[1]
        return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


@dataclass
class Ellipse(GSRule):
    center: tuple[float, float]
    major_axis: float = 0
    minor_axis: float = 0
    rotation: float = 0  # e.g., pi/3
    special_info: str = ""
    fill_mode: Literal["no", "white", "black"] = "no"

    def __post_init__(self):
        # Calculate points on the ellipse in counterclockwise order
        data_points = 1000
        t_values = np.linspace(0, 2 * np.pi, data_points)

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
        return np.pi * self.major_axis * self.minor_axis

    def get_centroid(self) -> tuple[float, float]:
        return self.center

    def get_point(self, theta=None) -> tuple[float, float]:
        if theta is None:
            theta = np.random.uniform(0, 2 * np.pi)
        theta = theta % (2 * np.pi)

        n = len(self.curve_points)
        if np.isclose(theta, 0.5 * np.pi):
            return self.curve_points[n // 4]
        elif np.isclose(theta, 1.5 * np.pi):
            return self.curve_points[int(3 * n // 4)]
        else:
            slope = np.tan(theta + self.rotation)
            intercept = self.center[1] - slope * self.center[0]
            line = (slope, intercept)

        quarter_size = len(self.curve_points) // 4
        quarter_idx = theta // (0.5 * np.pi)  # i_th curve
        start_idx = int(quarter_size * quarter_idx * 0.8)
        end_idx = int(quarter_size * (quarter_idx + 1) * 1.2)

        distances = [distance_point_to_line(point, line) for point in self.curve_points[start_idx:end_idx]]
        min_distance_idx = np.argmin(distances)
        point = self.curve_points[start_idx + min_distance_idx]

        return tuple(point)

    def get_point_stage1(self, theta=None) -> tuple[float, float]:
        if theta is None:
            theta = np.random.uniform(0, 2 * np.pi)

        # Parametric equations for the ellipse without rotation
        x = 0.5 * self.major_axis * np.cos(theta)
        y = 0.5 * self.minor_axis * np.sin(theta)

        # Rotate the point by the ellipse's rotation angle
        cos_angle = np.cos(self.rotation)
        sin_angle = np.sin(self.rotation)

        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle

        # Translate the point by the ellipse's center
        x_final = x_rot + self.center[0]
        y_final = y_rot + self.center[1]

        return (x_final, y_final)

    def get_tangent_line(self, point: tuple[float, float]) -> tuple[float, float]:
        a = 0.5 * self.major_axis
        b = 0.5 * self.minor_axis
        # Translate the point to the origin of the ellipse
        x, y = point[0] - self.center[0], point[1] - self.center[1]

        # Rotate the point to align the ellipse with the axes
        cos_angle = np.cos(-self.rotation)
        sin_angle = np.sin(-self.rotation)

        x_rot = x * cos_angle - y * sin_angle
        y_rot = x * sin_angle + y * cos_angle

        # Ensure the point is on the ellipse
        assert np.abs((x_rot / a) ** 2 + (y_rot / b) ** 2 - 1) < 1e-4, "The point is not on the ellipse"

        # Gradient of the ellipse at the point
        dx, dy = -y_rot / (b**2), x_rot / (a**2)

        # Rotate the gradient back
        dx_rot = dx * cos_angle + dy * sin_angle
        dy_rot = -dx * sin_angle + dy * cos_angle

        slope = dy_rot / dx_rot
        intercept = point[1] - slope * point[0]
        if slope > 1e5:
            slope = float("inf")
            intercept = point[0]
        return (slope, intercept)

    def to_circle(self, radius: float):
        self.special_info = "circle"
        self.radius = radius
        self.major_axis = self.radius * 2
        self.minor_axis = self.radius * 2
        self.rotation = 0
        self.adjust_curve_points()


# TODO: add more types
@dataclass
class Spiral(GSRule):
    # Archimedean spiral  r = a + b(θ)*θ
    # b(\theta) = b + ε sin(ωθ+φ)
    center: tuple[float, float]
    initial_radius: float  # a
    growth_rate: float  # b
    sin_params: list[float]
    max_theta: float  # max theta to plot, e.g. 4*pi means the spiral will make 2 turns

    def to_dict(self) -> dict[str, Any]:
        return {"type": "spiral"} | asdict(self)

    def get_bbox(self) -> list[tuple[float, float]]:
        points = []
        thetas = np.linspace(0, self.max_theta, 1000)
        for theta in thetas:
            r = self.radius(theta)
            x = self.center[0] + r * np.cos(theta)
            y = self.center[1] + r * np.sin(theta)
            points.append((x, y))
        return self.bbox_from_points(points)

    def get_area(self) -> float:
        return np.pi * (self.radius(self.max_theta) ** 2)

    def get_centroid(self) -> tuple[float, float]:
        return self.center

    def radius(self, theta: float) -> float:
        epsilon, omega, phi = self.sin_params
        return self.initial_radius + (self.growth_rate + epsilon * np.sin(omega * theta + phi)) * theta


@dataclass
class Fusiform(GSRule):
    # use symetric parabolas to generate a fusiform
    # y = 4*p * (x-x_0)^2 + c
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

        data_points = 1000
        x = np.linspace(0, 1, data_points)
        epsilon, omega, phi = self.sin_params
        sin_wave = epsilon * np.sin(omega * x + phi)
        y1 = 4 * self.focal_length * (x - self.x_offset) ** 2 + self.y_offset + sin_wave
        y2 = 2 * self.y_symmetric_axis - y1

        curve_points_upper = np.column_stack([x[::-1], y2[::-1]])  # counterclockwise
        curve_points_lower = np.column_stack([x, y1])
        self.curve_points = np.concatenate([curve_points_upper, curve_points_lower])

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

    def get_point(self, theta: float) -> tuple[float, float]:
        theta = theta % (2 * np.pi)

        n = len(self.curve_points)
        if np.isclose(theta, 0.5 * np.pi):
            return self.curve_points[n // 4]
        elif np.isclose(theta, 1.5 * np.pi):
            return self.curve_points[int(3 * n // 4)]
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
            return self.curve_points[n // 4]
        elif np.isclose(theta, 1.5 * np.pi):
            return self.curve_points[int(3 * n // 4)]

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
        self.num_points = 100
        # Ensure there are exactly 4 control points for a cubic curve
        assert len(self.control_points) == 4, "A cubic Bézier curve requires exactly 4 control points."

        # Unpack control points
        p0, p1, p2, p3 = self.control_points

        # Precompute curve points
        self.curve_points = self._compute_curve_points(p0, p1, p2, p3)

    def to_dict(self):
        return {"type": "curves", "control_points": [self.control_points]}

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


class ShapeGenerator:
    def __init__(self, rule_args) -> None:
        self.rule_args = rule_args
        self.opt_shapes = ["polygon", "line", "ellipse", "spiral"]

        self.shape_prob = []

        for shape_type in self.opt_shapes:
            shape_prob = eval(f"rule_args.{shape_type}_shape_level")
            self.shape_prob.append(shape_prob)
        self.shape_prob = [x / sum(self.shape_prob) for x in self.shape_prob]

    def __call__(self) -> Polygon | Line | Ellipse | Spiral:
        # Generate a shape(GSRule) randomly
        shape_type = np.random.choice(self.opt_shapes, p=self.shape_prob)
        if shape_type == "polygon":
            shape = self.generate_polygon()
        elif shape_type == "line":
            shape = self.generate_line()
        elif shape_type == "ellipse":
            shape = self.generate_ellipse()
        elif shape_type == "spiral":
            shape = self.generate_spiral()
        else:
            raise ValueError("invalid type of shape")
        return shape

    def generate_polygon(self, points: Optional[list[tuple[float, float]]] = None, max_points=6) -> Polygon:
        if points is None:
            num_points = randint(3, max_points + 1)  # at least 3 points
            points = [(uniform(0.2, 0.8), uniform(0.2, 0.8)) for _ in range(num_points)]

        polygon = Polygon(points)
        polygon.to_simple_polygon()
        while not polygon.is_convex() or polygon.get_area() < 0.01 or not polygon.check_angle():
            points = [(uniform(0.2, 0.8), uniform(0.2, 0.8)) for _ in range(num_points)]
            polygon = Polygon(points)
            polygon.to_simple_polygon()

        special_polygon = np.random.choice(["no", "rectangle", "equilateral triangle"])
        if special_polygon == "rectangle":
            width, height = (uniform(0.1, 0.6), uniform(0.1, 0.6))
            rotation = uniform(0, 2 * np.pi)
            polygon.to_rectangle(width, height, rotation)
        elif special_polygon == "equilateral triangle":
            side_len = uniform(0.1, 0.6)
            rotation = uniform(0, 2 * np.pi)
            polygon.to_equilateral_triangle(side_len, rotation)
        elif len(points) == 3:
            polygon.special_info += "triangle"

        polygon.normalize_points()
        return polygon

    def generate_line(self, points: Optional[list[tuple[float, float]]] = None, min_length=0.2) -> Line:
        if points is None:
            point1 = (uniform(0, 1), uniform(0, 1))
            point2 = (uniform(0, 1), uniform(0, 1))

        line_type = np.random.choice(["line", "segment", "ray"])
        line = Line(type=line_type, points=[point1, point2])
        while line.get_length() < min_length:
            point1 = (uniform(0, 1), uniform(0, 1))
            point2 = (uniform(0, 1), uniform(0, 1))
            line = Line(type=line_type, points=[point1, point2])
        return line

    def generate_ellipse(
        self,
        center: Optional[tuple[float, float]] = None,
        major_axis: float = 0.0,
        minor_axis: float = 0.0,
        rotation: float = 0.0,
    ) -> Ellipse:
        if center is not None:
            ellipse = Ellipse(center, major_axis, minor_axis, rotation)
            if major_axis == minor_axis:
                ellipse.to_circle(radius=0.5 * major_axis)
            return ellipse

        else:
            center = (uniform(0.2, 0.8), uniform(0.2, 0.8))
            major_axis = normal(0.5, 0.1)
            minor_axis = uniform(0.3 * major_axis, 0.95 * major_axis)
            rotation = uniform(0, np.pi)
            ellipse = Ellipse(center, major_axis, minor_axis, rotation)
            while ellipse.get_area() < 0.01:
                major_axis = normal(0.5, 0.1)
                minor_axis = uniform(0.3 * major_axis, 0.95 * major_axis)
                ellipse = Ellipse(center, major_axis, minor_axis, rotation)

            special_ellipse = np.random.choice(["no", "circle"])
            if special_ellipse == "circle":
                ellipse.to_circle(radius=0.5 * minor_axis)
            return ellipse

    def generate_spiral(self) -> Spiral:
        center = (uniform(0.2, 0.8), uniform(0.2, 0.8))
        max_theta = uniform(5 * np.pi, 12 * np.pi)
        initial_radius = normal(5e-4, 1e-4)
        growth_rate = normal(1e-2, 2e-3)
        epsilon = normal(1e-3, 2e-4)
        omega = normal(1, 0.2)
        phi = uniform(0, np.pi)
        sin_params = [epsilon, omega, phi]
        return Spiral(center, initial_radius, growth_rate, sin_params, max_theta)

    def generate_fusiform(self) -> Fusiform:
        focal_length = normal(0.3, 0.03)
        x_offset = normal(0.5, 0.01)
        y_offset = normal(0.4, 0.03)
        y_symmetric_axis = normal(0.5, 0.01)

        epsilon = normal(0.1, 0.02)
        omega = normal(3 * np.pi, 0.1)
        phi = normal(0, 0.01)
        sin_params = [epsilon, omega, phi]
        return Fusiform(focal_length, x_offset, y_offset, y_symmetric_axis, sin_params)

    def generate_initial_chamber(self) -> Ellipse:
        center = (0.5 + normal(0, 0.002), 0.5 + normal(0, 0.002))
        major_axis = max(0.02, normal(0.02, 6e-3))
        minor_axis = uniform(0.8 * major_axis, major_axis)
        rotation = uniform(0, np.pi)
        special_info = "initial chamber"
        return Ellipse(center, major_axis, minor_axis, rotation, special_info)

    def generate_axial_filling(self, num_volutions: int, rule_args) -> list[dict]:
        axial_filling = []

        for i in range(2):
            start_volution = randint(0, max(1, num_volutions // 4))

            if rule_args.overlap_axial_and_poles_folds:
                end_volution = num_volutions
            else:
                end_volution = randint(num_volutions // 2, num_volutions)

            start_angle_main = -normal(0.1, 0.02) * np.pi + i * np.pi
            end_angle_main = normal(0.1, 0.02) * np.pi + i * np.pi

            axial_filling_main = {
                "type": "main",
                "start_angle": start_angle_main,
                "end_angle": end_angle_main,
                "start_volution": start_volution,
                "end_volution": end_volution,
            }
            axial_filling.append(axial_filling_main)

            # generate extension of axial fillilng
            max_extend_angle1 = (start_angle_main - (i - 0.5) * np.pi) % (2 * np.pi)
            axial_filling_extend1 = {
                "type": "extension",
                "start_angle": start_angle_main - max_extend_angle1 * normal(0.6, 0.1),
                "end_angle": start_angle_main,
                "start_volution": 0,
                "end_volution": randint(end_volution, num_volutions + 1),
            }
            max_extend_angle2 = ((0.5 - i) * np.pi - end_angle_main) % (2 * np.pi)
            axial_filling_extend2 = {
                "type": "extension",
                "start_angle": end_angle_main,
                "end_angle": end_angle_main + max_extend_angle2 * normal(0.6, 0.1),
                "start_volution": 0,
                "end_volution": randint(end_volution, num_volutions + 1),
            }
            axial_filling.append(axial_filling_extend1)
            axial_filling.append(axial_filling_extend2)

        return axial_filling

    def generate_poles_folds(self, num_volutions: int, axial_filling: list, rule_args) -> list[dict]:
        poles_folds = []
        for i in range(2):
            if not rule_args.overlap_axial_and_poles_folds and axial_filling:
                start_volution = axial_filling[3 * i]["end_volution"]
            else:
                start_volution = randint(num_volutions // 2, num_volutions)
            end_volution = num_volutions
            start_angle = -normal(0.2, 0.03) * np.pi + i * np.pi
            end_angle = normal(0.2, 0.03) * np.pi + i * np.pi
            poles_folds.append(
                {
                    "start_angle": start_angle,
                    "end_angle": end_angle,
                    "start_volution": start_volution,
                    "end_volution": end_volution,
                }
            )
        return poles_folds
