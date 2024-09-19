from typing import Optional, Union
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from numpy.random import randint, uniform, normal
from data.utils import distance_2points


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

    def bbox_from_points(self, points: list[tuple[float, float]]) -> list[tuple[float, float]]:
        min_x = min(x for x, y in points)
        max_x = max(x for x, y in points)
        min_y = min(y for x, y in points)
        max_y = max(y for x, y in points)
        return [(min_x, max_y), (max_x, min_y)]


@dataclass
class Polygon(GSRule):
    points: list[tuple[float, float]]
    special_info: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"type": "polygon"} | asdict(self)

    def get_bbox(self) -> list[tuple[float, float]]:
        return self.bbox_from_points(self.points)

    def get_area(self) -> float:
        if "triangle" in self.special_info:
            A, B, C = self.points
            a = distance_2points(B, C)
            b = distance_2points(A, C)
            c = distance_2points(A, B)
            s = (a + b + c) / 2
            area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        else:
            bbox = self.get_bbox()
            area = (bbox[1][0] - bbox[0][0]) * (bbox[0][1] - bbox[1][1])
        return area

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
            return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

        angles = []
        for i in range(len(self.points)):
            v1 = np.array(self.points[i]) - np.array(self.points[i - 1])
            v2 = np.array(np.array(self.points[i]) - self.points[(i + 1) % len(self.points)])
            angles.append(angle_between(v1, v2))

        min_angle = min(angles)
        return min_angle > thres

    def to_simple_polygon(self):
        from .utils import polar_angle, distance_2points

        n = len(self.points)
        center = (
            sum([p[0] for p in self.points]) / n,
            sum([p[1] for p in self.points]) / n,
        )
        self.points.sort(key=lambda p: (polar_angle(center, p), -distance_2points(center, p)))

    def to_equilateral_triangle(self, side_len: float, rotation: float):
        self.special_info = "equilateral triangle. "
        self.side_len = side_len

        x0, y0 = self.points[0]
        r = rotation
        x1 = x0 + side_len * np.cos(r)
        y1 = y0 + side_len * np.sin(r)
        x2 = x0 + side_len * np.cos(r + np.pi / 3)
        y2 = y0 + side_len * np.sin(r + np.pi / 3)
        self.points = [(x0, y0), (x1, y1), (x2, y2)]

    def to_rectangle(self, width: float, height: float, rotation: float):
        self.special_info += "rectangle. "
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

    def get_point(self, theta=None) -> tuple[float, float]:
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
        assert np.isclose((x_rot / a) ** 2 + (y_rot / b) ** 2, 1), "The point is not on the ellipse"

        # Gradient of the ellipse at the point
        dx, dy = -y_rot / (b**2), x_rot / (a**2)

        # Rotate the gradient back
        dx_rot = dx * cos_angle + dy * sin_angle
        dy_rot = -dx * sin_angle + dy * cos_angle

        slope = dy_rot / dx_rot
        intercept = point[1] - slope * point[0]
        return (slope, intercept)

    def to_circle(self, radius: float):
        self.special_info = "circle. "
        self.radius = radius
        self.major_axis = self.radius * 2
        self.minor_axis = self.radius * 2
        self.rotation = 0


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

    # focal_length_2: float = 0.0
    # x_offset_2: float = 0.0
    # y_offset_2: float = 0.0

    x_start: float = 0.0
    x_end: float = 1.0
    center: tuple[float, float] = field(init=False)
    ratio: float = field(init=False)
    special_info: str = ""

    precision: float = 1e-2

    def __post_init__(self):
        self.center = (self.x_offset, self.y_symmetric_axis)

        self.data_points = int(1000 / self.precision)
        x = np.linspace(0, 1, self.data_points)
        epsilon, omega, phi = self.sin_params
        sin_wave = epsilon * np.sin(omega * x + phi)
        y1 = 4 * self.focal_length * (x - self.x_offset) ** 2 + self.y_offset + sin_wave
        y2 = 2 * self.y_symmetric_axis - y1

        close_indices = np.where(np.abs(y1 - y2) < self.precision)[0]
        if len(close_indices) > 0:
            left_intersection = int(close_indices[close_indices < self.data_points // 2][-1])
            right_intersection = int(close_indices[close_indices > self.data_points // 2][0])

            self.x_start = x[left_intersection]
            self.x_end = x[right_intersection]
            width = abs(self.x_end - self.x_start)

            height = 2 * (self.y_symmetric_axis - self.y_offset - sin_wave.min())
            self.ratio = width / height if height != 0 else float("inf")
        else:
            self.ratio = float("inf")

        """
        v0 for two parabolas to generate a fusiform
        """
        # center_x = 0.5 * (self.x_offset_1 + self.x_offset_2)
        # center_y = 0.5 * (self.y_offset_1 + self.y_offset_2)
        # self.center = (center_x, center_y)

        # # Ensure two parabolas have intersections
        # assert self.focal_length_1 > 0 and self.focal_length_2 < 0
        # assert self.y_offset_1 < self.y_offset_2

        # # Calculate the intersection points by solving the quadratic equation y1 = y2
        # a = self.focal_length_1 - self.focal_length_2
        # b = 2 * (self.x_offset_2 * self.focal_length_2 - self.x_offset_1 * self.focal_length_1)
        # c = (
        #     self.focal_length_1 * self.x_offset_1**2
        #     + self.y_offset_1
        #     - self.focal_length_2 * self.x_offset_2**2
        #     - self.y_offset_2
        # )

        # discriminant = b**2 - 4 * a * c
        # x1 = (-b + np.sqrt(discriminant)) / (2 * a)
        # x2 = (-b - np.sqrt(discriminant)) / (2 * a)
        # width = abs(x2 - x1)

        # height = abs(self.y_offset_2 - self.y_offset_1)
        # self.ratio = width / height if height != 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "fusiform_1"} | asdict(self)

    def get_bbox(self) -> list[tuple[float, float]]:
        raise NotImplementedError

    def is_closed(self) -> bool:
        return self.ratio < 1e3

    def get_point(self, theta: float) -> tuple[float, float]:
        theta = theta % (2 * np.pi)
        x_range = np.linspace(-0.5, 0.5, self.data_points)
        epsilon, omega, phi = self.sin_params
        sin_wave = epsilon * np.sin(omega * (x_range + self.x_offset) + phi)
        y_parabola = 4 * self.focal_length * (x_range**2) + self.y_offset + sin_wave - self.y_symmetric_axis

        y_max, y_min = max(-y_parabola), min(y_parabola)

        # Calculate points on the ray(with polar angle = theta)
        if np.abs(theta - 0.5 * np.pi) < self.precision or np.abs(theta - 1.5 * np.pi) < self.precision:
            y_line = None
        else:
            slope = np.tan(theta)
            y_line = slope * x_range

        # Calculate points on the fusiform(before offset)
        if 0 <= theta < np.pi:  # upper parabola
            y_fusiform = -y_parabola
            vertex_indice = np.where(y_fusiform == y_max)[0]
        else:  # lower parabola
            y_fusiform = y_parabola
            vertex_indice = np.where(y_fusiform == y_min)[0]

        if y_line is not None:
            intersection_indices = np.where(np.abs(y_fusiform - y_line) < self.precision)[0]
        else:
            intersection_indices = vertex_indice

        if 0.5 * np.pi <= theta < 1.5 * np.pi:
            idx = int(intersection_indices[intersection_indices < self.data_points // 1.8][0])
        else:
            idx = int(intersection_indices[intersection_indices > self.data_points // 2.2][-1])

        x = x_range[idx]
        y = y_fusiform[idx]

        return (x + self.center[0], y + self.center[1])


@dataclass
class Fusiform_2(GSRule):
    # use symetric parabola-like curves to generate a fusiform
    # x = 4*p * (y - y_0) ^ m + c => y = ((x-c) / 4*p) ** (1/m) + y_0
    focal_length: float = 0.0  # p
    x_offset: float = 0.0  # c
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

    precision: float = 1e-2

    def __post_init__(self):
        self.center = (self.x_symmetric_axis, self.y_offset)

        self.data_points = int(1000 / self.precision)
        x = np.linspace(0, 1, self.data_points)
        x_left = x[: int(self.data_points / 2)]

        left_intersection = np.argmin(np.abs(x - self.x_offset))
        self.x_start = x[left_intersection]
        self.x_end = 2 * self.x_symmetric_axis - self.x_start
        if self.x_start == 0.0 or self.x_end > 1.0:
            self.ratio = float("inf")
        else:
            right_intersection = np.where(np.isclose(x, self.x_end))[0][0]
            self.intersections = [left_intersection, right_intersection]
            self.width = self.x_end - self.x_start
            self.sin_params[1] = self.sin_params[1] / self.width  # omega

            epsilon, omega, phi = self.sin_params
            sin_wave = epsilon * np.sin(omega * (x - self.x_start) + phi)
            y_left = (np.abs(x_left - self.x_offset) / (4 * self.focal_length)) ** (1 / self.power) + self.y_offset
            y_right = np.flip(y_left)
            y1 = np.concatenate([y_left, y_right]) + sin_wave
            y2 = 2 * self.y_offset - y1
            self.height = max(y1[left_intersection:right_intersection]) - min(y2[left_intersection:right_intersection])
            self.ratio = self.width / self.height if self.height != 0 else float("inf")

    def to_dict(self) -> dict[str, Any]:
        return {"type": "fusiform_2"} | asdict(self)

    def get_bbox(self) -> list[tuple[float, float]]:
        raise NotImplementedError

    def is_closed(self) -> bool:
        return self.ratio < 1e3

    def get_point(self, theta: float) -> tuple[float, float]:
        theta = theta % (2 * np.pi)
        x_range = np.linspace(0, 1, self.data_points)
        x_left = x_range[: int(self.data_points / 2)]

        epsilon, omega, phi = self.sin_params
        sin_wave = epsilon * np.sin(omega * (x_range - self.x_start) + phi)
        y_left = (np.abs(x_left - self.x_offset) / (4 * self.focal_length)) ** (1 / self.power) + self.y_offset
        y_right = np.flip(y_left)
        y_range = np.concatenate([y_left, y_right]) + sin_wave
        y_upper = y_range[self.intersections[0] : self.intersections[1]]

        y_max, y_min = max(y_upper), min(2 * self.y_offset - y_upper)

        # Calculate points on the ray(with polar angle = theta)
        if np.abs(theta - 0.5 * np.pi) < self.precision or np.abs(theta - 1.5 * np.pi) < self.precision:
            y_line = None
        else:
            slope = np.tan(theta)
            y_line = slope * (x_range - self.center[0]) + self.center[1]
            y_line = y_line[self.intersections[0] : self.intersections[1]]

        # Calculate points on the fusiform(after offset)
        if 0 <= theta < np.pi:  # upper curve
            y_fusiform = y_upper
            vertex_indice = np.where(y_fusiform == y_max)[0]
        else:  # lower curve
            y_fusiform = 2 * self.y_offset - y_upper
            vertex_indice = np.where(y_fusiform == y_min)[0]

        if y_line is not None:
            intersection_indices = np.where(np.abs(y_fusiform - y_line) < self.precision)[0]
        else:
            intersection_indices = vertex_indice

        idx = int(intersection_indices.mean() + self.intersections[0])
        # if 0.5 * np.pi <= theta < 1.5 * np.pi:
        #     idx = int(intersection_indices[intersection_indices < self.data_points // 1.8][0])
        # else:
        #     idx = int(intersection_indices[intersection_indices > self.data_points // 2.2][-1])

        x = x_range[idx]
        y = y_range[idx]

        return (x, y)


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
            polygon.special_info += "triangle. "

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
        center = (0.5 + normal(0, 0.01), 0.5 + normal(0, 0.01))
        major_axis = normal(0.05, 0.01)
        minor_axis = uniform(0.8 * major_axis, major_axis)
        rotation = uniform(0, np.pi)
        special_info = "initial chamber. "
        return Ellipse(center, major_axis, minor_axis, rotation, special_info)
