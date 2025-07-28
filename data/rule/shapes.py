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
    def from_dict(data: dict[str, Any]) -> "GSRule":
        shape_type = data.get("type", "")
        if shape_type == "polygon":
            points = [tuple(point) for point in data["points"]]
            return Polygon(
                points=points,
                special_info=data.get("special_info", ""),
                fill_mode=data.get("fill_mode", "no"),
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
        elif shape_type == "sector":
            return Sector(
                center=data["center"],
                radius=data["radius"],
                start_angle=data["start_angle"],
                end_angle=data["end_angle"],
                special_info=data.get("special_info", ""),
                fill_mode=data.get("fill_mode", "border"),
            )
        elif shape_type == "star":
            return Star(
                center=data["center"],
                outer_radius=data["outer_radius"],
                inner_radius=data["inner_radius"],
                num_points=data["num_points"],
                rotation=data.get("rotation", 0),
                special_info=data.get("special_info", ""),
                fill_mode=data.get("fill_mode", "border"),
            )
        raise ValueError(f"Unknown shape type: {shape_type}")


@dataclass
class Polygon(GSRule):
    points: list[tuple[float, float]]
    special_info: str = ""
    fill_mode: Literal["no", "white", "black", "border"] = "border"

    def __post_init__(self):
        assert all(isinstance(point, tuple) for point in self.points), "Points must be tuples"

    def to_dict(self) -> dict[str, Any]:
        return {"type": "polygon"} | asdict(self)

    def get_bbox(self) -> list[tuple[float, float]]:
        return self.bbox_from_points(self.points)

    def _get_area_v(self) -> float:
        n = len(self.points)
        area = 0.0
        for i in range(n):
            x_i, y_i = self.points[i]
            x_next, y_next = self.points[(i + 1) % n]
            area += x_i * y_next - x_next * y_i
        return area / 2.0

    def get_area(self) -> float:
        return abs(self._get_area_v())

    def get_centroid(self) -> tuple[float, float]:
        A = self._get_area_v()
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
        assert 0 < C_x < 1 and 0 < C_y < 1
        return C_x, C_y

    def normalize_points(self):
        while True:
            min_x = min(x for x, y in self.points)
            min_y = min(y for x, y in self.points)
            max_x = max(x for x, y in self.points)
            max_y = max(y for x, y in self.points)
            max_range = max(max_x - min_x, max_y - min_y) * 1.2
            if min_x > 0 and min_y > 0 and max_x < 1 and max_y < 1:
                break
            self.points = [((x - min_x) / max_range, (y - min_y) / max_range) for x, y in self.points]
            offset = uniform(-0.3, 0.3)
            self.points = [(x + offset, y + offset) for x, y in self.points]

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

    def check_angle(self, thres_low=0.15 * np.pi, thres_high=0.85 * np.pi) -> bool:
        pass_check = True

        # Check if each angle is greater than thres
        def angle_between(v1, v2):
            return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))

        angles = []
        for i in range(len(self.points)):
            v1 = np.array(self.points[i]) - np.array(self.points[i - 1])
            v2 = np.array(np.array(self.points[i]) - self.points[(i + 1) % len(self.points)])
            angles.append(angle_between(v1, v2))

        min_angle = min(angles)
        max_angle = max(angles)
        if min_angle < thres_low or max_angle > thres_high:
            pass_check = False

        # Check whether all angles are around np.pi/2 if not a rectangle
        if (
            len(self.points) == 4
            and "rectangle" not in self.special_info
            and "square" not in self.special_info
        ):
            all_around = True
            for angle in angles:
                if abs(angle - np.pi / 2) > rule_args.general_quadrilateral_angle_thres:
                    all_around = False
                    break
            if all_around:
                pass_check = False

        # Check whether all angles are around np.pi/3 if not a equilateral triangle
        if len(self.points) == 3 and "equilateral triangle" not in self.special_info:
            all_around = True
            for angle in angles:
                if abs(angle - np.pi / 3) > rule_args.general_triangle_angle_thres:
                    all_around = False
            if all_around:
                pass_check = False

        return pass_check

    def to_simple_polygon(self):
        n = len(self.points)
        center = (sum([p[0] for p in self.points]) / n, sum([p[1] for p in self.points]) / n)
        self.points.sort(key=lambda p: (polar_angle(center, p), -distance_2points(center, p)))

    def check_points_distance(self) -> bool:
        n = len(self.points)
        pass_check = True
        for i in range(n - 1):
            if distance_2points(self.points[i], self.points[i + 1]) < rule_args.polygon_points_min_distance:
                pass_check = False
        return pass_check

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

    def to_square(self, side_len: float, rotation: float):
        self.special_info = "square"
        self.side_len = side_len

        x0, y0 = self.points[0]
        r = rotation

        x1 = x0 + side_len * np.cos(r)
        y1 = y0 + side_len * np.sin(r)
        x2 = x1 - side_len * np.sin(r)
        y2 = y1 + side_len * np.cos(r)
        x3 = x0 - side_len * np.sin(r)
        y3 = y0 + side_len * np.cos(r)
        self.points = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]

    def to_rectangle(self, width: float, height: float, rotation: float):
        self.special_info = "rectangle"
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

    def to_regular_pentagon(self, side_len: float, rotation: float):
        """Generate a regular pentagon with given side length and rotation"""
        self.special_info = "regular pentagon"
        self.side_len = side_len

        # Calculate the circumradius of the pentagon
        # For a regular pentagon, circumradius = side_length / (2 * sin(π/5))
        circumradius = side_len / (2 * np.sin(np.pi / 5))

        x0, y0 = self.points[0]

        # Generate 5 vertices of the regular pentagon
        points = []
        for i in range(5):
            angle = rotation + i * 2 * np.pi / 5
            x = x0 + circumradius * np.cos(angle)
            y = y0 + circumradius * np.sin(angle)
            points.append((x, y))

        self.points = points

    def to_regular_hexagon(self, side_len: float, rotation: float):
        """Generate a regular hexagon with given side length and rotation"""
        self.special_info = "regular hexagon"
        self.side_len = side_len

        # For a regular hexagon, circumradius = side_length
        circumradius = side_len

        x0, y0 = self.points[0]

        # Generate 6 vertices of the regular hexagon
        points = []
        for i in range(6):
            angle = rotation + i * 2 * np.pi / 6
            x = x0 + circumradius * np.cos(angle)
            y = y0 + circumradius * np.sin(angle)
            points.append((x, y))

        self.points = points


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
    fill_mode: Literal["no", "white", "black", "border"] = "border"

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
        curve_points = self.curve_points.tolist()
        return self.bbox_from_points(curve_points)

    def get_area(self) -> float:
        return np.pi * self.major_axis * self.minor_axis / 4

    def get_centroid(self) -> tuple[float, float]:
        return self.center

    def get_point(self, theta=None) -> tuple[float, float]:
        if theta is None:
            theta = np.random.uniform(0, 2 * np.pi)
        theta = theta % (2 * np.pi)

        n = len(self.curve_points)
        if np.isclose(theta, 0.5 * np.pi):
            point = self.curve_points[n // 4]
        elif np.isclose(theta, 1.5 * np.pi):
            point = self.curve_points[int(3 * n // 4)]
        else:
            slope = np.tan(theta + self.rotation)
            intercept = self.center[1] - slope * self.center[0]
            line = (slope, intercept)

            quarter_size = len(self.curve_points) // 4
            quarter_idx = theta // (0.5 * np.pi)  # i_th curve
            start_idx = int(quarter_size * quarter_idx * 0.8)
            end_idx = int(quarter_size * (quarter_idx + 1) * 1.2)

            distances = [
                distance_point_to_line(point, line) for point in self.curve_points[start_idx:end_idx]
            ]
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


@dataclass
class Spiral(GSRule):
    # Archimedean spiral  r = a + b(\theta) * \theta
    # b(\theta) = b + \epsilon sin(\omega \theta + \phi)
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
        def integrand(theta):
            r = self.radius(theta)
            return 0.5 * r**2

        # Only integrate over the last complete 2 * \pi
        theta_start = max(0, self.max_theta - 2 * np.pi)
        theta_end = self.max_theta
        area, _ = quad(integrand, theta_start, theta_end)

        return area

    def get_centroid(self) -> tuple[float, float]:
        return self.center

    def radius(self, theta: float) -> float:
        epsilon, omega, phi = self.sin_params
        return self.initial_radius + (self.growth_rate + epsilon * np.sin(omega * theta + phi)) * theta


@dataclass
class Sector(GSRule):
    """Sector (fan shape) - a region bounded by two radii and an arc"""

    center: tuple[float, float]
    radius: float
    start_angle: float  # starting angle in radians
    end_angle: float  # ending angle in radians
    special_info: str = ""
    fill_mode: Literal["no", "white", "black", "border"] = "border"

    def __post_init__(self):
        # Ensure angles are in [0, 2π) range
        self.start_angle = self.start_angle % (2 * np.pi)
        self.end_angle = self.end_angle % (2 * np.pi)

        # Calculate the angular span
        if self.end_angle <= self.start_angle:
            self.angular_span = self.end_angle + 2 * np.pi - self.start_angle
        else:
            self.angular_span = self.end_angle - self.start_angle

        # Generate points for the sector boundary
        self._generate_points()

    def _generate_points(self):
        """Generate points that define the sector boundary"""
        # Vertices
        self.arc_start_point = (
            self.center[0] + self.radius * np.cos(self.start_angle),
            self.center[1] + self.radius * np.sin(self.start_angle),
        )
        self.arc_end_point = (
            self.center[0] + self.radius * np.cos(self.end_angle),
            self.center[1] + self.radius * np.sin(self.end_angle),
        )

        # Number of points for the arc
        arc_points = max(50, int(self.angular_span / (2 * np.pi) * 200))

        # Generate arc points
        if self.end_angle <= self.start_angle:
            # Arc crosses 0 angle
            theta_values = np.linspace(self.start_angle, self.start_angle + self.angular_span, arc_points)
        else:
            theta_values = np.linspace(self.start_angle, self.end_angle, arc_points)

        # Arc points
        arc_x = self.center[0] + self.radius * np.cos(theta_values)
        arc_y = self.center[1] + self.radius * np.sin(theta_values)
        arc_points_array = np.column_stack((arc_x, arc_y))

        # Complete sector boundary: center -> start of arc -> arc -> end of arc -> center
        self.points = np.vstack(
            [[self.center], arc_points_array, [self.center]]  # center point  # arc points  # back to center
        )

    def to_dict(self) -> dict[str, Any]:
        return {"type": "sector"} | asdict(self)

    def get_bbox(self) -> list[tuple[float, float]]:
        # Get bounding box from all boundary points
        return self.bbox_from_points(self.points.tolist())

    def get_area(self) -> float:
        """Calculate sector area: (1/2) * r² * θ"""
        return 0.5 * self.radius**2 * self.angular_span

    def get_centroid(self) -> tuple[float, float]:
        """Calculate sector centroid"""
        # For a sector, centroid is at (2/3) * r from center along the angle bisector
        bisector_angle = self.start_angle + self.angular_span / 2
        centroid_distance = (2 / 3) * self.radius * np.sin(self.angular_span / 2) / (self.angular_span / 2)

        centroid_x = self.center[0] + centroid_distance * np.cos(bisector_angle)
        centroid_y = self.center[1] + centroid_distance * np.sin(bisector_angle)

        return (centroid_x, centroid_y)

    def get_arc_point(self, t: Optional[float] = None) -> tuple[float, float]:
        """Get a point on the arc. t should be in [0, 1] where 0 is start_angle and 1 is end_angle"""
        if t is None:
            t = np.random.uniform(0, 1)

        t = np.clip(t, 0, 1)
        angle = self.start_angle + t * self.angular_span  # type: ignore

        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)

        return (x, y)

    def get_radius_endpoint(self, at_start: bool = True) -> tuple[float, float]:
        """Get the endpoint of a radius (either at start_angle or end_angle)"""
        angle = self.start_angle if at_start else self.end_angle
        x = self.center[0] + self.radius * np.cos(angle)
        y = self.center[1] + self.radius * np.sin(angle)
        return (x, y)

    def contains_point(self, point: tuple[float, float]) -> bool:
        """Check if a point is inside the sector"""
        px, py = point
        cx, cy = self.center

        # Check if point is within radius
        distance = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        if distance > self.radius:
            return False

        # Check if point is within angular range
        point_angle = np.arctan2(py - cy, px - cx) % (2 * np.pi)

        if self.end_angle <= self.start_angle:
            # Sector crosses 0 angle
            return point_angle >= self.start_angle or point_angle <= self.end_angle
        else:
            return self.start_angle <= point_angle <= self.end_angle

    def get_angular_span_degrees(self) -> float:
        """Get angular span in degrees"""
        return np.degrees(self.angular_span)


@dataclass
class Star(GSRule):
    """Star shape - a polygon with alternating outer and inner vertices"""

    center: tuple[float, float]
    outer_radius: float
    inner_radius: float
    num_points: int  # number of star points (e.g., 5 for a 5-pointed star)
    rotation: float = 0  # rotation angle in radians
    special_info: str = ""
    fill_mode: Literal["no", "white", "black", "border"] = "border"

    def __post_init__(self):
        assert self.num_points >= 3, "Star must have at least 3 points"
        assert self.outer_radius > self.inner_radius > 0, "Outer radius must be greater than inner radius"

        # Generate star vertices
        self._generate_star_points()

    def _generate_star_points(self):
        """Generate the vertices of the star"""
        self.points = []

        # Total number of vertices is 2 * num_points (alternating outer and inner)
        total_vertices = 2 * self.num_points
        angle_step = 2 * np.pi / total_vertices

        for i in range(total_vertices):
            angle = self.rotation + i * angle_step

            # Alternate between outer and inner radius
            if i % 2 == 0:
                # Outer vertex
                radius = self.outer_radius
            else:
                # Inner vertex
                radius = self.inner_radius

            x = self.center[0] + radius * np.cos(angle)
            y = self.center[1] + radius * np.sin(angle)
            self.points.append((x, y))

    def to_dict(self) -> dict[str, Any]:
        return {"type": "star"} | asdict(self)

    def get_bbox(self) -> list[tuple[float, float]]:
        return self.bbox_from_points(self.points)

    def get_area(self) -> float:
        """Calculate star area using the shoelace formula"""
        n = len(self.points)
        area = 0.0
        for i in range(n):
            x_i, y_i = self.points[i]
            x_next, y_next = self.points[(i + 1) % n]
            area += x_i * y_next - x_next * y_i
        return abs(area) / 2.0

    def get_centroid(self) -> tuple[float, float]:
        """Calculate star centroid"""
        # For a regular star, the centroid is at the center
        return self.center

    def get_outer_point(self, point_index: int) -> tuple[float, float]:
        """Get the coordinates of a specific outer point (0-indexed)"""
        if point_index < 0 or point_index >= self.num_points:
            raise ValueError(f"Point index must be between 0 and {self.num_points - 1}")

        # Outer points are at even indices
        return self.points[point_index * 2]

    def get_inner_point(self, point_index: int) -> tuple[float, float]:
        """Get the coordinates of a specific inner point (0-indexed)"""
        if point_index < 0 or point_index >= self.num_points:
            raise ValueError(f"Point index must be between 0 and {self.num_points - 1}")

        # Inner points are at odd indices
        return self.points[point_index * 2 + 1]

    def get_all_outer_points(self) -> list[tuple[float, float]]:
        """Get all outer points of the star"""
        return [self.points[i] for i in range(0, len(self.points), 2)]

    def get_all_inner_points(self) -> list[tuple[float, float]]:
        """Get all inner points of the star"""
        return [self.points[i] for i in range(1, len(self.points), 2)]

    def scale(self, outer_scale: float, inner_scale: Optional[float] = None):
        """Scale the star by changing the radii"""
        if inner_scale is None:
            inner_scale = outer_scale

        self.outer_radius *= outer_scale
        self.inner_radius *= inner_scale
        self._generate_star_points()

    def rotate(self, angle: float):
        """Rotate the star by the given angle (in radians)"""
        self.rotation += angle
        self._generate_star_points()

    def contains_point(self, point: tuple[float, float]) -> bool:
        """Check if a point is inside the star using ray casting algorithm"""
        x, y = point
        n = len(self.points)
        inside = False

        p1x, p1y = self.points[0]
        for i in range(1, n + 1):
            p2x, p2y = self.points[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def to_regular_star(self, num_points: int, outer_radius: float, inner_radius: float):
        """Convert to a regular star with specified parameters"""
        self.num_points = num_points
        self.outer_radius = outer_radius
        self.inner_radius = inner_radius
        self.special_info = f"regular {num_points}-pointed star"
        self._generate_star_points()


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
    fill_mode: Literal["no", "white", "black", "border"] = "border"

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
            return tuple(self.curve_points[n // 4])
        elif np.isclose(theta, 1.5 * np.pi):
            return tuple(self.curve_points[int(3 * n // 4)])
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
    fill_mode: Literal["no", "white", "black", "border"] = "border"

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
    fill_mode: Literal["no", "white", "black", "border"] = "border"

    def __post_init__(self):
        self.num_points = 100
        # Ensure there are exactly 4 control points for a cubic curve
        assert len(self.control_points) == 4, "A cubic Bézier curve requires exactly 4 control points."

        # Unpack control points
        p0, p1, p2, p3 = self.control_points

        # Precompute curve points
        self.curve_points = self._compute_curve_points(p0, p1, p2, p3)

    def to_dict(self):
        return {"type": "curve"} | asdict(self)

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
    fill_mode: Literal["no", "white", "black", "border"] = "border"

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
        self.opt_shapes = ["polygon", "line", "ellipse", "spiral", "sector", "star"]

        self.shape_prob = []

        for shape_type in self.opt_shapes:
            shape_prob = eval(f"rule_args.{shape_type}_shape_level")
            self.shape_prob.append(shape_prob)
        self.shape_prob = [x / sum(self.shape_prob) for x in self.shape_prob]

    def __call__(self) -> Polygon | Line | Ellipse | Spiral | Sector | Star:
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
        elif shape_type == "sector":
            shape = self.generate_sector()
        elif shape_type == "star":
            shape = self.generate_star()
        else:
            raise ValueError("invalid type of shape")
        return shape

    def generate_polygon(self, points: Optional[list[tuple[float, float]]] = None, max_points=6) -> Polygon:
        if points is None:
            num_points = randint(3, max_points + 1)  # at least 3 points
            points = [(uniform(0.2, 0.8), uniform(0.2, 0.8)) for _ in range(num_points)]

        polygon = Polygon(points)
        polygon.to_simple_polygon()
        while (
            not polygon.is_convex()
            or polygon.get_area() < 0.01
            or not polygon.check_angle()
            or not polygon.check_points_distance()
        ):
            points = [(uniform(0.2, 0.8), uniform(0.2, 0.8)) for _ in range(num_points)]
            polygon = Polygon(points)
            polygon.to_simple_polygon()

        special_polygon = np.random.choice(
            ["no", "square", "rectangle", "equilateral triangle", "regular pentagon", "regular hexagon"],
            p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
        )
        if special_polygon == "square":
            side_len = uniform(0.1, 0.6)
            rotation = uniform(0, 2 * np.pi)
            polygon.to_square(side_len, rotation)
        elif special_polygon == "rectangle":
            width, height = (uniform(0.1, 0.6), uniform(0.1, 0.6))
            if width >= height:
                height = width / uniform(
                    rule_args.rectangle_ratio_thres[0], rule_args.rectangle_ratio_thres[1]
                )
            else:
                width = height / uniform(
                    rule_args.rectangle_ratio_thres[0], rule_args.rectangle_ratio_thres[1]
                )
            rotation = uniform(0, 2 * np.pi)
            polygon.to_rectangle(width, height, rotation)
        elif special_polygon == "equilateral triangle":
            side_len = uniform(0.1, 0.6)
            rotation = uniform(0, 2 * np.pi)
            polygon.to_equilateral_triangle(side_len, rotation)
        elif special_polygon == "regular pentagon":
            side_len = uniform(0.1, 0.5)
            rotation = uniform(0, 2 * np.pi)
            polygon.to_regular_pentagon(side_len, rotation)
        elif special_polygon == "regular hexagon":
            side_len = uniform(0.1, 0.5)
            rotation = uniform(0, 2 * np.pi)
            polygon.to_regular_hexagon(side_len, rotation)
        elif len(points) == 3:
            polygon.special_info = "triangle"

        polygon.normalize_points()
        return polygon

    def generate_line(
        self, points: Optional[list[tuple[float, float]]] = None, min_length=0.2, max_length=0.8
    ) -> Line:
        def random_point_on_edge(edge) -> tuple:
            if edge == "left":
                return (0, uniform(0.1, 0.9))
            elif edge == "right":
                return (1, uniform(0.1, 0.9))
            elif edge == "top":
                return (uniform(0.1, 0.9), 1)
            else:  # bottom
                return (uniform(0.1, 0.9), 0)

        if points is not None:
            line = Line(type="segment", points=points)
            return line

        line_type = np.random.choice(["line", "segment", "ray"])
        while True:
            if line_type == "segment":
                point1 = (uniform(0, 1), uniform(0, 1))
                point2 = (uniform(0, 1), uniform(0, 1))
                line = Line(type=line_type, points=[point1, point2])
            elif line_type == "ray":
                point1 = (uniform(0, 1), uniform(0, 1))
                point2 = random_point_on_edge(np.random.choice(["left", "right", "top", "bottom"]))
                line = Line(type=line_type, points=[point1, point2])
            elif line_type == "line":
                edge1, edge2 = np.random.choice(["left", "right", "top", "bottom"], size=2, replace=False)
                point1 = random_point_on_edge(edge1)
                point2 = random_point_on_edge(edge2)
                line = Line(type=line_type, points=[point1, point2])
            if min_length < line.get_length() < max_length:
                break
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
            minor_axis = major_axis / uniform(
                rule_args.ellipse_ratio_thres[0], rule_args.ellipse_ratio_thres[1]
            )
            rotation = uniform(0, np.pi)
            ellipse = Ellipse(center, major_axis, minor_axis, rotation)
            while ellipse.get_area() < 0.01:
                major_axis = normal(0.5, 0.1)
                minor_axis = major_axis / uniform(
                    rule_args.ellipse_ratio_thres[0], rule_args.ellipse_ratio_thres[1]
                )
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

    def generate_sector(
        self,
        center: Optional[tuple[float, float]] = None,
        radius: float = 0.0,
        start_angle: float = 0.0,
        end_angle: float = 0.0,
    ) -> Sector:
        """Generate a sector (fan shape) with random or specified parameters"""
        if center is not None and radius > 0:
            # Use provided parameters
            sector = Sector(center, radius, start_angle, end_angle)
            return sector
        else:
            # Generate random parameters
            center = (uniform(0.2, 0.8), uniform(0.2, 0.8))
            radius = uniform(0.1, 0.6)  # Reasonable radius range

            # Generate angular span - can be anywhere from 30 degrees to 300 degrees
            min_span = np.pi / 12  # 15 degrees
            max_span = 5 * np.pi / 6  # 150 degrees
            angular_span = uniform(min_span, max_span)

            # Random starting angle
            start_angle = uniform(0, 2 * np.pi)
            end_angle = start_angle + angular_span

            sector = Sector(center, radius, start_angle, end_angle)

            # Ensure the sector has reasonable area
            while sector.get_area() < 0.01:
                radius = uniform(0.1, 0.4)
                angular_span = uniform(min_span, max_span)
                start_angle = uniform(0, 2 * np.pi)
                end_angle = start_angle + angular_span
                sector = Sector(center, radius, start_angle, end_angle)

            return sector

    def generate_star(
        self,
        center: Optional[tuple[float, float]] = None,
        outer_radius: float = 0.0,
        inner_radius: float = 0.0,
        num_points: int = 0,
        rotation: float = 0.0,
    ) -> Star:
        """Generate a star with random or specified parameters"""
        if center is not None and outer_radius > 0 and inner_radius > 0 and num_points > 0:
            # Use provided parameters
            star = Star(center, outer_radius, inner_radius, num_points, rotation)
            return star
        else:
            # Generate random parameters
            center = (uniform(0.2, 0.8), uniform(0.2, 0.8))
            outer_radius = uniform(0.15, 0.4)  # Reasonable outer radius range

            # Inner radius should be smaller than outer radius
            inner_radius = outer_radius * uniform(0.4, 0.7)  # 40% to 70% of outer radius

            # Number of star points (5 or 6 points)
            num_points = randint(5, 7)

            # Random rotation
            rotation = uniform(0, 2 * np.pi)

            star = Star(center, outer_radius, inner_radius, num_points, rotation)

            # Ensure the star has reasonable area
            while star.get_area() < 0.01:
                outer_radius = uniform(0.15, 0.4)
                inner_radius = outer_radius * uniform(0.3, 0.7)
                star = Star(center, outer_radius, inner_radius, num_points, rotation)

            return star

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
        # size = np.random.choice(["small", "large"], p=[0.6, 0.4])
        # if size == "small":
        #     major_axis = normal(0.009, 0.001)
        # elif size == "large":
        #     major_axis = normal(0.028, 0.003)

        # major_axis = max(0.03, normal(0.03, 6e-3))
        major_axis = uniform(0.03, 0.05)
        minor_axis = uniform(0.8 * major_axis, major_axis)
        rotation = uniform(0, np.pi)
        special_info = "initial chamber"
        return Ellipse(center, major_axis, minor_axis, rotation, special_info, fill_mode="border")

    def generate_axial_filling(self, num_volutions: int, rule_args) -> list[dict]:
        axial_filling = []

        # Generate shared parameters to ensure symmetry
        start_volution = randint(0, max(1, num_volutions // 4))

        if rule_args.overlap_axial_and_poles_folds:
            end_volution = num_volutions - 1
        else:
            end_volution = randint(num_volutions // 2, num_volutions - 1)

        # Generate a single angle parameter to ensure symmetry
        angle_offset = normal(0.1, 0.02) * np.pi

        for i in range(2):
            # Use the same angle_offset for both sides, just offset by π
            start_angle_main = -angle_offset + i * np.pi
            end_angle_main = angle_offset + i * np.pi

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
                "start_angle": start_angle_main - max_extend_angle1 * normal(0.5, 0.1),
                "end_angle": start_angle_main,
                "start_volution": 0,
                "end_volution": randint(num_volutions - 2, num_volutions),
            }
            max_extend_angle2 = ((0.5 - i) * np.pi - end_angle_main) % (2 * np.pi)
            axial_filling_extend2 = {
                "type": "extension",
                "start_angle": end_angle_main,
                "end_angle": end_angle_main + max_extend_angle2 * normal(0.5, 0.1),
                "start_volution": 0,
                "end_volution": randint(num_volutions - 2, num_volutions),
            }
            axial_filling.append(axial_filling_extend1)
            axial_filling.append(axial_filling_extend2)

        return axial_filling

    def generate_poles_folds(self, num_volutions: int, axial_filling: list, rule_args) -> list[dict]:
        poles_folds = []

        if not rule_args.overlap_axial_and_poles_folds and axial_filling:
            start_volution = axial_filling[0]["end_volution"]
        else:
            start_volution = randint(num_volutions - 3, num_volutions - 1)
        end_volution = num_volutions - 1
        start_angle = -normal(0.3, 0.03) * np.pi
        end_angle = normal(0.3, 0.03) * np.pi
        for i in range(2):
            poles_folds.append(
                {
                    "start_angle": start_angle + i * np.pi,
                    "end_angle": end_angle + i * np.pi,
                    "start_volution": start_volution,
                    "end_volution": end_volution,
                }
            )
        return poles_folds
