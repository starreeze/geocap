from typing import TypeVar, cast

import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative


def distance_2points(point1: tuple[float, float], point2: tuple[float, float]) -> float:
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def polar_angle(point1: tuple[float, float], point2: tuple[float, float]) -> float:
    """
    Calculate the angle between the line (ray from point1 to point2) and horizontal axis.
    """
    x1, y1 = point1
    x2, y2 = point2
    delta_y = y2 - y1
    delta_x = x2 - x1

    angle_radians = np.arctan2(delta_y, delta_x)
    if angle_radians < 0:
        angle_radians += 2 * np.pi

    return angle_radians


def calculate_angle(vertex: tuple, point1: tuple, point2: tuple) -> float:
    """
    Calculate the angle between three points with vertex as the center point.

    Parameters:
    vertex (tuple): The vertex point (x, y)
    point1 (tuple): First point (x, y)
    point2 (tuple): Second point (x, y)

    Returns:
    float: The angle in degrees between 0 and 180
    """
    # Convert points to numpy arrays
    v = np.array(vertex)
    p1 = np.array(point1)
    p2 = np.array(point2)

    # Calculate vectors from vertex to points
    vec1 = p1 - v
    vec2 = p2 - v

    # Calculate dot product and magnitudes
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

    # Calculate angle in radians and convert to degrees
    angle_rad = np.arccos(np.clip(dot_product / norm_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def line_given2points(points: list[tuple[float, float]]) -> tuple[float, float]:
    """
    Calculate slope and intercept of the line formed by two points.
    Return: (slope, intercept)
    """
    assert len(points) == 2
    x1, y1 = points[0]
    x2, y2 = points[1]

    if abs(x1 - x2) < 1e-9:  # vertical line
        return float("inf"), x1  # intercept on x-axis

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    return (slope, intercept)


def another_2points_on_line(
    line: tuple[float, float], point: tuple[float, float]
) -> list[tuple[float, float]]:
    """
    Given a line and a point on the line, return another 2 points on the line.
    """
    x, y = point
    slope, intercept = line

    if slope != float("inf"):
        assert np.isclose(slope * x + intercept, y), "The point is not on the line."
        if abs(slope) <= 1:
            x1 = x + np.random.uniform(0.05, 0.2)
            y1 = slope * x1 + intercept
            x2 = x - np.random.uniform(0.05, 0.2)
            y2 = slope * x2 + intercept
        else:
            y1 = y + np.random.uniform(0.05, 0.2)
            x1 = (y1 - intercept) / slope
            y2 = y - np.random.uniform(0.05, 0.2)
            x2 = (y2 - intercept) / slope

    else:
        assert x == intercept, "The point is not on the line."
        x1 = x
        x2 = x
        y1 = y + np.random.uniform(0, 0.5)
        y2 = y - np.random.uniform(0, 0.5)

    return [(x1, y1), (x2, y2)]


def distance_point_to_line(point: tuple[float, float], line: tuple[float, float]) -> float:
    """
    Calculate the distance from a given point and a line.
    The line is given in form of y = kx + b.
    """
    x0, y0 = point
    k, b = line
    if k != float("inf"):
        numerator = abs(k * x0 - y0 + b)
        denominator = np.sqrt(k**2 + 1)
        distance = numerator / denominator
    else:
        distance = abs(x0 - b)

    return distance


def find_intersection(
    line1: tuple[float, float], line2: tuple[float, float]
) -> tuple[float, float] | tuple[None, None]:
    """
    Find intersection of two lines, each line is given by tuple of (slope, intercept)
    Returns: coordinates of intersection.
    """
    k1, b1 = line1
    k2, b2 = line2

    # parallel
    if abs(k1 - k2) < 1e-9:
        return None, None

    if k1 == float("inf"):
        x = b1
        y = k2 * x + b2
    elif k2 == float("inf"):
        x = b2
        y = k1 * x + b1
    else:
        x = (b2 - b1) / (k1 - k2)
        y = k1 * x + b1

    return (x, y)


def find_symmetric_point(line: tuple[float, float], point: tuple[float, float]) -> tuple[float, float]:
    k, b = line
    x1, y1 = point

    x2 = (x1 + k * (y1 - b)) / (k**2 + 1)
    y2 = k * x2 + b

    x_prime = 2 * x2 - x1
    y_prime = 2 * y2 - y1

    return (x_prime, y_prime)


def overlap_area(bbox1, bbox2) -> float:
    min_x1, max_y1 = bbox1[0]
    max_x1, min_y1 = bbox1[1]
    min_x2, max_y2 = bbox2[0]
    max_x2, min_y2 = bbox2[1]

    # Calculate the intersection coordinates
    overlap_min_x = max(min_x1, min_x2)
    overlap_max_x = min(max_x1, max_x2)
    overlap_min_y = max(min_y1, min_y2)
    overlap_max_y = min(max_y1, max_y2)

    # Calculate the width and height of the overlap
    overlap_width = max(0, overlap_max_x - overlap_min_x)
    overlap_height = max(0, overlap_max_y - overlap_min_y)

    # Calculate the area of the overlap
    overlap_area = overlap_width * overlap_height

    return overlap_area


def no_overlap(shapes, new_shape, exclude_shape=None, thres=0.2) -> bool:
    if new_shape is None:
        return False

    if exclude_shape is None:
        exclude_shape = []

    iou_sum = 0
    for cur_shape in shapes:
        if cur_shape not in exclude_shape:
            cur_bbox = cur_shape.get_bbox()
            new_bbox = new_shape.get_bbox()
            cur_area = (cur_bbox[1][0] - cur_bbox[0][0]) * (cur_bbox[0][1] - cur_bbox[1][1])
            new_area = (new_bbox[1][0] - new_bbox[0][0]) * (new_bbox[0][1] - new_bbox[1][1])
            intersection = overlap_area(cur_bbox, new_bbox)
            union = cur_area + new_area - intersection
            iou_sum += intersection / union

    if iou_sum > thres:
        return False
    return True


def get_tangent_line(
    tangent_point: tuple[float, float], curve_points: list[tuple[float, float]]
) -> tuple[float, float]:
    assert tangent_point in curve_points, "tangent point is not on the curve"

    x, y = zip(*curve_points)
    # Remove points with duplicate x values
    unique_x = []
    unique_y = []
    for x_val, y_val in zip(x, y):
        if x_val not in unique_x:
            unique_x.append(x_val)
            unique_y.append(y_val)
    x, y = unique_x, unique_y

    curve_func = interp1d(x, y, kind="cubic", fill_value="extrapolate")

    slope = derivative(curve_func, tangent_point[0], dx=1e-6)

    y_target = curve_func(tangent_point[0])
    intercept = y_target - slope * tangent_point[0]

    return (slope, intercept)


def find_perpendicular_line(line: tuple[float, float], point: tuple[float, float]):
    slope, intercept = line
    x, y = point

    if np.abs(slope) > 1e-3:
        perpendicular_slope = -1 / slope
        perpendicular_intercept = y - perpendicular_slope * x
    else:
        perpendicular_slope = float("inf")
        perpendicular_intercept = x

    return perpendicular_slope, perpendicular_intercept


T = TypeVar("T")


def round_floats(obj: T, precision=2) -> T:
    if isinstance(obj, float):
        return cast(T, round(obj, precision))
    if isinstance(obj, dict):
        return cast(T, {k: round_floats(v, precision) for k, v in obj.items()})
    if isinstance(obj, np.ndarray):
        return cast(T, np.round(obj, precision))
    if isinstance(obj, (list, tuple)):
        return cast(T, type(obj)([round_floats(x, precision) for x in obj]))  # type: ignore
    return obj
