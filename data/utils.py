import numpy as np


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


def another_2points_on_line(line: tuple[float, float], point: tuple[float, float]) -> list[tuple[float, float]]:
    """
    Given a line and a point on the line, return another 2 points on the line.
    """
    x, y = point
    slope, intercept = line
    assert np.isclose(slope * x + intercept, y), "The point is not on the line."

    if slope != float("inf"):
        x1 = x + np.random.uniform(0.05, 0.5)
        y1 = slope * x1 + intercept
        x2 = x - np.random.uniform(0.05, 0.5)
        y2 = slope * x2 + intercept
    else:
        x1 = x
        x2 = x
        y1 = y + np.random.uniform(0, 1)
        y2 = y - np.random.uniform(0, 1)

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
