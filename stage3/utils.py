import cv2
import numpy as np

from common.args import feat_recog_args


def bresenham(
    point1: list[int] | tuple[int, int], point2: list[int] | tuple[int, int]
) -> list[tuple[int, int]]:
    """Bresenham's Line Algorithm to generate points between (x1, y1) and (x2, y2)"""
    x1, y1 = point1
    x2, y2 = point2

    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy

    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy

    return points


def get_bbox(image: np.ndarray, threshold: float = 0.05) -> list:
    assert image.ndim == 2, "grayscale image required."
    height, width = image.shape

    # Function to check if a row/column meets the threshold
    def meets_threshold(line, total_pixels):
        non_white_count = np.sum(line < 255)
        return non_white_count > total_pixels * threshold

    # Find the relaxed top boundary
    for min_row in range(height):
        if meets_threshold(image[min_row, :], width):
            break

    # Find the relaxed bottom boundary
    for max_row in range(height - 1, -1, -1):
        if meets_threshold(image[max_row, :], width):
            break

    # Find the relaxed left boundary
    for min_col in range(width):
        if meets_threshold(image[:, min_col], height):
            break

    # Find the relaxed right boundary
    for max_col in range(width - 1, -1, -1):
        if meets_threshold(image[:, max_col], height):
            break

    return [min_row, max_row, min_col, max_col]


def split_into_segments(lst, n) -> list:
    # Split a list into n segments
    k, m = divmod(len(lst), n)
    segments = [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]
    return [seg for seg in segments if seg]


def fit_line(points: list) -> tuple:
    """
    Fit a line to a set of points using numpy's least squares method and return the slope and intercept.

    Parameters:
    points (list): A list of tuples representing the points [(x1, y1), (x2, y2), ...]

    Returns:
    tuple: A tuple containing the slope and intercept of the fitted line (slope, intercept)
    """
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Using numpy's least squares method to fit a line
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    return slope, intercept


def resize_img(img, short_edge_len=448):
    # Get original dimensions
    orig_height, orig_width = img.shape[:2]

    # Calculate new dimensions while maintaining aspect ratio
    if orig_height < orig_width:
        # Portrait orientation
        new_height = short_edge_len
        new_width = int(orig_width * (new_height / orig_height))
    else:
        # Landscape orientation
        new_width = short_edge_len
        new_height = int(orig_height * (new_width / orig_width))

    # Resize images
    img = cv2.resize(img, (new_width, new_height))

    return img


def calculate_angle(vertex: tuple, point1: tuple, point2: tuple) -> float | None:
    """
    Calculate the angle between three points with vertex as the center point.

    Parameters:
    vertex (tuple): The vertex point (x, y)
    point1 (tuple): First point (x, y)
    point2 (tuple): Second point (x, y)

    Returns:
    float: The angle in degrees between 0 and 180
    """
    # Check if any two points are the same
    if vertex == point1 or vertex == point2 or point1 == point2:
        return None
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

    # Ensure angle is between 0 and 180
    return min(angle_deg, 180 - angle_deg)


def circle_weight_array(window_size: int):
    # 2-D array with reward values, positive inside the inscribed circle, negative outside
    pos_weight_array = np.zeros((window_size, window_size))
    neg_weight_array = np.zeros((window_size, window_size))

    # Calculate the center and radius of the inscribed circle
    center = (window_size - 1) / 2
    outter_radius = (window_size // 2) / window_size
    inner_radius = (window_size // 2) * feat_recog_args.inner_radius_ratio / window_size

    total_positive_weight = 0
    total_negative_weight = 0

    # Fill the reward array based on distance from the center
    for i in range(window_size):
        for j in range(window_size):
            # Calculate relative distance (normalized by window_size)
            distance = np.sqrt((i - center) ** 2 + (j - center) ** 2) / window_size

            # Set reward value - positive inside circle, negative outside
            if distance <= inner_radius:
                # inner circle: exponential decreasing positive reward (1.0 at center, 0.5 at inner edge)
                pos_weight_array[i, j] = 0.5 * (1 + np.exp(-10 * distance))
                total_positive_weight += pos_weight_array[i, j]
            elif distance <= outter_radius:
                # outter circle: exponential increaseing negative reward (-0.5 at inner edge, 0 at outer edge)
                neg_weight_array[i, j] = -0.5 * (np.exp(-10 * (distance - inner_radius)))
                total_negative_weight += neg_weight_array[i, j]

    return pos_weight_array, neg_weight_array, total_positive_weight, abs(total_negative_weight)
