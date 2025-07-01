import cv2
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

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


def get_circle_points(center: tuple, radius: int, angle_range: list[int]) -> list[tuple]:
    """
    Get points on a circle with a given center and radius.

    Args:
        center (tuple): The center of the circle (x, y)
        radius (int): The radius of the circle
        angle_range (list[int]): The range of angles to generate points for

    Returns:
        list[tuple]: A list of points on the circle
    """
    points = []
    x_center, y_center = center

    # Convert angle range from degrees to radians
    start_angle_rad = np.radians(angle_range[0])
    end_angle_rad = np.radians(angle_range[1])
    assert start_angle_rad < end_angle_rad, "Start angle must be less than end angle"

    # Calculate number of points based on radius (more points for larger radius)
    # This ensures smooth circle approximation
    angles = np.linspace(start_angle_rad, end_angle_rad, 30)

    # Generate points along the arc
    for angle in angles:
        # Calculate point coordinates
        x = int(x_center + radius * np.cos(angle))
        y = int(y_center + radius * np.sin(angle))

        points.append((x, y))

    # Remove points with same x
    points_with_unique_x = []
    for point in points:
        if point[0] not in [p[0] for p in points_with_unique_x]:
            points_with_unique_x.append(point)

    # Sort points by x
    points_with_unique_x.sort(key=lambda x: x[0])

    return points_with_unique_x


def simple_smooth(data, window_size=3):
    if len(data) < window_size:
        return data

    smoothed = np.zeros_like(data)
    half_window = window_size // 2

    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        smoothed[i] = np.mean(data[start:end])

    return smoothed


def classify_ellipsoidal_vs_fusiform(contour):
    """
    Determine if the contour is more ellipsoidal or fusiform

    Args:
        contour: OpenCV contour, format: [[x, y], [x, y], ...]

    Returns:
        str: "ellipsoidal" 或 "fusiform"
    """
    # 将轮廓点转换为numpy数组
    points = contour.reshape(-1, 2).astype(np.float32)

    # 使用PCA找到主轴方向
    pca = PCA(n_components=2)
    pca.fit(points)

    # 获取主轴方向向量
    major_axis = pca.components_[0]

    # 将点投影到主轴上
    center = np.mean(points, axis=0)
    centered_points = points - center
    projections = np.dot(centered_points, major_axis)

    # 计算轮廓的长轴长度
    major_length = np.max(projections) - np.min(projections)

    # 沿主轴方向均匀取样，分析轮廓宽度变化
    num_samples = 25
    sample_positions = np.linspace(np.min(projections), np.max(projections), num_samples)
    widths = []

    for pos in sample_positions:
        # 找到投影位置附近的点
        tolerance = major_length * 0.04  # 4%的容差
        nearby_indices = np.where(np.abs(projections - pos) <= tolerance)[0]

        if len(nearby_indices) > 0:
            nearby_points = points[nearby_indices]
            # 计算这些点到主轴的距离
            distances_to_axis = []
            for point in nearby_points:
                # 计算点到主轴的距离
                point_centered = point - center
                proj_on_axis = np.dot(point_centered, major_axis) * major_axis
                distance = np.linalg.norm(point_centered - proj_on_axis)
                distances_to_axis.append(distance)

            # 取最大距离作为该位置的宽度
            widths.append(max(distances_to_axis) * 2)
        else:
            # 使用插值填充缺失值
            if len(widths) > 0:
                widths.append(widths[-1])
            else:
                widths.append(0)

    # 分析宽度变化模式
    widths = np.array(widths)
    valid_widths = widths[widths > 0]

    if len(valid_widths) < 5:
        return "ellipsoidal"  # 数据不足时默认为椭圆

    widths = valid_widths

    # 对宽度进行平滑处理
    smoothed_widths = simple_smooth(widths, window_size=3)

    # 1. 分析端部与中部的宽度关系
    # 取两端各20%和中间30%的区域
    end_samples = max(2, len(widths) // 5)  # 20%
    mid_samples = max(3, len(widths) // 3)  # 30%

    # 两端宽度
    left_end = np.mean(smoothed_widths[:end_samples])
    right_end = np.mean(smoothed_widths[-end_samples:])
    avg_end_width = (left_end + right_end) / 2

    # 中部宽度
    mid_start = len(smoothed_widths) // 2 - mid_samples // 2
    mid_end = len(smoothed_widths) // 2 + mid_samples // 2
    mid_width = np.mean(smoothed_widths[mid_start:mid_end])

    # 端中比
    end_to_mid_ratio = avg_end_width / mid_width if mid_width > 0 else 1.0

    # 2. 分析形状的对称性和单调性
    # 检查从端部到中部是否呈现单调增长模式
    left_half = smoothed_widths[: len(smoothed_widths) // 2]
    right_half = smoothed_widths[len(smoothed_widths) // 2 :][::-1]  # 反转右半部分

    # 计算左半部分的单调性（应该递增）
    left_increases = np.sum(np.diff(left_half) > 0)
    left_total = len(np.diff(left_half))
    left_monotonic = left_increases / left_total if left_total > 0 else 0

    # 计算右半部分的单调性（反转后应该递增）
    right_increases = np.sum(np.diff(right_half) > 0)
    right_total = len(np.diff(right_half))
    right_monotonic = right_increases / right_total if right_total > 0 else 0

    # 总体单调性
    overall_monotonicity = (left_monotonic + right_monotonic) / 2

    # 3. 分析宽度变化的平滑程度
    width_changes = np.diff(smoothed_widths)
    smoothness = 1 - (np.std(width_changes) / np.mean(smoothed_widths)) if np.mean(smoothed_widths) > 0 else 0
    smoothness = max(0, min(1, smoothness))  # 限制在[0,1]范围内

    # 4. 计算椭圆拟合误差
    # 椭圆应该更符合标准椭圆的宽度分布
    # 生成理想椭圆的宽度分布用于比较
    t = np.linspace(-1, 1, len(smoothed_widths))
    ideal_ellipse_widths = np.sqrt(1 - t**2) * np.max(smoothed_widths)

    # 归一化到相同的最大值
    normalized_widths = smoothed_widths / np.max(smoothed_widths)
    normalized_ideal = ideal_ellipse_widths / np.max(ideal_ellipse_widths)
    ellipse_error = np.mean(np.abs(normalized_widths - normalized_ideal))

    # 5. 综合评分
    fusiform_score = 0
    ellipsoidal_score = 0

    # 端中比评分 - 纺锤形端部应该明显更窄
    if end_to_mid_ratio < 0.3:
        fusiform_score += 4
    elif end_to_mid_ratio < 0.5:
        fusiform_score += 3
    elif end_to_mid_ratio < 0.7:
        fusiform_score += 1
    elif end_to_mid_ratio > 0.85:
        ellipsoidal_score += 2

    # 椭圆拟合误差评分 - 椭圆应该更符合标准椭圆（优先级高）
    if ellipse_error < 0.05:
        ellipsoidal_score += 4
    elif ellipse_error < 0.1:
        ellipsoidal_score += 3
    elif ellipse_error < 0.15:
        ellipsoidal_score += 1
    elif ellipse_error > 0.25:
        fusiform_score += 2

    # 单调性评分 - 纺锤形应该有很好的单调性，但椭圆也可能有
    if overall_monotonicity > 0.9 and end_to_mid_ratio < 0.6:
        fusiform_score += 2
    elif overall_monotonicity > 0.8 and end_to_mid_ratio < 0.5:
        fusiform_score += 1
    elif overall_monotonicity < 0.6:
        ellipsoidal_score += 1

    # 平滑度评分 - 纺锤形变化应该更平滑
    if smoothness > 0.9:
        fusiform_score += 1
    elif smoothness < 0.5:
        ellipsoidal_score += 1

    # 最终判断
    if fusiform_score > ellipsoidal_score:
        return "fusiform"
    else:
        return "ellipsoidal"
