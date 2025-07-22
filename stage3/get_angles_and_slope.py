import cv2
import numpy as np
from PIL import Image

from common.args import logger

size_ = (128, 128)


def find_contour(img):
    if type(img) is str:
        img_rgb = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        if img_rgb is None:
            logger.error(f"Cannot read Image {img}")
            raise Exception("Cannot read Image")
    else:
        img_rgb = np.asarray(Image.fromarray(img).resize(size_, Image.LANCZOS))
    img_gray = img_rgb[:, :, 3]
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # img2=np.zeros(img_rgb.shape)*(0,0,0,0)
    # cv2.drawContours(img2,[max(contours,key=cv2.contourArea)],0,(255,255,255,255),thickness=10)
    # cv2.imwrite("test.png",img2)
    if len(contours) == 0:
        return "random"
    return max(contours, key=cv2.contourArea)


def classify_points(contour, mid_x, n, center):
    distances = np.abs(contour[:, 0] - mid_x)

    closest_indices = np.argpartition(distances, n)[:n]
    candidate_points = contour[closest_indices]

    center_y = center[1]
    upper_points = candidate_points[candidate_points[:, 1] > center_y]
    lower_points = candidate_points[candidate_points[:, 1] < center_y]
    assert len(upper_points) > 0 and len(lower_points) > 0
    return upper_points, lower_points


def find_extremes(contour):
    x = contour[:, 0]
    y = contour[:, 1]

    leftmost_idx = np.argmin(x)
    rightmost_idx = np.argmax(x)
    topmost_idx = np.argmax(y)
    bottommost_idx = np.argmin(y)

    leftmost = contour[leftmost_idx]
    rightmost = contour[rightmost_idx]
    topmost = contour[topmost_idx]
    bottommost = contour[bottommost_idx]

    return np.array([leftmost, rightmost, topmost, bottommost])


def find_extrema_points(points):
    if len(points) < 3:
        raise ValueError("Input points must have at least 3 coordinates.")

    sorted_points = points[points[:, 0].argsort()]
    x = sorted_points[:, 0]
    y = sorted_points[:, 1]

    dy = np.diff(y)
    dx = np.diff(x) + 1e-8  # avoid dx = 0
    derivatives = dy / dx

    extrema_indices = []
    local_trend = 0
    for d in derivatives:
        if d == 0:
            continue
        else:
            local_trend = np.sign(d)
            break
    for i in range(len(derivatives)):
        if local_trend * derivatives[i] < 0:
            extrema_indices.append(i)
            local_trend = -local_trend
    extrema_points = sorted_points[extrema_indices]
    if extrema_points.size == 0:
        logger.warning(
            f"No extrema points found. Using midpoint {sorted_points[len(sorted_points) // 2]} instead."
        )
        return np.array([sorted_points[len(sorted_points) // 2]])
    return extrema_points


def get_start(extremas, mid_x, contour):
    x_coords = extremas[:, 0]
    abs_diffs = np.abs(x_coords - mid_x)
    idx = np.argmin(abs_diffs)
    p0 = extremas[idx]

    distances = np.linalg.norm(contour - p0, axis=1)
    start_idx = np.argmin(distances)

    tol = 1e-5
    y0 = p0[1]

    left_points = []
    current_idx = start_idx - 1
    while current_idx >= 0 and abs(contour[current_idx][1] - y0) <= tol:
        left_points.append(contour[current_idx])
        current_idx -= 1

    right_points = []
    current_idx = start_idx + 1
    while current_idx < len(contour) and abs(contour[current_idx][1] - y0) <= tol:
        right_points.append(contour[current_idx])
        current_idx += 1

    all_points = left_points[::-1] + [p0] + right_points

    if len(all_points) == 0:
        return p0

    all_points_sorted = sorted(all_points, key=lambda p: p[0])
    mid_index = len(all_points_sorted) // 2
    return all_points_sorted[mid_index]


def recheck(start, neighbor_points, contour):
    for n in neighbor_points:
        if start[0] == n[0] and start[1] == n[1]:
            return neighbor_points
    else:
        logger.warning("Start point has shifted out of neighborhood.")
        mask = (contour == start).all(axis=1)
        idx = np.where(mask)[0][0]
        window = contour[idx - len(contour) // 8 : idx + len(contour) // 8]
        neighbor_points = np.unique(np.vstack([neighbor_points, window]), axis=0)
        return neighbor_points


def get_angle(start, side, neighbor_points):
    mask = (neighbor_points == start).all(axis=1)
    idx = np.where(mask)[0][0]

    n_total = len(neighbor_points)
    left_list = []
    right_list = []
    count_same = 1

    left_points = neighbor_points[:idx][::-1] if idx > 0 else np.array([])
    right_points = neighbor_points[idx + 1 :] if idx < n_total - 1 else np.array([])

    n_left = len(left_points)
    if n_left > 0:
        indices = np.linspace(0, n_left - 1, min(15, n_left), dtype=int)
        for i in indices:
            p = left_points[i]
            if p[0] == start[0] or p[1] == start[1]:
                if count_same < 3:
                    left_list.append(p)
                    count_same += 1
            else:
                left_list.append(p)

    n_right = len(right_points)
    if n_right > 0:
        indices = np.linspace(0, n_right - 1, min(15, n_right), dtype=int)
        for i in indices:
            p = right_points[i]
            if p[0] == start[0] or p[1] == start[1]:
                if count_same < 3:
                    right_list.append(p)
                    count_same += 1
            else:
                right_list.append(p)

    new_points = left_list[::-1] + [start] + right_list
    neighbor_points = np.array(new_points)
    win_size = len(neighbor_points)
    n = win_size
    half_win = max(1, win_size // 2)
    new_idx = len(left_list)

    dir_vec_dict = {
        "l": np.array([1, 0]),
        "r": np.array([-1, 0]),
        "u": np.array([0, -1]),
        "d": np.array([0, 1]),
    }
    dirvec = dir_vec_dict[side]
    angles = []

    for i in range(1, half_win + 1):
        idxA = new_idx - i
        idxB = new_idx + i

        if idxA < 0 or idxB >= n:
            break

        A = neighbor_points[idxA]
        B = neighbor_points[idxB]

        SA = A - start
        SB = B - start

        if np.linalg.norm(SA) < 1e-5 or np.linalg.norm(SB) < 1e-5:
            continue

        norm_dir = np.linalg.norm(dirvec)
        cosA = np.dot(SA, dirvec) / (np.linalg.norm(SA) * norm_dir)
        cosB = np.dot(SB, dirvec) / (np.linalg.norm(SB) * norm_dir)

        cosA = np.clip(cosA, -1.0, 1.0)
        cosB = np.clip(cosB, -1.0, 1.0)

        angleA = np.arccos(cosA)
        angleB = np.arccos(cosB)

        if angleA < 0:
            angleA += 2 * np.pi
        if angleB < 0:
            angleB += 2 * np.pi

        if np.abs(angleA - np.pi / 2) <= 1e-5 or np.abs(angleB - np.pi / 2) <= 1e-5:
            if len(angles) > 3:
                logger.info(f"Side {side} discarded a point with angles {angleA:.4f}, {angleB:.4f}")
                continue

        theta = angleA + angleB
        if theta < 0:
            theta += 2 * np.pi
        elif theta >= 2 * np.pi:
            theta -= 2 * np.pi

        angles.append(theta)

    if not angles:
        return np.pi  # flat

    return np.mean(angles)


def separate_slopes(slope):
    center = np.mean(slope, axis=0)

    dx = slope[:, 0] - center[0]
    dy = slope[:, 1] - center[1]
    angles = np.arctan2(dy, dx)

    angles_2pi = np.where(angles < 0, angles + 2 * np.pi, angles)

    sorted_indices = np.argsort(angles_2pi)
    sorted_points = slope[sorted_indices]
    sorted_angles = angles_2pi[sorted_indices]

    angular_diffs = np.diff(sorted_angles)
    wrap_around_diff = (sorted_angles[0] - sorted_angles[-1] + 2 * np.pi) % (2 * np.pi)
    angular_diffs = np.append(angular_diffs, wrap_around_diff)

    largest_diff_indices = np.argsort(angular_diffs)[-4:]
    split_positions = np.sort(largest_diff_indices) + 1
    # print(split_positions)

    if split_positions[-1] == len(sorted_points):
        regions = np.split(sorted_points, split_positions[:-1])
    else:
        regions = np.split(sorted_points, split_positions)

    if len(regions) == 5:
        region1 = np.vstack((regions[-1], regions[0]))
        regions = [region1, regions[1], regions[2], regions[3]]

    if len(regions) != 4:
        raise ValueError("Not enough regions.")

    centers = np.array([np.mean(region, axis=0) for region in regions])

    offsets = centers - center

    upper_regions = []
    lower_regions = []
    for i, offset in enumerate(offsets):
        if offset[1] >= 0:
            upper_regions.append(regions[i])
        else:
            lower_regions.append(regions[i])

    upper_centers = [np.mean(region, axis=0) for region in upper_regions]
    upper_sorted = [region for _, region in sorted(zip(upper_centers, upper_regions), key=lambda x: x[0][0])]

    lower_centers = [np.mean(region, axis=0) for region in lower_regions]
    lower_sorted = [region for _, region in sorted(zip(lower_centers, lower_regions), key=lambda x: x[0][0])]

    ordered_regions = []
    if len(upper_sorted) > 0:
        ordered_regions.append(upper_sorted[0])
    if len(upper_sorted) > 1:
        ordered_regions.append(upper_sorted[1])
    if len(lower_sorted) > 0:
        ordered_regions.append(lower_sorted[0])
    if len(lower_sorted) > 1:
        ordered_regions.append(lower_sorted[1])

    return ordered_regions


def deriv2(region):
    sorted_indices = np.lexsort((region[:, 1], region[:, 0]))
    sorted_region = region[sorted_indices]

    x = sorted_region[:, 0]
    y = sorted_region[:, 1]
    return check_curvature(x, y)


def check_curvature(x, y):
    n = len(y)
    if n < 2:
        return "flat"

    x_norm = (x - np.mean(x)) / (max(x) - min(x) + 1e-10)
    y_norm = (y - np.mean(y)) / (max(x) - min(x) + 1e-10)

    # Fit a line to get the general trend
    line_coeffs = np.polyfit(x_norm, y_norm, 1)
    m = line_coeffs[0]

    # Angle of the line
    theta = np.arctan(m)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rotate the coordinate system so the fitted line becomes the new x-axis
    x_rotated = x_norm * cos_theta + y_norm * sin_theta
    y_rotated = -x_norm * sin_theta + y_norm * cos_theta

    # Now fit the cubic polynomial on the rotated coordinates
    coeffs = np.polyfit(x_rotated, y_rotated, 3)
    """DEBUG
    import matplotlib.pyplot as plt
    plt.axis('equal')
    # Plot original normalized points
    plt.plot(x_norm, y_norm, 'o', label='Normalized Points')

    # Plot the fitted line (the new x-axis)
    line_x = np.array([min(x_norm), max(x_norm)])
    line_y = m * line_x + line_coeffs[1]
    plt.plot(line_x, line_y, 'r-', label='Fitted Line (New X-axis)')

    # To visualize the fitted cubic, create points along the rotated x-axis,
    # calculate the corresponding y from the polynomial, and then rotate
    # these (x,y) pairs back to the original coordinate system.
    function = np.poly1d(coeffs)
    px_rotated = np.linspace(min(x_rotated), max(x_rotated), 100)
    py_rotated = function(px_rotated)

    # Rotate the fitted curve back to the original coordinate system for plotting
    px_orig = px_rotated * cos_theta - py_rotated * sin_theta
    py_orig = px_rotated * sin_theta + py_rotated * cos_theta
    plt.plot(px_orig, py_orig, 'g-', label='Fitted Cubic Curve')

    print(coeffs)
    plt.legend()
    plt.show()
    plt.clf()
    #"""
    return coeffs


def conv_scoring(coeff, side, t3_threshold=0.5):
    if side == "u":
        positive = 1
    elif side == "d":
        positive = -1
    else:
        raise ValueError("Invalid side")

    def relu(x):
        return np.maximum(x, 0)

    # score = relu((np.abs(coeff[0]) - np.abs(coeff[1]) - t3_threshold) * 10) + coeff[1] * positive * 3
    score = coeff[1] * positive * 3
    return score


def extend(exclusion, contour):
    exclusion_dict = {}
    threshold = len(contour) // 16
    for point in exclusion:
        y, x = point[1], point[0]
        exclusion_dict.setdefault(y, []).append(x)
    result_list = []
    for point in contour:
        x, y = point
        if y in exclusion_dict:
            for ex_x in exclusion_dict[y]:
                if abs(x - ex_x) <= threshold:
                    result_list.append(point)
                    break
    return np.array(result_list, dtype=int) if result_list else np.empty((0, 2), dtype=int)


def get_convexity(contour, extremes, extended_lower, extended_upper, sample_rate):
    n_left_right = contour.shape[0] // sample_rate
    unique_left = np.array(sorted(contour, key=lambda p: abs(p[0] - extremes[0][0]))[:n_left_right])
    unique_right = np.array(sorted(contour, key=lambda p: abs(p[0] - extremes[1][0]))[:n_left_right])
    unique_left = np.array(sorted(unique_left, key=lambda p: p[1]))
    unique_right = np.array(sorted(unique_right, key=lambda p: p[1]))

    remove_set = set(map(tuple, np.vstack((unique_left, unique_right, extended_lower, extended_upper))))
    slopes = contour[~np.array([tuple((p)) in remove_set for p in contour])]
    slopes = separate_slopes(slopes)
    d2x_dy2 = [deriv2(slope) for slope in slopes]
    # logger.info(f"XP: {(np.pi/2 - np.arctan(d2x_dy2[0][2])+np.arctan(d2x_dy2[1][2]) + np.pi/2) * 180 / np.pi} {(np.arctan(d2x_dy2[2][2]) + np.pi/2+np.pi/2 - np.arctan(d2x_dy2[3][2])) * 180 / np.pi}")
    try:
        convex_score = [
            conv_scoring(d2x_dy2[0], "u"),
            conv_scoring(d2x_dy2[1], "u"),
            conv_scoring(d2x_dy2[2], "d"),
            conv_scoring(d2x_dy2[3], "d"),
        ]  # ↑concave, ↓convex
    except Exception as e:
        logger.info(f"{e} Cannot correctly divide 4 blocks. Retrying...")
        if sample_rate > 2:
            convex_score = get_convexity(contour, extremes, extended_lower, extended_upper, sample_rate - 1)
        else:
            raise Exception("The Fossil Picture is weird! Check the picture.")
    return convex_score


def get_angles_and_slope(path, debug=False):
    contour: np.ndarray = find_contour(path)
    contour = contour.squeeze(1)

    center = np.mean(contour, axis=0)
    max_y = np.max(contour[:, 1])
    contour[:, 1] = max_y - contour[:, 1]
    extremes = find_extremes(contour)
    mid_x = extremes[0][0] + (extremes[1][0] - extremes[0][0]) / 2
    center_y = np.mean(contour[:, 1])
    upper_mid_points, lower_mid_points = classify_points(
        contour, mid_x, n=contour.shape[0] // 3, center=center
    )
    extremes[2] = get_start(find_extrema_points(upper_mid_points), mid_x, contour)
    extremes[3] = get_start(find_extrema_points(lower_mid_points), mid_x, contour)

    n = contour.shape[0] // 4
    upper_selected = []
    lower_selected = []
    if np.abs(extremes[2][0] - mid_x) < np.abs(extremes[1][0] - extremes[0][0]) // 5:
        upper_selected.append(extremes[2][0])
    else:
        upper_selected.append(mid_x)
    if np.abs(extremes[3][0] - mid_x) < np.abs(extremes[1][0] - extremes[0][0]) // 5:
        lower_selected.append(extremes[3][0])
    else:
        lower_selected.append(mid_x)
    assert len(upper_selected) == 1 and len(lower_selected) == 1
    upper_chosen = upper_selected[0]
    lower_chosen = lower_selected[0]
    all_upper, _ = classify_points(contour, upper_chosen, n, center)
    _, all_lower = classify_points(contour, lower_chosen, n, center)

    unique_upper = np.unique(all_upper, axis=0)
    unique_lower = np.unique(all_lower, axis=0)
    upper_extrema = find_extrema_points(unique_upper)
    lower_extrema = find_extrema_points(unique_lower)

    n_left_right = contour.shape[0] // 12
    unique_left = np.array(sorted(contour, key=lambda p: abs(p[0] - extremes[0][0]))[:n_left_right])
    unique_right = np.array(sorted(contour, key=lambda p: abs(p[0] - extremes[1][0]))[:n_left_right])
    unique_left = np.array(sorted(unique_left, key=lambda p: p[1]))
    unique_right = np.array(sorted(unique_right, key=lambda p: p[1]))

    left_start = extremes[0]
    right_start = extremes[1]
    upper_start = get_start(upper_extrema, mid_x, contour)
    lower_start = get_start(lower_extrema, mid_x, contour)

    unique_lower = recheck(lower_start, unique_lower, contour)
    unique_upper = recheck(upper_start, unique_upper, contour)

    left_angle = get_angle(left_start, "l", unique_left)
    right_angle = get_angle(right_start, "r", unique_right)
    upper_angle = get_angle(upper_start, "u", unique_upper)
    lower_angle = get_angle(lower_start, "d", unique_lower)

    upper_mid_points, lower_mid_points = classify_points(contour, mid_x, n, center)
    extended_lower = extend(lower_mid_points, contour)
    extended_upper = extend(upper_mid_points, contour)

    convex_score = get_convexity(contour, extremes, extended_lower, extended_upper, 12)

    if debug:
        # V-DEBUG
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt

        img = mpimg.imread(path)
        h, w = img.shape[0], img.shape[1]

        plt.imshow(img, origin="upper", extent=[0, w, 0, h])

        plt.axis("equal")

        plt.xlim(0, w)
        plt.ylim(0, h)

        s = 2
        plt.axis("equal")
        plt.scatter(contour[:, 0], contour[:, 1], c="b", s=s)
        plt.scatter(unique_upper[:, 0], unique_upper[:, 1], c="g", s=s)
        plt.scatter(unique_lower[:, 0], unique_lower[:, 1], c="g", s=s)
        plt.scatter(unique_left[:, 0], unique_left[:, 1], c="g", s=s)
        plt.scatter(unique_right[:, 0], unique_right[:, 1], c="g", s=s)

        plt.scatter(upper_extrema[:, 0], upper_extrema[:, 1], c="r", s=s)
        plt.scatter(lower_extrema[:, 0], lower_extrema[:, 1], c="r", s=s)
        top_y = int(center_y * 2)
        plt.plot([mid_x] * top_y, range(0, top_y), c="black")
        plt.scatter(left_start[0], left_start[1], c="yellow", s=s)
        plt.scatter(right_start[0], right_start[1], c="yellow", s=s)
        plt.scatter(upper_start[0], upper_start[1], c="yellow", s=s)
        plt.scatter(lower_start[0], lower_start[1], c="yellow", s=s)

        # plt.scatter(extended_upper[:,0], extended_upper[:,1], c='purple', s=s)
        # plt.scatter(extended_lower[:,0], extended_lower[:,1], c='purple', s=s)

        plt.title(
            f"Left: {left_angle * 180 / 3.1416:.2f}° | Right: {right_angle * 180 / 3.1416:.2f}° | Upper: {upper_angle * 180 / 3.1416:.2f}° | Lower: {lower_angle * 180 / 3.1416:.2f}°"
        )
        plt.savefig("dataset/DEBUG_TEST.png")
        plt.clf()

    return (
        left_angle,
        right_angle,
        upper_angle,
        lower_angle,
        convex_score[0],
        convex_score[1],
        convex_score[2],
        convex_score[3],
    )


def main():
    # name = 'Chusenella_tenuis_1_3'
    # name = 'Fusulinella_juncea_1_10'
    # name = 'Eostaffella_Eostaffella_pseudostruvei_var_angusta_1_3'

    # name = 'Pseudoschwagerina_minatoi_1_4'
    # name = 'Fusulina_haworthi_1_6'
    # name = 'Fusulina_dunbari_1_3'

    # name = 'Fusulina_acme_1_10'
    # name = "Fusulina_acme_1_11"
    # name = "Pseudoschwagerina_simplex_1_1"

    # name = 'Neoschwagerina_majulensis_1_3'
    # name = 'Pseudoschwagerina_simplex_1_1'
    # name = 'Pseudoschwagerina_convexa_1_3'

    # name = 'Fusulina_cappsensis_1_1'
    # name = 'Fusulinella_hanzawai_1_1'
    # name = 'Rugosofusulina_serrata_1_1'

    name = "Fusulina_higginsvillensis_1_3"

    # name = 'Neoschwagerina_takagamiensis_1_1'
    # name = 'Rugosofusulina_latioralis_1_1'
    # name = "Pseudoschwagerina_intermedia_1_3"
    # name = 'Pseudoschwagerina_grinnelli_1_8'
    # name = 'Chusenella_minuta_1_1' # normal
    # name = 'Fusulinella_pinguis_1_7' # few samples
    # name = 'Chusenella_referta_1_2' # Another deeply concaved example
    # name = "Chusenella_extensa_1_3" # heavily biased
    # name = "Chusenella_absidata_1_1" # square like
    # name = "Chusenella_absidata_1_5" # Deeply concaved at bottom
    # name = np.random.choice(os.listdir("D:\\Python_Proj\\Work\\pics")).strip(".png")
    path = f"D:\\Python_Proj\\Work\\pics\\{name}.png"
    try:
        left_angle, right_angle, upper_angle, lower_angle, lu_slope, ru_slope, ld_slope, rd_slope = (
            get_angles_and_slope(path, debug=True)
        )
    except Exception as e:
        logger.error(f"{e} {name} Failed")
        assert 0
    logger.info(
        f"{name} Angle Info \nLeft: {left_angle * 180 / 3.1416:.2f}° | Right: {right_angle * 180 / 3.1416:.2f}° | Upper: {upper_angle * 180 / 3.1416:.2f}° | Lower: {lower_angle * 180 / 3.1416:.2f}°"
    )
    # print(lu_slope, ru_slope, ld_slope, rd_slope)
    logger.info(
        f"{name} Slope Info \nLU: {np.mean(lu_slope):.2f} | RU: {np.mean(ru_slope):.2f} | LD: {np.mean(ld_slope):.2f} | RD: {np.mean(rd_slope):.2f}"
    )


if __name__ == "__main__":
    # for _ in range(1000):
    main()
