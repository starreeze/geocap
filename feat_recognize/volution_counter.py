import os
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from feat_recognize.initial_chamber import detect_initial_chamber
from feat_recognize.utils import bresenham, fit_line, get_bbox, split_into_segments, resize_img


class VolutionCounter:
    def __init__(
        self,
        feat_recog_args,
        width_ratio: float = 0.3,
        adsorption_thres: float = 0.8,
        volution_thres: float = 0.85,
        step: int = 3,
        num_segments: int = 50,
        filter_max_y_ratio: float = 0.01,
        max_adsorption_time: int = 7,
        use_initial_chamber: bool = True,
    ):
        self.feat_recog_args = feat_recog_args
        self.width_ratio = width_ratio
        self.adsorption_thres = adsorption_thres
        self.volution_thres = volution_thres
        if hasattr(feat_recog_args, "volution_thres"):
            self.volution_thres = feat_recog_args.volution_thres
        self.step = step
        self.num_segments = num_segments
        self.filter_max_y_ratio = filter_max_y_ratio
        self.max_adsorption_time = max_adsorption_time
        self.use_initial_chamber = use_initial_chamber

    def process_img(self, img_path: str):
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_rgb = img[:, :, :3]

        # Opening preprocess and resize
        kernel = np.ones((3, 3), np.int8)
        img_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, kernel)
        # img_rgb = cv2.resize(img_rgb, (896, 448))
        # img = cv2.resize(img, (896, 448))
        img_rgb = resize_img(img_rgb)
        img = resize_img(img)

        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        assert img_gray.ndim == 2, "grayscale image required."

        # Detect initial chamber (with a high confidence level)
        initial_chamber = detect_initial_chamber(img_gray, param2=self.feat_recog_args.houghcircle_params["param2"])
        if self.use_initial_chamber and initial_chamber is not None:
            self.center = initial_chamber[:-1].tolist()
            self.success_initial_chamber = True
        else:
            min_row, max_row, min_col, max_col = get_bbox(img_gray)
            self.center = [(min_col + max_col) // 2, (min_row + max_row) // 2]
            self.success_initial_chamber = False

        # Binarization
        img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)

        # Morphological opening to remove noise
        kernel = np.ones((5, 5), np.int8)
        self.img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)

        # self.get_outer_volution(img)
        self.get_outer_volution()

    def set_scan_direction(self, line: list[tuple[int, int]]):
        self.direction = np.sign(self.center[1] - line[0][1])

    # def get_outer_volution(self, img: np.ndarray):
    #     assert img.shape[-1] == 4
    #     # Process countour info
    #     img_countour = img[:, :, 3]
    #     contours, _ = cv2.findContours(img_countour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #     contour = max(contours, key=cv2.contourArea)

    #     # Get initial line for volution detection
    #     line_upper = []
    #     line_lower = []
    #     x_mid, y_mid = self.center
    #     mid_img_width = int(self.width_ratio * 0.5 * img.shape[1])
    #     x_range = [x_mid - mid_img_width, x_mid + mid_img_width]
    #     for point in contour:
    #         x, y = point[0]
    #         if x_range[0] <= x <= x_range[1]:
    #             if y > y_mid:
    #                 line_lower.append((x, y))
    #             elif y < y_mid:
    #                 line_upper.append((x, y))

    #     self.line_upper = sorted(line_upper, key=lambda point: point[0])
    #     self.line_lower = sorted(line_lower, key=lambda point: point[0])

    def get_outer_volution(self):
        img_gray = self.img_gray
        h, w = img_gray.shape

        x_mid = self.center[0]
        y_top = int(np.min(np.where(img_gray[:, x_mid] == 0)[0]))
        y_bottom = int(np.max(np.where(img_gray[:, x_mid] == 0)[0]))

        mid_img_width = int(self.width_ratio * 0.5 * w)

        line_upper = [(x_mid, y_top)]
        line_lower = [(x_mid, y_bottom)]
        for i in range(1, mid_img_width):
            x1 = x_mid + i
            x2 = x_mid - i
            for x in [x1, x2]:
                # add top point to line_upper
                for y in range(h):
                    if img_gray[y, x] == 0:
                        line_upper.append((x, y))
                        break

                # add bottom point to line_lower
                for y in range(h - 1, -1, -1):
                    if img_gray[y, x] == 0:
                        line_lower.append((x, y))
                        break

        self.line_upper = sorted(line_upper, key=lambda point: point[0])
        self.line_lower = sorted(line_lower, key=lambda point: point[0])

    def is_adsorption(self, points: list[tuple[int, int]]) -> bool:
        indensity = 0
        for x, y in points:
            if self.img_gray[y, x] == 0:
                indensity += 1
        return indensity / len(points) > self.adsorption_thres

    def is_volution(self, check_adsorption_mask: list[bool]) -> bool:
        if check_adsorption_mask:
            adsorption_rate = sum(check_adsorption_mask) / len(check_adsorption_mask)
            return adsorption_rate > self.volution_thres
        else:
            return False

    def move(self, points: list[tuple[int, int]]) -> tuple[bool, bool]:
        """Move the points towards the center of the volution."""
        finish = False
        for i in range(len(points)):
            x, y = points[i]
            target_x, target_y = self.center
            # Calculate the direction vector
            direction_x = target_x - x
            direction_y = target_y - y

            if direction_y * self.direction <= 0:
                finish = True
                break

            # Normalize the direction vector
            magnitude = np.sqrt(direction_x**2 + direction_y**2)
            direction_x /= magnitude
            direction_y /= magnitude

            # Move a small distance towards the target point
            step_size = self.step * magnitude / abs(target_y - y)  # self.step / cos(\theta)
            points[i] = (int(x + direction_x * step_size), int(y + direction_y * step_size))

        adsorption_mask = self.is_adsorption(points)
        return adsorption_mask, finish

    def catch_frontier(
        self, line_segments: list[list[tuple[int, int]]], step_forward: list[int], i: int, mask: bool, finish: bool
    ):
        max_step = max(step_forward)
        num_step = max_step - step_forward[i] - 1
        for _ in range(num_step):
            mask, finish = self.move(line_segments[i])
            step_forward[i] += 1
        return mask, finish

    def filter_segments(self, line_segments: list[list[tuple[int, int]]], step_forward: Optional[list] = None):
        y_means = np.array([np.mean([point[1] for point in segment]) for segment in line_segments])
        ref_y = np.median(y_means)
        filter_max_y = self.filter_max_y_ratio * self.img_gray.shape[1]

        filtered_segments = []
        filtered_step_forward = []
        for i, segment in enumerate(line_segments):
            save = True
            # Filter out segment that are far away from ref_y
            for point in segment:
                if abs(point[1] - ref_y) > filter_max_y:
                    save = False
                    break

            # Filter out discontinuous segment
            for s in range(len(segment) - 1):
                point = segment[s]
                next_point = segment[s + 1]
                if abs(point[1] - next_point[1]) > 3:
                    save = False
                    break

            if save:
                filtered_segments.append(segment)
                if step_forward:
                    filtered_step_forward.append(step_forward[i])

        line_segments = filtered_segments
        if step_forward is not None:
            step_forward = filtered_step_forward

        return line_segments, step_forward

    def get_continuous_black_line(self, vertex: tuple[int, int], theta: float, max_expand_len: int):
        points = []
        for r in range(max_expand_len):
            x = int(vertex[0] + r * np.cos(theta))
            y = int(vertex[1] + r * np.sin(theta))
            if x < 0 or x >= self.img_gray.shape[1] or y < 0 or y >= self.img_gray.shape[0]:
                break
            if self.img_gray[y, x] == 255:
                break
            points.append((x, y))

        return points, len(points)

    def get_expand_points(self, vertex: tuple[int, int], theta_ref: float, theta_margin: float, max_expand_len: int):
        expand_points = []
        max_score = -1
        theta_range = np.arange(theta_ref - theta_margin, theta_ref + theta_margin, 0.01)
        for theta in theta_range:
            points, num_points = self.get_continuous_black_line(vertex, theta, max_expand_len)
            score = (num_points / max_expand_len) - 1.0 * (abs(theta - theta_ref) / theta_margin)
            if score > max_score:
                expand_points = points
                max_score = score
                new_theta_ref = theta

        return expand_points, new_theta_ref

    def expand_line_segments(
        self,
        line_segments: list[list[tuple[int, int]]],
        num_volutions: int,
        theta_margin: float = 0.5 * np.pi,
        max_expand_times: int = 10,
        max_expand_len: int = 10,
        base_expand_ratio: float = 0.8,
    ):
        num_original_points = sum([len(segment) for segment in line_segments])
        num_segs = len(line_segments)
        max_expand_ratio = base_expand_ratio**num_volutions

        # Calculate the slope of the left part
        left_points = []
        for segment in line_segments[: num_segs // 3]:
            left_points.extend(segment)
        vertex = line_segments[0][0]
        slope, intercept = fit_line(left_points)
        if np.isnan(slope):
            return line_segments
        theta_ref = np.arctan(slope) + np.pi

        # Expand left black pixel
        num_expand_points = 0
        for _ in range(max_expand_times):
            expand_points, _ = self.get_expand_points(vertex, theta_ref, theta_margin, max_expand_len)

            n = len(line_segments[0])
            if expand_points:
                vertex = expand_points[-1]
                for i in range(0, len(expand_points), n):
                    new_segments = expand_points[i + n : i : -1]  # reverse order
                    if len(new_segments) >= n:
                        line_segments.insert(0, new_segments)
                        num_expand_points += len(new_segments)

            if num_expand_points >= max_expand_ratio * num_original_points:
                break

        # Calculate the slope of the right part
        right_points = []
        for segment in line_segments[2 * num_segs // 3 :]:
            right_points.extend(segment)
        vertex = line_segments[-1][-1]
        slope, intercept = fit_line(right_points)
        if np.isnan(slope):
            return line_segments
        theta_ref = np.arctan(slope)

        # Expand right black pixel
        num_expand_points = 0
        for _ in range(max_expand_times):
            expand_points, _ = self.get_expand_points(vertex, theta_ref, theta_margin, max_expand_len)

            n = len(line_segments[-1])
            if expand_points:
                vertex = expand_points[-1]
                for i in range(0, len(expand_points), n):
                    new_segments = expand_points[i : i + n + 1]
                    if len(new_segments) >= n:
                        line_segments.append(new_segments)
                        num_expand_points += len(new_segments)

            if num_expand_points >= max_expand_ratio * num_original_points:
                break

        return line_segments

    def update_line_segments(self, line_segments: list[list[tuple[int, int]]], num_split: int = 9):
        if not line_segments:
            return line_segments
        new_line = []
        while num_split > len(line_segments):
            num_split = num_split // 2
        split_len = len(line_segments) // num_split

        if len(line_segments) - split_len > 0:
            for idx in range(0, len(line_segments) - split_len, split_len):
                p1 = line_segments[idx][0]
                p2 = line_segments[idx + split_len][0]
                new_line.extend(bresenham(p1, p2))

            new_line.extend(bresenham(p2, line_segments[-1][-1]))
            line_segments = split_into_segments(new_line, self.num_segments)
        return line_segments

    def reach_step_limit(self, step_forward: list[int]) -> bool:
        max_step = max(step_forward)
        limit = 0.1 * self.img_gray.shape[0] / self.step
        return max_step >= limit

    def scan_in_volution(self, line_segments: list[list[tuple[int, int]]]):
        check_adsorption_mask = [self.is_adsorption(segment) for segment in line_segments]

        finish = False
        step_forward = [0 for _ in range(len(line_segments))]

        # Move line_segments until all segments leave current volution
        while any(check_adsorption_mask) and not finish:
            for i, adsorption in enumerate(check_adsorption_mask):
                if adsorption:
                    check_adsorption_mask[i], finish = self.move(line_segments[i])
                    step_forward[i] += 1
            if self.reach_step_limit(step_forward):
                break

        # Post process: filter out segments that moved too far
        line_segments, step_forward = self.filter_segments(line_segments, step_forward)

        assert isinstance(step_forward, list)
        thickness = np.mean(step_forward) * self.step

        for i in range(len(line_segments)):
            _, finish = self.catch_frontier(line_segments, step_forward, i, check_adsorption_mask[i], finish)

        return line_segments, thickness, finish

    def scan_between_volutions(self, line_segments: list[list[tuple[int, int]]]):
        finish = False
        check_adsorption_mask = [self.is_adsorption(segment) for segment in line_segments]
        step_forward = [0 for _ in range(len(line_segments))]
        cur_adsorption_time = [0 for _ in range(len(line_segments))]

        while not self.is_volution(check_adsorption_mask) and not finish:
            for i, adsorption in enumerate(check_adsorption_mask):
                if adsorption:
                    cur_adsorption_time[i] += 1
                    if cur_adsorption_time[i] == self.max_adsorption_time:
                        # end adsorption and catch frontier segment
                        check_adsorption_mask[i], finish = self.catch_frontier(
                            line_segments, step_forward, i, check_adsorption_mask[i], finish
                        )
                        cur_adsorption_time[i] = 0
                else:
                    check_adsorption_mask[i], finish = self.move(line_segments[i])
                    step_forward[i] += 1

        if not finish:
            for i in range(len(line_segments)):
                check_adsorption_mask[i], finish = self.catch_frontier(
                    line_segments, step_forward, i, check_adsorption_mask[i], finish
                )

        line_segments, _ = self.filter_segments(line_segments, step_forward)

        return line_segments, finish

    def count_volutions(self, img_path: str) -> tuple[dict[int, list], dict[int, float], bool]:
        """
        Detect the volutions and measure the thickness of each volution in the image.
        Try to detect the initial chamber with a high confidence level.

        Parameters:
        img_path (string): The path of input image.

        Returns:
        volutions_dict (dict): Dictionary of detected volutions where:
            * Key (int): Volution index number, 1 represents the innermost volution:
                - Positive numbers represent upper volutions (e.g., 1, 2, 3...)
                - Negative numbers represent lower volutions (e.g., -1, -2, -3...)
            * Value (list): List of (x, y) tuples representing points along the volution curve,
                where x and y are normalized coordinates (0.0 to 1.0)

        thickness_dict (dict): Dictionary of volution thickness measurements where:
            * Key (int): Volution index number (matches keys in volutions_dict)
            * Value (float): Normalized thickness of the volution (0.0 to 1.0)

        success_initial_chamber (bool): Whether the initial chamber is detected successfully (with a high confidence level).
        """
        self.process_img(img_path)

        volutions = [[], []]  # upper and lower
        thickness_per_vol = [[], []]

        for i, line in enumerate([self.line_upper, self.line_lower]):
            self.set_scan_direction(line)
            finish = False
            line_segments = split_into_segments(line, self.num_segments)
            line_segments, _ = self.filter_segments(line_segments)
            line_segments = self.update_line_segments(line_segments)

            while not finish:
                detected_volution = [segment[0] for segment in line_segments]
                # Remove points with duplicate x values
                unique_x_points = {}
                for point in detected_volution:
                    x = point[0]
                    if x not in unique_x_points:
                        unique_x_points[x] = point
                detected_volution = list(unique_x_points.values())
                volutions[i].append(detected_volution)

                line_segments, thickness, finish = self.scan_in_volution(line_segments)

                thickness_per_vol[i].append(thickness)
                if finish:
                    break

                line_segments, finish = self.scan_between_volutions(line_segments)
                if finish:
                    break

                line_segments = [segment for segment in line_segments if self.is_adsorption(segment)]

                line_segments = self.update_line_segments(line_segments)

                line_segments = self.expand_line_segments(line_segments, len(volutions[i]))
                line_segments = self.update_line_segments(line_segments)

                if not line_segments:
                    finish = True

        # Return relative values
        h, w = self.img_gray.shape
        for volution in volutions:
            for line_segments in volution:
                for i, point in enumerate(line_segments):
                    line_segments[i] = (point[0] / w, point[1] / h)

        for thickness_list in thickness_per_vol:
            for i in range(len(thickness_list)):
                thickness_list[i] = thickness_list[i] / h

        # Reformat into dict
        volutions_dict = {}
        thickness_dict = {}
        for i in range(2):
            vols = volutions[i]
            thicks = thickness_per_vol[i]
            for j in range(len(vols)):
                vol_idx = (len(vols) - j) * (-1) ** i  # positive value for upper voluitons, negative for lower
                volutions_dict[vol_idx] = vols[j]
                thickness_dict[vol_idx] = thicks[j]

        return volutions_dict, thickness_dict, self.success_initial_chamber
