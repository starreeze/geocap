import json
import os
import re
import numpy as np

import cv2
from common.args import feat_recog_args
from feat_recognize.utils import circle_weight_array


class ProloculusDetector:
    def __init__(self, block_num: int = 3):
        self.block_num = block_num

    def find_center(self, size_proloculus: int, size_ratio: float = 1.0, threshold: float = 0.25):
        # Find center of proloculus using sliding window
        window_size = int(size_proloculus * size_ratio)

        img_array = self.img_center_block
        windows = np.lib.stride_tricks.sliding_window_view(img_array, (window_size, window_size))
        windows = windows.reshape(-1, window_size, window_size)

        pos_weight_array, neg_weight_array, total_positive_weight, total_negative_weight = (
            circle_weight_array(window_size)
        )
        candidate_centers = []
        for i, window in enumerate(windows):
            # Calculate the score of the window
            pos_score = np.sum(window / 255 * pos_weight_array) / total_positive_weight
            neg_score = np.sum(window / 255 * neg_weight_array) / total_negative_weight
            score = pos_score + neg_score

            if score > threshold:
                # Calculate the row and column indices from the flattened index
                row_idx = i // (img_array.shape[1] - window_size + 1)
                col_idx = i % (img_array.shape[1] - window_size + 1)

                # Calculate the center coordinates of the window
                center_y = row_idx + window_size // 2
                center_x = col_idx + window_size // 2

                # Calculate the distance to the center of the image
                distance = np.sqrt(
                    (center_x - self.block_width / 2) ** 2 + (center_y - self.block_height / 2) ** 2
                )

                # Add the center coordinates to candidate_centers
                candidate_centers.append((center_x, center_y, score, distance / self.block_width))

        candidate_centers = sorted(candidate_centers, key=lambda x: x[2] - x[3], reverse=True)

        return candidate_centers

    def detect_initial_chamber(self, image_path_to_detect: str, threshold: float = 0.25):
        self.img = cv2.imread(image_path_to_detect)
        self.width, self.height = self.img.shape[1], self.img.shape[0]
        # Convert to grayscale and extract center block
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_center_block = self.get_center_block()

        # Calculate score with different window size
        points_with_max_score = []
        min_size, max_size = 6, 100
        for size in range(min_size, min(self.block_height, max_size) + 1, 2):
            candidate_centers = self.find_center(size_proloculus=size, threshold=threshold)
            if len(candidate_centers) > 0:
                points_with_max_score.append(
                    {
                        "size": size,
                        "points": [candidate[:2] for candidate in candidate_centers[:]],
                        "score": [candidate[2] for candidate in candidate_centers[:]],
                        "distance": [candidate[3] for candidate in candidate_centers[:]],
                    }
                )

        sizes = [
            point_with_max_score["size"] * feat_recog_args.inner_radius_ratio
            for point_with_max_score in points_with_max_score
        ]
        scores = [
            point_with_max_score["score"][0] - point_with_max_score["distance"][0]
            for point_with_max_score in points_with_max_score
        ]

        # Find the point with the highest score
        max_score_index = scores.index(max(scores))
        max_score_point = points_with_max_score[max_score_index]["points"][0]
        diameter = sizes[max_score_index]

        x = max_score_point[0] + self.block_width * (self.block_num // 2)
        y = max_score_point[1] + self.block_height * (self.block_num // 2)
        return [x, y, diameter]

    def get_center_block(self):
        self.block_height = self.height // self.block_num
        self.block_width = self.width // self.block_num

        x_start = (self.block_num // 2) * self.block_width
        x_end = (self.block_num // 2 + 1) * self.block_width
        y_start = (self.block_num // 2) * self.block_height
        y_end = (self.block_num // 2 + 1) * self.block_height

        center_block = self.img[y_start:y_end, x_start:x_end]
        return center_block
