from typing import Optional

import cv2
import numpy as np


def detect_initial_chamber(
    img: np.ndarray,
    sub_img: Optional[np.ndarray] = None,
    sub_img_pos: Optional[tuple] = None,
    block_num: int = 5,
    dp: float = 1.5,
    minDist: float = 100,
    param1: float = 150,
    param2: float = 0.5,
):
    """
    Detect the initial chamber in an image using Hough Circle Transform.

    Parameters:
    img (np.ndarray): The input image.
    sub_img (Optional[np.ndarray]): A sub-image to detect the chamber in.
    sub_img_pos (Optional[tuple]): The position of the sub-image in the main image.
    block_num (int): Number of blocks to divide the image into.
    dp (float): Inverse ratio of the accumulator resolution to the image resolution.
    minDist (float): Minimum distance between the centers of the detected circles.
    param1 (float): First method-specific parameter for HoughCircles.
    param2 (float): Second method-specific parameter for HoughCircles.

    Returns:
    initial chamber (list | None): The detected initial chamber coordinates and radius.
    """
    if sub_img is None:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert img.ndim == 2, "grayscale img required."
        height, width = img.shape

        kernel = np.ones((3, 3), np.int8)
        img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        block_height = height // block_num
        block_width = width // block_num
        x_start = (block_num // 2) * block_width
        x_end = (block_num // 2 + 1) * block_width
        y_start = (block_num // 2) * block_height
        y_end = (block_num // 2 + 1) * block_height
        center_block = img_open[y_start:y_end, x_start:x_end]
        img_to_detect = center_block
    else:
        assert isinstance(sub_img, np.ndarray) and isinstance(sub_img_pos, tuple)
        if sub_img.ndim == 3:
            sub_img = cv2.cvtColor(sub_img, cv2.COLOR_BGR2GRAY)
        assert sub_img.ndim == 2, "grayscale img required."

        img_to_detect = sub_img
        x_start, y_start = sub_img_pos

    houghcircles = cv2.HoughCircles(
        img_to_detect, cv2.HOUGH_GRADIENT_ALT, dp=dp, minDist=minDist, param1=param1, param2=param2
    )
    if houghcircles is None:
        return None

    idx = 0
    min_r = 10000
    houghcircles = houghcircles.astype(np.int16)
    for i, cir in enumerate(houghcircles[0]):
        if cir[2] <= min_r:
            min_r = cir[2]
            idx = i

    initial_chamber = houghcircles[0][idx]
    initial_chamber[0] += x_start
    initial_chamber[1] += y_start

    return initial_chamber
