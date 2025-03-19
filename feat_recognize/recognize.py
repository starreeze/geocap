import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common.args import feat_recog_args
from feat_recognize.chomata_scan import chomatas_scan
from feat_recognize.initial_chamber import ProloculusDetector
from feat_recognize.utils import calculate_angle, resize_img
from feat_recognize.volution_counter import VolutionCounter


def recognize_feature(img_path: str) -> tuple:
    img_rgb = cv2.imread(img_path)
    orig_h, orig_w = img_rgb.shape[:2]
    # Opening preprocess and resize
    kernel = np.ones((3, 3), np.int8)
    img_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, kernel)

    img_rgb = resize_img(img_rgb)
    h, w = img_rgb.shape[:2]

    # Detect initial chamber
    proloculus_detector = ProloculusDetector()
    initial_chamber = proloculus_detector.detect_initial_chamber(img_path, threshold=0.25)
    if initial_chamber is None:  # retry with lower threshold
        initial_chamber = proloculus_detector.detect_initial_chamber(img_path, threshold=0.1)
    center = tuple(initial_chamber[:2]) if initial_chamber is not None else (orig_w // 2, orig_h // 2)

    # Detect volutions
    volution_counter = VolutionCounter(feat_recog_args)
    volutions_dict, thickness_dict = volution_counter.count_volutions(img_path, center)

    # Convert to absolute coordinates
    for idx, volution in volutions_dict.items():
        for i, point in enumerate(volution):
            volution[i] = (int(point[0] * orig_w), int(point[1] * orig_h))
        # Remove duplicate x values by keeping only one point per x coordinate
        unique_x_points = {}
        for point in volution:
            x = point[0]
            if x not in unique_x_points:
                unique_x_points[x] = point
        volution = list(unique_x_points.values())
        volutions_dict[idx] = volution

    # Detect chomata
    chomata_result = chomatas_scan(volutions_dict, img_path)
    tunnel_angles = calculate_tunnel_angles(chomata_result, center)
    tunnel_angles = dict(sorted(tunnel_angles.items(), key=lambda x: x[0]))

    return volutions_dict, thickness_dict, initial_chamber, tunnel_angles


def calculate_tunnel_angles(chomata_result, center: tuple[int, int]) -> dict[int, float]:
    tunnel_angles = {}
    for idx, chomata_pos in chomata_result.items():
        if len(chomata_pos) == 2:  # successfully detect 2 chomatas in a voludion
            tunnel_angle = calculate_angle(center, chomata_pos[0], chomata_pos[1])
            if tunnel_angle is None or tunnel_angle < 5:
                continue
            if abs(idx) not in tunnel_angles:
                tunnel_angles[abs(idx)] = int(tunnel_angle)
            else:  # average of upper and lower
                tunnel_angles[abs(idx)] += int(tunnel_angle)
                tunnel_angles[abs(idx)] = int(tunnel_angles[abs(idx)] / 2)

    return tunnel_angles


def main():
    img_path_root = "./dataset/common/images"  # path to the images dir
    options = [
        # "Rugosofusulina_jurmatensis_1_2.png",
        # "Fusulina_huntensis_1_3.png",
        # "Fusulinella_devexa_1_11.png",
        # "Fusulinella_famula_1_1.png",
        # "Pseudofusulina_modesta_1_1.png",
        # "Fusulinella_clarki_1_1.png",
        # "Fusulinella_clarki_1_6.png",
        # "Fusulina_boonensis_1_1.png",
        # "Fusulinella_cabezasensis_1_5.png",
        # "Chusenella_absidata_1_2.png",
        # "Rugosofusulina_yingebulakensis_1_1.png",
        # "Chusenella_leei_1_3.png",
        "Pseudoschwagerina_broggii_1_1.png"
    ]
    for i in range(len(options)):
        img_path = f"{img_path_root}/{options[i]}"

        volutions_dict, thickness_dict, initial_chamber, tunnel_angles = recognize_feature(img_path)

        img_rgb = cv2.imread(img_path)
        kernel = np.ones((3, 3), np.int8)
        img_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, kernel)
        # img_rgb = resize_img(img_rgb)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)
        kernel = np.ones((5, 5), np.int8)
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
        img_show = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        h, w, c = img_show.shape
        os.makedirs("vol_count_result", exist_ok=True)

        if initial_chamber is not None:
            cv2.circle(img_show, initial_chamber[:2], initial_chamber[2], (0, 255, 0), 2)

        for idx, volution in volutions_dict.items():
            for point in volution:
                x, y = point
                # x = int(x * w)
                # y = int(y * h)
                cv2.circle(img_show, (x, y), 1, (255, 0, 0), 2)
        plt.imsave(f"vol_count_result/{i}.png", img_show)
        # plt.imshow(img_show)
        # plt.show()


if __name__ == "__main__":
    main()
