import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from feat_recognize.initial_chamber import detect_initial_chamber
from feat_recognize.volution_counter import VolutionCounter
from feat_recognize.utils import resize_img

from common.args import feat_recog_args


def recognize_feature(img_path: str) -> tuple:
    img_rgb = cv2.imread(img_path)
    # Opening preprocess and resize
    kernel = np.ones((3, 3), np.int8)
    img_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, kernel)
    # w, h = (896, 448)
    # img_rgb = cv2.resize(img_rgb, (w, h))
    img_rgb = resize_img(img_rgb)
    h, w = img_rgb.shape[:2]

    # Detect volutions and detect initial chamber with a high confidence level
    volution_counter = VolutionCounter(feat_recog_args)
    volutions_dict, thickness_dict, success_initial_chamber = volution_counter.count_volutions(img_path)

    if success_initial_chamber:
        initial_chamber = detect_initial_chamber(img_rgb)
    else:  # Detect initial chamber with a low confidence level
        inner_volution_above = volutions_dict[1]
        min_x_above = inner_volution_above[0][0]
        max_x_above = inner_volution_above[-1][0]
        min_y = int(min(y for x, y in inner_volution_above) * h)

        inner_volution_below = volutions_dict[-1]
        min_x_below = inner_volution_below[0][0]
        max_x_below = inner_volution_below[-1][0]
        max_y = int(max(y for x, y in inner_volution_below) * h)

        min_x = int(min(min_x_above, min_x_below) * w)
        max_x = int(max(max_x_above, max_x_below) * w)

        sub_img = img_rgb[min_y:max_y, min_x:max_x, :]
        sub_img_pos = (min_x, min_y)

        # Try again with a lower confidence level
        initial_chamber = detect_initial_chamber(
            img_rgb,
            sub_img=sub_img,
            sub_img_pos=sub_img_pos,
            param2=feat_recog_args.houghcircle_params["param2"] * 0.2,
        )

    if initial_chamber is None:  # failed
        initial_chamber = [0, 0, 0]

    return volutions_dict, thickness_dict, initial_chamber


def main():
    img_path_root = "./common/images"  # path to the images
    options = [
        "Rugosofusulina_jurmatensis_1_2.png",
        "Fusulina_huntensis_1_3.png",
        "Fusulinella_devexa_1_11.png",
        "Fusulinella_famula_1_1.png",
        "Pseudofusulina_modesta_1_1.png",
        "Fusulinella_clarki_1_1.png",
        "Fusulinella_clarki_1_6.png",
        "Fusulina_boonensis_1_1.png",
        "Fusulinella_cabezasensis_1_5.png",
        "Chusenella_absidata_1_2.png",
        "Rugosofusulina_yingebulakensis_1_1.png",
        "Chusenella_leei_1_3.png",
    ]
    for i in range(len(options)):
        img_path = f"{img_path_root}/{options[i]}"

        volutions_dict, thickness_dict, initial_chamber = recognize_feature(img_path)

        img_rgb = cv2.imread(img_path)
        kernel = np.ones((3, 3), np.int8)
        img_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, kernel)
        img_rgb = resize_img(img_rgb)
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 2)
        kernel = np.ones((5, 5), np.int8)
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, kernel)
        img_show = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        h, w, c = img_show.shape
        os.makedirs("vol_count_result", exist_ok=True)
        cv2.circle(img_show, initial_chamber[:2], initial_chamber[2], (0, 255, 0), 2)

        for idx, volution in volutions_dict.items():
            for point in volution:
                x, y = point
                x = int(x * w)
                y = int(y * h)
                cv2.circle(img_show, (x, y), 1, (255, 0, 0), 2)
        plt.imsave(f"vol_count_result/{i}.png", img_show)


if __name__ == "__main__":
    main()
