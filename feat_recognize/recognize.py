import cv2
import numpy as np
from feat_recognize.initial_chamber import detect_initial_chamber
from feat_recognize.volution_counter import VolutionCounter

from common.args import feat_recog_args


def recognize_feature(img_path: str) -> tuple:
    img_rgb = cv2.imread(img_path)
    # Opening preprocess and resize
    kernel = np.ones((3, 3), np.int8)
    img_rgb = cv2.morphologyEx(img_rgb, cv2.MORPH_OPEN, kernel)
    img_rgb = cv2.resize(img_rgb, (896, 448))

    # Detect volutions
    volution_counter = VolutionCounter(feat_recog_args)
    volutions, thickness_per_vol, success_initial_chamber = volution_counter.count_volutions(img_rgb)

    # Detect initial chamber
    if success_initial_chamber:
        initial_chamber = detect_initial_chamber(img_rgb)
    else:
        inner_volution_above = volutions[0][-1]
        min_x_above = inner_volution_above[0][0]
        max_x_above = inner_volution_above[-1][0]
        min_y = min(y for x, y in inner_volution_above)

        inner_volution_below = volutions[1][-1]
        min_x_below = inner_volution_below[0][0]
        max_x_below = inner_volution_below[-1][0]
        max_y = max(y for x, y in inner_volution_below)

        min_x = min(min_x_above, min_x_below)
        max_x = max(max_x_above, max_x_below)
        sub_img = img_rgb[min_y:max_y, min_x:max_x, :]
        sub_img_pos = (min_x, min_y)
        # Try again with a lower confidence level
        initial_chamber = detect_initial_chamber(
            img_rgb, sub_img=sub_img, sub_img_pos=sub_img_pos, param2=feat_recog_args.houghcircle_params["param2"] * 0.4
        )

    if initial_chamber is None:  # failed
        initial_chamber = [0, 0, 0]

    return volutions, thickness_per_vol, initial_chamber


def main():
    img_path_root = "D:/Repositories/foscap/common/images"  # path to the images
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

        volutions, thickness_per_vol, initial_chamber = recognize_feature(img_path)
        print(initial_chamber)


if __name__ == "__main__":
    main()
