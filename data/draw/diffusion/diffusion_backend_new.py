import concurrent.futures
import os, json, sys
import numpy as np
from typing import Any, Sized
import matplotlib.pyplot as plt
import matplotlib.patches as pch
from PIL import Image, ImageDraw, ImageFilter
import random
import cv2
from io import BytesIO

# from shape_filter import getMostSimilarImages
from tqdm import tqdm
import re
import argparse

# from concurrent.futures import ThreadPoolExecutor
# import concurrent


class Figure_Engine:
    # No longer randomization is available: this will be automatically on.
    # xkcd will be on.
    def __init__(
        self,
        size: "tuple[float, float]" = (12.8, 12.8),
        dpi: int = 100,
        max_volution: int = 0,
        center: tuple = (0.5, 0.5),
    ) -> None:
        self.image = plt.figure(figsize=size, dpi=dpi)
        self.shape = (int(size[0] * dpi), int(size[1] * dpi))
        self.ax = self.image.add_subplot()
        plt.subplots_adjust(0, 0, 1, 1)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.spines["top"].set_visible(False)
        self.ax.spines["right"].set_visible(False)
        self.ax.spines["bottom"].set_visible(False)
        self.ax.spines["left"].set_visible(False)
        self.volution_memory = {}
        self.max_volution = max_volution
        self.center = center

    def draw(self, shape, width=None, color=None, trans=None):
        width, color, trans = self.__get_essential_info(shape, width, color, trans)
        try:
            index = self.__special_info_validator(shape["special_info"])
        except:
            index = None
        assert index == None or (-1 <= index and index <= 20)
        if shape["type"] == "fusiform_1":
            x_offset = shape["x_offset"]
            y_offset = shape["y_offset"]
            fc = shape["focal_length"]
            eps, ome, phi = shape["sin_params"]
            x_start = shape["x_start"]
            x_end = shape["x_end"]
            y_sim = shape["y_symmetric_axis"]

            self.__handle_fusiform_1(
                fc, x_offset, y_offset, eps, ome, phi, x_start, x_end, y_sim, color, trans, width, index
            )

        elif shape["type"] == "fusiform_2":
            x_offset = shape["x_offset"]
            y_offset = shape["y_offset"]
            fc = shape["focal_length"]
            eps, ome, phi = shape["sin_params"]
            power = shape["power"]
            x_start = shape["x_start"]
            x_end = shape["x_end"]

            self.__handle_fusiform_2(
                focal_length=fc,
                x_offset=x_offset,
                y_offset=y_offset,
                power=power,
                eps=eps,
                omega=ome,
                phi=phi,
                x_start=x_start,
                x_end=x_end,
                color=color,
                trans=trans,
                line_width=width,
                index=index,
            )

        elif shape["type"] == "ellipse":
            ellipse_x, ellipse_y = shape["center"]
            major = shape["major_axis"]
            minor = shape["minor_axis"]
            alpha = shape["rotation"] * 180 / np.pi

            self.__handle_ellipse(ellipse_x, ellipse_y, major, minor, alpha, width, color, trans, index)

        elif shape["type"] == "polygon":
            points: list = shape["points"]
            assert len(points) >= 3, "There should be more than 3 points within a polygon."

            self.__handle_polygon(points, width, color, trans)

        elif shape["type"] == "spindle":
            center_x, center_y = shape["center"]
            major = shape["major_axis"]
            minor = shape["minor_axis"]

            self.__handle_spindle(center_x, center_y, major, minor, width, color, index)

        elif shape["type"] == "curves":
            curves = shape["control_points"]
            if isinstance(curves[0][0], float):
                curves = [curves]
            curve_buf_x = []
            curve_buf_y = []
            for curve in curves:
                x, y = self.__handle_curve(curve, width, color, index)
                curve_buf_x.append(x)
                curve_buf_y.append(y)

            curve_buf_x = [item for x in curve_buf_x for item in x]
            curve_buf_y = [item for y in curve_buf_y for item in y]
            """
                debug_fig = plt.figure()
                ax = debug_fig.add_subplot().scatter(curve_buf_x, curve_buf_y, s=1, marker='x')
                debug_fig.savefig(f"scatter_points{index}.png")
                #"""
            self.__keep_memory(index, curve_buf_x, curve_buf_y)

    def __handle_ellipse(
        self,
        ellipse_x: float,
        ellipse_y: float,
        major: float,
        minor: float,
        alpha: float,
        line_width: int,
        color: Any,
        transparency: tuple = (0, 0, 0, 0),
        index=None,
    ):
        if major < minor:
            raise ValueError("The major axis is smaller than the minor axis, which is incorrect.")
        self.ax.add_patch(
            pch.Ellipse(
                (ellipse_x, ellipse_y),
                major,
                minor,
                angle=alpha,
                edgecolor=color,
                facecolor=transparency,
                linewidth=line_width * (self.shape[0] / 640),
            )
        )
        # Proloculus version
        self.__keep_memory(index, ellipse_x, ellipse_y)

    def __handle_polygon(self, points: list, line_width: int, color: Any, trans: tuple = (0, 0, 0, 0)):
        self.ax.add_patch(
            pch.Polygon(
                points,
                closed=True,
                edgecolor=color,
                linewidth=line_width * (self.shape[0] / 640),
                facecolor=trans,
            )
        )

    def __handle_spindle(
        self,
        center_x: float,
        center_y: float,
        major_axis: float,
        minor_axis: float,
        line_width: int,
        color: Any,
        index=None,
    ):
        theta = np.arange(0, 2 * 3.1416, 0.01)
        a = major_axis / 2
        b = minor_axis / 2

        rho = np.sqrt(1 / (np.cos(theta) ** 2 / a**2 + np.sin(theta) ** 2 / b**2))
        rho1 = np.sqrt(np.abs((a / 10) ** 2 * np.sin(2 * (theta + 1.5708))))
        rho2 = np.sqrt(np.abs((a / 10) ** 2 * np.sin(2 * theta)))
        rho_list = rho - rho1 - rho2  # shift on pi/4s.

        rho_ru = rho_list[np.where((theta < 3.1416 * 0.35))]
        theta_ru = theta[np.where((theta < 3.1416 * 0.35))]
        x_ru = (rho_ru) * np.cos(theta_ru) + center_x
        y_ru = (rho_ru) * np.sin(theta_ru) + center_y

        rho_rd = rho_list[np.where((theta > 3.1416 * 1.65))]
        theta_rd = theta[np.where((theta > 3.1416 * 1.65))]
        x_rd = (rho_rd) * np.cos(theta_rd) + center_x
        y_rd = (rho_rd) * np.sin(theta_rd) + center_y

        rho_l = rho_list[np.where((theta > 3.1416 * 0.65) & (theta < 3.1416 * 1.35))]
        theta_l = theta[np.where((theta > 3.1416 * 0.65) & (theta < 3.1416 * 1.35))]
        x_l = (rho_l) * np.cos(theta_l) + center_x
        y_l = (rho_l) * np.sin(theta_l) + center_y

        x_mu = np.linspace(x_ru[-1], x_l[0], num=5)
        y_mu = np.linspace(y_ru[-1], y_l[0], num=5)

        x_md = np.linspace(x_l[-1], x_rd[0], num=5)
        y_md = np.linspace(y_l[-1], y_rd[0], num=5)

        x = np.concat((x_ru, x_mu, x_l, x_md, x_rd), axis=None)
        y = np.concat((y_ru, y_mu, y_l, y_md, y_rd), axis=None)

        self.ax.plot(x, y, color=color, linewidth=line_width * (self.shape[0] / 640))

        self.__keep_memory(index, x, y)

    def __handle_fusiform_1(
        self,
        focal_length,
        x_offset,
        y_offset,
        eps,
        omega,
        phi,
        x_start,
        x_end,
        y_sim,
        color,
        trans,
        line_width,
        index,
    ):
        def f(x):
            return 4 * focal_length * (x - x_offset) ** 2 + y_offset + eps * np.sin(omega * x + phi)

        x = np.linspace(x_start, x_end, 1000)
        y1 = f(x)
        y2 = 2 * y_sim - y1

        for i in range(len(x)):
            self.ax.plot((x[i], x[i]), (y1[i], y2[i]), linewidth=1, color=trans)

        self.ax.plot(x, y1, x, y2, linewidth=line_width * (self.shape[0] / 640), color=color)

        self.__keep_memory(index, np.concatenate([x, x]), np.concatenate([y1, y2]))

    def __handle_fusiform_2(
        self,
        focal_length,
        x_offset,
        y_offset,
        power,
        eps,
        omega,
        phi,
        x_start,
        x_end,
        color,
        trans,
        line_width,
        index,
    ):
        x = np.linspace(x_start, x_end, 1000)
        x_left = x[:500]
        sin_wave = eps * np.sin(omega * (x - x_start) + phi)
        y_left = (np.abs(x_left - x_offset) / (4 * focal_length)) ** (1 / power) + y_offset
        y_right = np.flip(y_left)  # 得到开口向左的上半部分
        y1 = np.concatenate([y_left, y_right]) + sin_wave
        y2 = 2 * y_offset - y1  # 得到整个纺锤形的下半部分
        # fix the side-wise crack
        x_start_lst = np.array([x_start for _ in range(50)])
        x_end_lst = np.array([x_end for _ in range(50)])
        y_start_lst = np.linspace(y1[0], y2[0], 50)
        y_end_lst = np.linspace(y1[-1], y2[-1], 50)
        for i in range(len(x)):
            self.ax.plot(
                (x[i], x[i]), (y1[i], y2[i]), linewidth=line_width * (self.shape[0] / 640), color=trans
            )
        self.ax.plot(
            x,
            y1,
            x,
            y2,
            x_start_lst,
            y_start_lst,
            x_end_lst,
            y_end_lst,
            linewidth=line_width * (self.shape[0] / 640),
            color=color,
        )

        self.__keep_memory(
            index,
            np.concatenate([x, x, x_start_lst, x_end_lst]),
            np.concatenate([y1, y2, y_start_lst, y_end_lst]),
        )

    def __handle_curve(self, control_points, width: int = 5, color=(0, 0, 0, 1), index=None):
        curve_points = []
        t_values = np.linspace(0, 1, 600)
        for t in t_values:
            one_minus_t = 1 - t
            point = (
                one_minus_t**3 * np.array(control_points[0])
                + 3 * one_minus_t**2 * t * np.array(control_points[1])
                + 3 * one_minus_t * t**2 * np.array(control_points[2])
                + t**3 * np.array(control_points[3])
            )
            curve_points.append(tuple(point))
        curve_points = np.array(curve_points)
        self.ax.plot(
            curve_points[:, 0], curve_points[:, 1], linewidth=width * (self.shape[0] / 640), color=color
        )

        return curve_points[:, 0], curve_points[:, 1]

    def __get_essential_info(self, shape, width, color, trans):
        if width is None:
            width = self.__get_width(shape)
        if color is None:
            color = self.__get_color(shape)
        color = np.clip(color, 0, 1)
        if trans is None:
            trans = self.__get_transparency(shape)
        return width, color, trans

    def __get_width(self, shape):
        try:
            return shape["width"]
        except:
            return 3

    def __get_color(self, shape):
        try:
            return shape["color"]
        except:
            return (random.random(), random.random(), random.random())

    def __get_transparency(self, shape):
        try:
            if shape["fill_mode"] == "no":
                trans = (0, 0, 0, 0)
            elif shape["fill_mode"] == "white":
                trans = (1, 1, 1, 1)
            elif shape["fill_mode"] == "black":
                trans = (0, 0, 0, 1)
        except:
            trans = (0, 0, 0, 0)  # no
        return trans

    def transfer_to_cv2_wrapper(self):
        buf2 = BytesIO()
        self.image.savefig(buf2, transparent=True, format="png")
        buf2.seek(0)

        query_img_array = np.frombuffer(buf2.getvalue(), dtype=np.uint8)
        query_img = cv2.imdecode(query_img_array, cv2.IMREAD_UNCHANGED)

        return query_img

    def transfer_to_cv2(self):
        buf = BytesIO()
        # buf2=BytesIO()
        self.image.savefig(buf, format="png")
        # self.image.savefig(buf2,transparent=True,format="png")
        buf.seek(0)
        # buf2.seek(0)

        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, 1)
        # query_img_array=np.frombuffer(buf2.getvalue(),dtype=np.uint8)
        # query_img=cv2.imdecode(query_img_array,cv2.IMREAD_UNCHANGED)

        return img

    def __special_info_validator(self, special_info):
        for i in range(20):  # How a fusi has volutions over 20 layers?
            if special_info == f"volution {i}":
                return i
            if special_info == "initial chamber":
                return -1
        # print(special_info)
        return None

    def __keep_memory(self, index, x, y):
        assert (isinstance(x, float) and isinstance(y, float)) or (len(x) == len(y))  # type: ignore
        # Suppose the center is the same
        if index != None and index >= 0:
            angle_value = []
            for x0, y0 in zip(x, y):  # type: ignore
                x0 -= self.center[0]
                y0 -= self.center[1]
                if x0 == 0:
                    if y0 > 0:
                        angle_value.append(np.pi / 2)
                    else:
                        angle_value.append(3 * np.pi / 2)
                else:
                    if x0 > 0 and y0 >= 0:
                        this_angle = np.arctan(y0 / x0)
                    elif x0 > 0 and y0 < 0:
                        this_angle = 2 * np.pi + np.arctan(y0 / x0)
                    elif x0 < 0:
                        this_angle = np.pi + np.arctan(y0 / x0)
                    angle_value.append(this_angle)
            angle_value = np.array(angle_value)
            indices = np.argsort(angle_value)
            x = np.array(x)[indices]
            y = np.array(y)[indices]
            angle_value = angle_value[indices]
            self.volution_memory[index] = (x, y, angle_value)
        if index == -1:
            self.volution_memory[index] = self.center

    def close(self):
        plt.close(self.image)


def generate_basic_shape_wrapper(data_dict):
    shapes = data_dict["shapes"]
    ni = data_dict["ni"]
    figure = Figure_Engine(max_volution=int(ni["num_volutions"]), center=ni["center"])
    fixed_color = np.random.randn(3) / 6 + np.array((0.1, 0.1, 0.1))
    volution_max = {}
    filtered_shapes = []

    def get_volution_index(shape):
        a = shape["special_info"]
        return int(a[len("volution ") :])

    for shape in shapes:
        if re.match("volution [0-9]+", shape["special_info"]) is not None:
            if volution_max == {}:
                volution_max = shape
            else:
                if get_volution_index(shape) > get_volution_index(volution_max):
                    volution_max = shape
    for shape in shapes:
        if shape["special_info"] == volution_max["special_info"]:
            filtered_shapes.append(shape)
    for shape in filtered_shapes:
        figure.draw(shape)
    query_img = figure.transfer_to_cv2_wrapper()
    figure.close()
    return query_img


def generate_basic_shape(shapes: list, ni: dict) -> tuple:
    figure = Figure_Engine(max_volution=int(ni["num_volutions"]), center=ni["center"])
    fixed_color = np.random.randn(3) / 6 + np.array((0.1, 0.1, 0.1))
    for shape in shapes:
        figure.draw(shape)
    basic_img = figure.transfer_to_cv2()
    figure.close()
    return basic_img, figure.volution_memory, figure.max_volution


def generate_basic_mask(volution_memory: dict, filling: list, debug=None) -> np.ndarray:
    from bisect import bisect

    mask = plt.figure(figsize=(12.8, 12.8), dpi=100)
    ax = mask.add_subplot()
    plt.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # volution_memory structure:
    # -1 -> initial chamber, (x,y) as the middle point
    # other numbers -> volutions, (x, y, angle) as the points of the volutions, sorted by angle[0:2pi~6.28]
    for fill in filling:  # experiment suggests that diffusing these together works better
        start_angle = fill["start_angle"]
        if start_angle < 0:
            start_angle += 2 * np.pi
        end_angle = fill["end_angle"]
        if end_angle < 0:
            end_angle += 2 * np.pi
        start_volution = volution_memory[fill["start_volution"]]  # shape: x,y,a
        end_volution = volution_memory[fill["end_volution"]]
        if start_angle < end_angle:
            angles = np.linspace(start_angle, end_angle, 1200)
        else:  # start_angle > end_angle, ignore ==
            angles1 = np.linspace(start_angle, np.pi * 2, 600)
            angles2 = np.linspace(0, end_angle, 600)
            angles = np.concatenate([angles1, angles2])
        for angle in angles:
            if angle < 0:
                angle += 2 * np.pi
            elif angle > 2 * np.pi:
                angle -= 2 * np.pi
            index_at_start = bisect(start_volution[2], angle)
            index_at_end = bisect(end_volution[2], angle)
            try:
                ax.plot(
                    [start_volution[0][index_at_start], end_volution[0][index_at_end]],  # x
                    [start_volution[1][index_at_start], end_volution[1][index_at_end]],  # y
                    color="black",
                    linewidth=5,
                )
            except IndexError:
                ax.plot(
                    [start_volution[0][index_at_start - 1], end_volution[0][index_at_end - 1]],  # x
                    [start_volution[1][index_at_start - 1], end_volution[1][index_at_end - 1]],  # y
                    color="black",
                    linewidth=5,
                )
    if debug != None:
        mask.savefig(f"{debug}_Mask_original.png")
    buf = BytesIO()
    mask.savefig(buf, format="png")
    buf.seek(0)

    img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img_array, 1)

    plt.close(mask)

    return img


def get_hu_moments(img: np.ndarray):
    edges = cv2.Canny(img, 75, 140)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moments = cv2.moments(contours[0])
    hu_moments = cv2.HuMoments(moments)  # Hu Moments for current picture
    return hu_moments


# def get_best_match_wrapper(data_dict):
#    return getMostSimilarImages(data_dict["pool_prefix"],data_dict["img"],data_dict["bbox"],n=16,max_sample=99999,batchsize=128,debug=False,multi_process=False,tmp_dir=data_dict["tmp_dir"])[0][0]

# def get_best_match(img, pool_prefix:str, bbox, tmp_dir):
# hu1 = get_hu_moments(img)
# picture_hus = [get_hu_moments(cv2.imread(filename)) for filename in ref_paths]
# distances = [hu1**2 + hu**2 for hu in picture_hus] # Euclid
# minimum_index = np.argmin(distances)
# return cv2.imread(ref_paths[minimum_index])
#    return getMostSimilarImages(pool_prefix,img,bbox,n=16,max_sample=99999,batchsize=128,debug=False,multi_process=True, tmp_dir=tmp_dir)[0][0]


def get_random_match(dir: str):
    files = [os.path.join(dir, file) for file in os.listdir(dir)]
    lucky_guy = random.choice(files)
    return lucky_guy


def diffuse(
    img: np.ndarray,
    mask: np.ndarray,
    best_ref_poles: str,
    ref_path: str,
    num_refs: int,
    mode: str,
    debug=None,
) -> np.ndarray:

    def get_random_ref_image(ref_path: list) -> np.ndarray:
        ref_image = cv2.imread(random.choice(ref_path))
        return ref_image

    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img = img.convert("L")
    img = img.convert("RGB")
    img = np.asarray(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    mask = Image.fromarray(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    mask = mask.convert("L")
    mask = np.asarray(mask)
    mask = 255 - mask
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)

    if mode == "axial":
        ref_paths = [os.path.join(ref_path, f"ref_axial_{x:08}.png") for x in range(num_refs)]
        ref_image = get_random_ref_image(ref_paths)
    elif mode == "poles":
        # ref_paths = [os.path.join(ref_path, f"ref_{x:08}.png") for x in range(num_refs)]
        # ref_image = get_best_match(img, ref_paths)
        ref_image = cv2.imread(best_ref_poles, cv2.IMREAD_UNCHANGED)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGBA2BGR)

    if debug != None:
        cv2.imwrite(f"{debug}_Image.png", img)
        cv2.imwrite(f"{debug}_Reference.png", ref_image)

    synthesis, depth_pred = inference_single_image(
        ref_image.copy(), img.copy(), mask.copy(), ddim_steps=120, scale=10, seed=0, enable_shape_control=True
    )

    synthesis = crop_padding_and_resize(img, synthesis)
    if debug != None:
        cv2.imwrite(f"{debug}_SYNTH_1.png", synthesis)

    return synthesis.astype(np.uint8)


def generate_septa(septas: list, debug=None) -> np.ndarray:
    figure = Figure_Engine()
    fixed_color = np.random.randn(3) / 6 + np.array((0.1, 0.1, 0.1))
    for septa in septas:
        figure.draw(septa, color=fixed_color)
    cv2_fig = figure.transfer_to_cv2()
    if debug != None:
        cv2.imwrite(f"test_septa_xkcd.jpg", cv2_fig)
    cv2_fig = Image.fromarray(cv2.cvtColor(cv2_fig, cv2.COLOR_BGR2RGB))
    cv2_fig = cv2_fig.convert("L")
    cv2_fig = cv2_fig.convert("RGB")
    cv2_fig = np.asarray(cv2_fig)
    cv2_fig = cv2.cvtColor(cv2_fig, cv2.COLOR_RGB2BGR)
    figure.close()
    return cv2_fig


def generate_one_img(
    idx, sample, img_path: str, ref_path: str, ref_poles_pool: str, num_refs: int, keyword, best_ref=None
):
    debug_folder = keyword
    if not os.path.exists(f"{debug_folder}/DEBUG"):
        os.makedirs(f"{debug_folder}/DEBUG", exist_ok=True)
    basic_img, volution_memory, max_volution = generate_basic_shape(
        sample["shapes"], sample["numerical_info"]
    )
    if best_ref is None:
        try:
            # best_ref_poles=get_best_match(query_img,ref_poles_pool,sample["numerical_info"]["fossil_bbox"])
            # print(best_ref_poles)
            best_ref_poles = "pics/Neoschwagerina_takagamiensis_1_2.png"
        except Exception as e:
            print(e)
            best_ref_poles = "pics/Neoschwagerina_takagamiensis_1_2.png"
    else:
        best_ref_poles = best_ref
    basic_mask = generate_basic_mask(volution_memory, sample["axial_filling"])
    # diffused_basic_img = diffuse(basic_img, basic_mask, best_ref_poles, ref_path, num_refs, mode = 'axial',debug=f'{debug_folder}/DEBUG/{idx}_pre')
    diffused_basic_img = diffuse(
        basic_img, basic_mask, best_ref_poles, ref_path, num_refs, mode="axial", debug=None
    )
    # septa_overlayer = generate_septa(sample["septa_folds"])
    # diffused_basic_img = np.minimum(diffused_basic_img,septa_overlayer)
    poles_mask = generate_basic_mask(volution_memory, sample["poles_folds"], debug=None)
    # diffused_img = diffuse(diffused_basic_img, poles_mask, best_ref_poles, ref_path, num_refs, mode = 'poles',debug=f'{debug_folder}/DEBUG/{idx}_post')
    diffused_img = diffuse(
        diffused_basic_img, poles_mask, best_ref_poles, ref_path, num_refs, mode="poles", debug=None
    )
    septa_overlayer = generate_septa(sample["septa_folds"])
    blended_img = np.minimum(diffused_img, septa_overlayer)
    # blended_img = diffused_img
    blended_img = cv2.cvtColor(blended_img, cv2.COLOR_BGR2GRAY)
    img_path = f"{keyword}/{img_path}"
    if not os.path.exists(f"{keyword}"):
        os.mkdir(f"{keyword}")
    cv2.imwrite(img_path, blended_img)


# def process_single(f, idx_sample: tuple, vars):
#     generate_one_img(
#         idx_sample[0],
#         idx_sample[1],
#         os.path.join(
#             "fos_data/figure-4",
#             f"{idx_sample[0]:08d}.jpg",
#         ),
#         ref_path="fos_data/reference",
#         ref_poles_pool="pics/",
#         num_refs=10,  # some number
#         keyword="4"
#     ) # stand by for repackage
def generate_one_img_wrapper(wrapped):
    print(f"Running pic {wrapped['output_file']}")
    generate_one_img(
        wrapped["idx_sample"],
        wrapped["sample"],
        wrapped["output_file"],
        ref_path="fos_data/reference",
        ref_poles_pool="pics/",
        num_refs=10,  # some number
        keyword=wrapped["kwd"],  # "100k_finetune_internvl_part2"
        best_ref=wrapped["best_ref"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rules", type=str)
    parser.add_argument("--best_match", type=str)
    parser.add_argument("--start_pos", type=int, default=0)
    parser.add_argument("--end_pos", type=int, default=None)
    parser.add_argument("--kwd", type=str)
    args = parser.parse_args()

    with open(args.rules, "r") as f:
        # "/home/nfs04/xingsy/geocap/dataset/rules_stage2_100k_part2.json"
        samples = json.load(f)
        # random.shuffle(samples)
        # samples=samples[:10]
        assert isinstance(samples, list)
    best_ref_list = []
    with open(args.best_match, "r") as f:
        # "stage2_100k_bestmatch_part2.json"
        for line in f:
            line = line.strip()
            best_ref_list.append(line)
    if args.end_pos is not None:
        samples = samples[args.start_pos : args.end_pos]
        best_ref_list = best_ref_list[args.start_pos : args.end_pos]
    else:
        samples = samples[args.start_pos :]
        best_ref_list = best_ref_list[args.start_pos :]

    for idx_sample, sample in enumerate(tqdm(samples)):
        # if idx_sample not in [7,8,9]:
        #     continue
        generate_one_img(
            idx_sample + args.start_pos,
            sample,
            f"{idx_sample+args.start_pos:08d}.jpg",
            ref_path="fos_data/reference",
            ref_poles_pool="pics/",
            num_refs=10,  # some number
            keyword=args.kwd,  # "100k_finetune_internvl_part2"
            best_ref=best_ref_list[idx_sample],
        )


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1,3,4"
    from run_gradio3_demo import inference_single_image, crop_padding_and_resize

    np.random.seed(0)
    random.seed(0)
    main()
