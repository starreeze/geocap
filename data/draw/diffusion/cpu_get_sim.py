from shape_filter import size_, getMostSimilarImages
import json
from iterwrap import iterate_wrapper
from multiprocessing import Pool, Pipe, cpu_count
from tqdm import tqdm
import numpy as np
import re
import cv2
import argparse
import os


def convertPos(X, Y):
    if type(X) == list and type(Y) == list:
        return [int(x * size_[0]) for x in X], [int((1 - y) * size_[0]) for y in Y]
    elif type(X) == np.ndarray and type(Y) == np.ndarray:
        return (X * size_[0]).astype(int), ((1 - Y) * size_[0]).astype(int)  # type: ignore
    elif (type(X) == float and type(Y) == float) or ((type(X) == np.float64 and type(Y) == np.float64)):
        return int(size_[0] * X), int(size_[0] * (1 - Y))  # type: ignore


def convertLength(W):
    if type(W) == float:
        return int(W * size_[0])


def wrapper_retard(data_dicts, _in_pipe):
    ret = []
    for data_dict in data_dicts:
        ret.append(generate_basic_shape_wrapper(data_dict))
        _in_pipe.send(1)
    return ret


def draw(cv2img, shape):
    def __handle_ellipse(
        cv2img, ellipse_x: float, ellipse_y: float, major: float, minor: float, alpha: float
    ):
        if major < minor:
            raise ValueError("The major axis is smaller than the minor axis, which is incorrect.")
        ellipse_x, ellipse_y = convertPos(ellipse_x, ellipse_y)  # type: ignore
        major = convertLength(major)  # type: ignore
        minor = convertLength(minor)  # type: ignore
        cv2.ellipse(cv2img, (ellipse_x, ellipse_y), (major, minor), alpha, 0, 360, color=(255, 255, 255, 255))  # type: ignore

    def __handle_polygon(cv2img, points: list):
        points = [convertPos(p[0], p[1]) for p in points]
        cv2.polylines(cv2img, points, True, color=(255, 255, 255, 255))  # type: ignore

    def __handle_spindle(cv2img, center_x: float, center_y: float, major_axis: float, minor_axis: float):
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

        x, y = convertPos(x, y)  # type: ignore

        cv2img[y, x] = [255, 255, 255, 255]

    def __handle_fusiform_1(cv2img, focal_length, x_offset, y_offset, eps, omega, phi, x_start, x_end, y_sim):
        def f(x):
            return 4 * focal_length * (x - x_offset) ** 2 + y_offset + eps * np.sin(omega * x + phi)

        x = np.linspace(x_start, x_end, 1000)
        y1 = f(x)
        y2 = 2 * y_sim - y1

        # for i in range(len(x)):
        #     cv2.line(cv2img,convertPos(x[i],x[i]),convertPos(y1[i],y2[i]),color=255) # type: ignore

        for idx in range(len(x)):
            if idx != len(x) - 1:
                cv2.line(cv2img, convertPos(x[idx], y1[idx]), convertPos(x[idx + 1], y1[idx + 1]), color=(255, 255, 255, 255))  # type: ignore
                cv2.line(cv2img, convertPos(x[idx], y2[idx]), convertPos(x[idx + 1], y2[idx + 1]), color=(255, 255, 255, 255))  # type: ignore

    def __handle_fusiform_2(cv2img, focal_length, x_offset, y_offset, power, eps, omega, phi, x_start, x_end):
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
        # for i in range(len(x)):
        #     cv2.line(cv2img,convertPos(x[i],x[i]),convertPos(y1[i],y2[i]),color=255) # type: ignore
        for idx in range(len(x)):
            if idx != len(x) - 1:
                cv2.line(cv2img, convertPos(x[idx], y1[idx]), convertPos(x[idx + 1], y1[idx + 1]), color=(255, 255, 255, 255))  # type: ignore
                cv2.line(cv2img, convertPos(x[idx], y2[idx]), convertPos(x[idx + 1], y2[idx + 1]), color=(255, 255, 255, 255))  # type: ignore
        for idx in range(len(x_start_lst)):
            if idx != len(x_start_lst) - 1:
                cv2.line(cv2img, convertPos(x_start_lst[idx], y_start_lst[idx]), convertPos(x_start_lst[idx + 1], y_start_lst[idx + 1]), color=(255, 255, 255, 255))  # type: ignore
        for idx in range(len(x_end_lst)):
            if idx != len(x_end_lst) - 1:
                cv2.line(cv2img, convertPos(x_end_lst[idx], y_end_lst[idx]), convertPos(x_end_lst[idx + 1], y_end_lst[idx + 1]), color=(255, 255, 255, 255))  # type: ignore

    def __handle_curve(cv2img, control_points):
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
        for i in range(len(curve_points[:, 0])):
            if i != len(curve_points[:, 0]) - 1:
                cv2.line(cv2img, convertPos(curve_points[:, 0][i], curve_points[:, 1][i]), convertPos(curve_points[:, 0][i + 1], curve_points[:, 1][i + 1]), color=(255, 255, 255, 255))  # type: ignore

        return curve_points[:, 0], curve_points[:, 1]

    if shape["type"] == "fusiform_1":
        x_offset = shape["x_offset"]
        y_offset = shape["y_offset"]
        fc = shape["focal_length"]
        eps, ome, phi = shape["sin_params"]
        x_start = shape["x_start"]
        x_end = shape["x_end"]
        y_sim = shape["y_symmetric_axis"]
        __handle_fusiform_1(cv2img, fc, x_offset, y_offset, eps, ome, phi, x_start, x_end, y_sim)
    elif shape["type"] == "fusiform_2":
        x_offset = shape["x_offset"]
        y_offset = shape["y_offset"]
        fc = shape["focal_length"]
        eps, ome, phi = shape["sin_params"]
        power = shape["power"]
        x_start = shape["x_start"]
        x_end = shape["x_end"]
        __handle_fusiform_2(
            cv2img,
            focal_length=fc,
            x_offset=x_offset,
            y_offset=y_offset,
            power=power,
            eps=eps,
            omega=ome,
            phi=phi,
            x_start=x_start,
            x_end=x_end,
        )
    elif shape["type"] == "ellipse":
        ellipse_x, ellipse_y = shape["center"]
        major = shape["major_axis"]
        minor = shape["minor_axis"]
        alpha = shape["rotation"] * 180 / np.pi
        __handle_ellipse(cv2img, ellipse_x, ellipse_y, major, minor, alpha)
    elif shape["type"] == "polygon":
        points: list = shape["points"]
        assert len(points) >= 3, "There should be more than 3 points within a polygon."
        __handle_polygon(cv2img, points)
    elif shape["type"] == "spindle":
        center_x, center_y = shape["center"]
        major = shape["major_axis"]
        minor = shape["minor_axis"]
        __handle_spindle(cv2img, center_x, center_y, major, minor)
    elif shape["type"] == "curves":
        curves = shape["control_points"]
        curve_buf_x = []
        curve_buf_y = []
        for curve in curves:
            x, y = __handle_curve(cv2img, curve)
            curve_buf_x.append(x)
            curve_buf_y.append(y)

        curve_buf_x = [item for x in curve_buf_x for item in x]
        curve_buf_y = [item for y in curve_buf_y for item in y]
        """
            debug_fig = plt.figure()
            ax = debug_fig.add_subplot().scatter(curve_buf_x, curve_buf_y, s=1, marker='x')
            debug_fig.savefig(f"DEBUG/scatter_points{index}.png")
            #"""


def generate_basic_shape_wrapper(data_dict):
    shapes = data_dict["shapes"]
    ni = data_dict["ni"]
    cv2img0 = np.zeros((size_[0], size_[1], 4), dtype=np.uint8)
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
        draw(cv2img0, shape)
    # if "fusiform_2" in filtered_shapes[0]["type"] or filtered_shapes[0]["type"]=="fusiform_2":
    #     cv2.imwrite("query_img.png",cv2img0)
    #     sys.exit()
    query_img = cv2img0
    return query_img


def get_best_match_wrapper(data_dict):
    return getMostSimilarImages(
        data_dict["pool_prefix"],
        data_dict["img"],
        data_dict["bbox"],
        n=1,
        max_sample=99999,
        batchsize=128,
        debug=False,
        multi_process=False,
        tmp_dir=data_dict["tmp_dir"],
    )[0][0]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Get Most Similar Images")

    # 添加参数
    parser.add_argument("--rules", type=str, required=True, help="Path of rules file")
    parser.add_argument("--save_path", type=str, required=True, help="Path of result file")
    parser.add_argument("--start_pos", type=int, required=False, default=0)
    parser.add_argument("--end_pos", type=int, required=False, default=-1)
    # 解析参数
    args = parser.parse_args()

    dirpath = os.path.dirname(args.save_path)

    # 如果目录不存在，则创建
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    with open(args.rules, "r") as f:
        if args.end_pos == -1:
            samples = json.load(f)[args.start_pos :]
        else:
            samples = json.load(f)[args.start_pos : args.end_pos]
    # with open("sample0.json","w") as fff:
    #     json.dump(samples,fff,indent=2)
    batch_size = min(cpu_count(), len(samples))
    # batch_size=1
    pool_prefix = "pics_12xx/"
    rule_name = args.rules.split("/")[-1]
    tmp_dir = f"/dev/shm/.{rule_name}/"

    multi_proc1 = True
    multi_proc2 = True

    with open(args.save_path, "w") as f:
        basic_shape_batched_data = []
        basic_shape_temp_batch = []
        batched_data = []
        temp_batch = []
        basic_shapes = []
        best_refs_poles = []
        if multi_proc1:
            with Pool(cpu_count()) as proc_pool:
                _out_pipe, _in_pipe = Pipe(True)
                for index, sample in enumerate(tqdm(samples)):
                    basic_shape_temp_batch.append(
                        {"shapes": sample["shapes"], "ni": sample["numerical_info"]}
                    )
                    if index >= len(samples) / batch_size - 1:
                        basic_shape_batched_data.append([basic_shape_temp_batch, _in_pipe])
                        basic_shape_temp_batch = []
                # profiler=Profiler()
                # profiler.start()
                try:
                    proc1_pbar = tqdm(total=len(samples))
                    proc1_pbar.set_description("genBasicShape")
                    res = proc_pool.starmap_async(wrapper_retard, basic_shape_batched_data)
                    proc_pool.close()
                    # for b in tqdm(basic_shape_batched_data):
                    while True:  # type: ignore
                        try:
                            if res.ready():
                                break
                            if proc1_pbar.n < proc1_pbar.total:
                                msg = _out_pipe.recv()
                                proc1_pbar.update()
                            # sys.stdout.write("created!\n")
                            # sys.stdout.flush()
                        except:
                            pass
                    print("genBasicShape done!")
                    # res=iterate_wrapper(
                    #     wrapper_retard,
                    #     basic_shape_batched_data,
                    #     output=None,
                    #     num_workers=batch_size,
                    #     run_name="generate_basic_shape_wrapper",
                    #     bar=True,
                    #     tmp_dir = tmp_dir
                    # )
                    # print("iter_wrap done!")
                    for ress in res.get():  # type: ignore
                        basic_shapes.extend(ress)
                except Exception as e:
                    print(e)
                # profiler.stop()
                # profiler.print()
        else:
            # profiler=Profiler()
            # profiler.start()
            for index, sample in enumerate(tqdm(samples)):
                basic_shapes.append(
                    generate_basic_shape_wrapper({"shapes": sample["shapes"], "ni": sample["numerical_info"]})
                )
            # profiler.stop()
            # profiler.print()
            # while True:
            #     pass
            # basic_img, volution_memory, max_volution, query_img = generate_basic_shape_wrapper(
            #     sample["shapes"], sample["numerical_info"]
            # )
        if multi_proc2:
            # profiler=Profiler()
            # profiler.start()
            for index, query_img in tqdm(enumerate(basic_shapes)):
                batched_data.append(
                    {
                        "pool_prefix": pool_prefix,
                        "img": query_img,
                        "bbox": samples[index]["numerical_info"]["fossil_bbox"],
                        "tmp_dir": tmp_dir,
                    }
                )
                # if len(temp_batch)>=batch_size:
                #     batched_data.append(temp_batch)
                #     temp_batch=[]
            try:
                res = iterate_wrapper(
                    get_best_match_wrapper,
                    data=batched_data,
                    output=None,
                    num_workers=batch_size,
                    run_name="get_best_match_wrapper",
                    bar=True,
                    tmp_dir="/dev/shm",
                )
                best_refs_poles.extend(res)  # type: ignore
                # best_ref_poles=get_best_match(query_img,"pics/",sample["numerical_info"]["fossil_bbox"],tmp_dir="./tmp_dir_1/")
            # best_ref_poles="pics/Neoschwagerina_takagamiensis_1_2.png"
            except Exception as e:
                print(e)
            # profiler.stop()
            # profiler.print()
        else:
            # profiler=Profiler()
            # profiler.start()
            for bs in tqdm(basic_shapes):
                best_refs_poles.append(
                    get_best_match_wrapper(
                        {
                            "pool_prefix": pool_prefix,
                            "img": bs,
                            "bbox": samples[index]["numerical_info"]["fossil_bbox"],
                            "tmp_dir": tmp_dir,
                        }
                    )
                )
            # profiler.stop()
            # profiler.print()
            # best_ref_poles = get_random_match("pics/")
        for best_ref_poles in best_refs_poles:
            f.write(best_ref_poles)
            f.write("\n")
            f.flush()
