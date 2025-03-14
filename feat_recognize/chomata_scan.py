import scipy.signal
from svgpathtools import Path, Line

import cv2
import math
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import scipy


def chomatas_scan(
    data: dict,
    img_path: str,
    path_segment_n: int = 100,
    peak_bin_num: int = 10,
    peak_height_ratio=0.5,
    default_slice_total_height=20,
    merge_threshold=0.1,
    find_platforms_threshold=0.1,
    chomata_sensitivity=5.0,
):
    """
    # 旋脊识别chomatas_scan

    ## 参数列表

    * data: 螺旋边缘点序列，格式`{1:[(p1x, p1y), (p2x, p2y)...], 2:[(...)...], ..., -1:[(...)...], -2:[(...)...]}`，
    注意正数索引表示上半区volution，负数表示下半区，连续数字应尽可能表示相邻的两条volution，坐标轴与opencv相同，点序列不应存在拥有相同x坐标的两个点

    * img_path: 待识别图像路径，应当使用原图，而非二值化图像

    * path_segment_n: volution细分片段总数

    * peak_bin_num: 已弃用

    * peak_height_ratio: 已弃用

    * default_slice_total_height: 最外圈识别时所扫描的宽度

    ## 返回值

    * `{1:[(p1x, p1y, c1), (p2x, p2y, c2)], 2:[(q1x, q1y, c1), (q2x, q2y, c2)], ..., -1:[(...)...], -2:[(...)...]}`，c表示置信度（即黑色部分面积，范围统一到了0~path_segment_n）
    """
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    slice_step = 1
    slice_total_height = None
    peak_min_distance = None
    peak_min_height = None
    peak_min_width = 5

    def scan(
        path: Path,
        path_segment_n,
        slice_step,
        slice_total_height,
        peak_min_distance,
        peak_min_width,
        peak_min_height,
        peak_bin_num,
        controller,
    ):

        line_n = path_segment_n
        y_step = slice_step
        y_total = slice_total_height
        distance = peak_min_distance
        peak_height = peak_min_height
        peak_width = peak_min_width
        barrier_num = peak_bin_num
        distance = int(path.length() / barrier_num)
        if distance < 1:
            distance = 1

        def getOneSlice_down(path: Path, iy):
            step = 1 / line_n
            slice0 = {
                "start_points": [],
                "start_Ts": [],
                "end_points": [],
                "pixel_avg": [],
                "pixel_total": [],
            }
            for i in range(line_n):
                T = step * (i + 1)
                if T < 1 and T > 0:
                    p = path.point(T)
                    cmp = path.unit_tangent(T)

                    end_vec = cmp * complex(0, 1) * (y_step * iy) + p
                    start_point = (math.ceil(p.real), math.ceil(p.imag))
                    end_point = (math.ceil(end_vec.real), math.ceil(end_vec.imag))
                    mask = np.zeros(img.shape, np.uint8)
                    cv2.line(mask, start_point, end_point, (255, 255, 255), lineType=cv2.LINE_AA)
                    masked_pixels = img[mask > 0]
                    pixel_total = np.sum(255 - masked_pixels)
                    pixel_average = pixel_total / len(masked_pixels)
                    slice0["start_points"].append(start_point)
                    slice0["start_Ts"].append(T)
                    slice0["end_points"].append(end_point)
                    slice0["pixel_avg"].append(pixel_average)
                    slice0["pixel_total"].append(pixel_total)
            return slice0

        def getOneSlice_up(path: Path, iy):
            step = 1 / line_n
            slice0 = {
                "start_points": [],
                "start_Ts": [],
                "end_points": [],
                "pixel_avg": [],
                "pixel_total": [],
            }
            for i in range(line_n):
                T = step * (i + 1)
                if T < 1 and T > 0:
                    p = path.point(T)
                    cmp = path.unit_tangent(T)
                    end_vec = p - cmp * complex(0, 1) * (y_step * iy)
                    start_point = (math.floor(p.real), math.floor(p.imag))
                    end_point = (math.floor(end_vec.real), math.floor(end_vec.imag))
                    mask = np.zeros(img.shape, np.uint8)
                    cv2.line(mask, start_point, end_point, (255, 255, 255), lineType=cv2.LINE_AA)
                    masked_pixels = img[mask > 0]
                    pixel_total = np.sum(255 - masked_pixels)
                    pixel_average = pixel_total / len(masked_pixels)
                    slice0["start_points"].append(start_point)
                    slice0["start_Ts"].append(T)
                    slice0["end_points"].append(end_point)
                    slice0["pixel_avg"].append(pixel_average)
                    slice0["pixel_total"].append(pixel_total)
            return slice0

        if controller > 0:
            getOneSlice = getOneSlice_up
        else:
            getOneSlice = getOneSlice_down

        def getMinPeakHeight():
            step = 1 / line_n
            points = []
            for i in range(line_n):
                T = step * (i + 1)
                if T < 1 and T > 0:
                    p = path.point(T)
                    points.append((math.floor(p.real), math.floor(p.imag)))
            masked_pixels = [int(img[p0[1]][p0[0]]) for p0 in points]
            return int((255 - min(masked_pixels)) * peak_height_ratio)

        def points_max(points, pixel_avg):
            return min(zip(points, pixel_avg), key=lambda x: x[1])[0]

        peak_bins = []
        for b in range(barrier_num):
            peak_bins.append({"gate": 1 / barrier_num * (b + 1), "Ts": [], "end_points": [], "pixel_avg": []})
        frame_areas = []
        frame_areas_gradient = []
        frame_areas_gradient2 = []
        img_bg = np.ones((y_step * y_total + 1, line_n), np.uint8) * 255

        def bresenham_line(p1, p2):
            x0 = p1[0]
            y0 = p1[1]
            x1 = p2[0]
            y1 = p2[1]
            points = []
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx - dy

            while True:
                points.append((x0, y0))
                if x0 == x1 and y0 == y1:
                    break
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x0 += sx
                if e2 < dx:
                    err += dx
                    y0 += sy

            return points

        for y in range(y_total):
            y_idx = y + 1
            slice0 = getOneSlice(path, y_idx)
            frame = 255 - np.array(slice0["pixel_avg"])
            frame2 = np.array(slice0["pixel_total"])
            frame_area = []
            xs = list(range(len(frame2)))
            for end_frame_pos in range(len(frame2)):
                frame_area.append(np.trapz(frame2[0 : end_frame_pos + 1]))
            frame_areas_gradient.append(scipy.signal.savgol_filter(frame2, 15, 3))
            frame_area_g2 = np.gradient(frame2, xs)
            frame_area_g2 = scipy.signal.savgol_filter(frame_area_g2, 15, 3)
            # frame_area = np.gradient(frame_area,xs)
            frame_areas.append(frame_area)
            frame_areas_gradient2.append(frame_area_g2)
            if y == y_total - 1:
                for iii in range(len(slice0["start_points"])):
                    # st = slice0["start_points"][iii]
                    # ed = slice0["end_points"][iii]
                    # mi = ((st[0]+ed[0])/2,(st[1]+ed[1])/2)
                    # dst_st = (iii,0)
                    # dst_ed = (iii,y_step*y_total)
                    # dst_mi = (iii,y_step*y_total/2)
                    # M = cv2.getAffineTransform(np.array([st,mi,ed],dtype=np.float32),np.array([dst_st,dst_mi,dst_ed],dtype=np.float32))
                    # cutted_original = cv2.bitwise_and(img,img,mask=slice0["masked_pixels"][iii])
                    # dst = cv2.warpAffine(cutted_original,M,(img_bg.shape[1],img_bg.shape[0]))
                    ps = bresenham_line(slice0["start_points"][iii], slice0["end_points"][iii])
                    for yyy in range(img_bg.shape[0]):
                        if yyy < len(ps):
                            if ps[yyy][1] < len(img) and ps[yyy][0] < len(img[ps[yyy][1]]):
                                img_bg[yyy][iii] = img[ps[yyy][1]][ps[yyy][0]]
            peak_height = getMinPeakHeight()
            peaks, _ = find_peaks(frame, distance=distance, width=peak_width, height=peak_height)
            peaks = list(peaks)
            for peak in peaks:
                for bin in peak_bins:
                    if slice0["start_Ts"][peak] < bin["gate"]:
                        bin["Ts"].append(slice0["start_Ts"][peak])
                        bin["end_points"].append(slice0["end_points"][peak])
                        bin["pixel_avg"].append(slice0["pixel_avg"][peak])
                        break
        # print(peak_bins)
        peak_bins = [bin for bin in peak_bins if len(bin["Ts"]) > 0]
        peak_bins.sort(
            key=lambda x: (255 - sum(x["pixel_avg"]) / len(x["pixel_avg"])) + len(x["Ts"]) * 1000,
            reverse=True,
        )

        def standardization(unstd_data):
            # mu = np.mean(unstd_data, axis=0)
            mu = 0
            sigma = np.max(abs(unstd_data))
            return (unstd_data - mu) / sigma

        def normalization(unnorm_data):
            _range = np.max(unnorm_data) - np.min(unnorm_data)
            return (unnorm_data - np.min(unnorm_data)) / _range

        def my_sigmoid_modified(s_x):
            return 1 / (1 + math.exp(-10 * (s_x - 0.5)))

        def find_platforms(grad2_, areas):
            grad2_neg = np.zeros(len(grad2_), dtype=np.float64)
            grad2_neg[grad2_ < 0] = grad2_[grad2_ < 0]
            grad2_neg = abs(grad2_neg)
            ed_peak, un_pe = find_peaks(grad2_neg, height=find_platforms_threshold)
            grad2_pos = np.zeros(len(grad2_), dtype=np.float64)
            grad2_pos[grad2_ > 0] = grad2_[grad2_ > 0]
            st_peak, un_pe = find_peaks(grad2_pos, height=find_platforms_threshold)
            ist = 0
            ied = 0
            p_pairs = []
            while ist < len(st_peak) or ied < len(ed_peak):
                min_p = None
                if ist < len(st_peak):
                    min_p = (st_peak[ist], 1)
                if ied < len(ed_peak):
                    if min_p is None or min_p[0] > ed_peak[ied]:
                        min_p = (ed_peak[ied], -1)
                if min_p[1] < 0:  # type: ignore
                    ied += 1
                else:
                    ist += 1
                p_pairs.append(min_p)
            if len(p_pairs) == 1:
                if p_pairs[0][1] < 0:
                    p_pairs = [(0, 1)] + p_pairs
                else:
                    p_pairs.append((len(grad2_) - 1, -1))
            else:
                if p_pairs[0][1] < 0:
                    p_pairs = [(0, 1)] + p_pairs
                if p_pairs[-1][1] > 0:
                    p_pairs.append((len(grad2_) - 1, -1))
            p_pairs_merged = []
            temp = None
            for ip in range(len(p_pairs)):
                if temp is None:
                    temp = p_pairs[ip]
                    continue
                if p_pairs[ip][1] == temp[1]:
                    if abs(grad2_[p_pairs[ip][0]]) > abs(grad2_[temp[0]]):
                        temp = p_pairs[ip]
                    continue
                else:
                    p_pairs_merged.append(temp)
                    temp = p_pairs[ip]
            p_pairs_merged.append(temp)
            ip = 0
            p_pairs_final_merged = []
            temp = []

            def cal_area(idx0):
                total_area = 0
                for area in areas:
                    total_area += (normalization(area))[idx0]
                return total_area / len(areas)

            while ip < len(p_pairs_merged):
                if temp == []:
                    temp = [p_pairs_merged[ip], p_pairs_merged[ip + 1]]
                else:
                    if abs(cal_area(p_pairs_merged[ip][0]) - cal_area(temp[1][0])) < merge_threshold:
                        temp[1] = p_pairs_merged[ip + 1]
                        if grad2_[p_pairs_merged[ip][0]] > grad2_[temp[0][0]]:
                            temp[0] = p_pairs_merged[ip]
                    else:
                        p_pairs_final_merged.extend(temp)
                        temp = [p_pairs_merged[ip], p_pairs_merged[ip + 1]]
                ip += 2
            p_pairs_final_merged.extend(temp)
            return p_pairs_merged

        peaks_plat = find_platforms(
            standardization(sum(frame_areas_gradient2) / len(frame_areas_gradient2)), frame_areas
        )
        plt.figure(dpi=100)
        plt.imshow(img_bg, origin="lower")
        std_grad2 = (
            standardization(sum(frame_areas_gradient2) / len(frame_areas_gradient2)) * y_step * y_total
        )
        plt.plot(list(range(len(frame_areas_gradient2[0]))), std_grad2)
        plt.plot(
            list(range(len(frame_areas_gradient[0]))),
            standardization(sum(frame_areas_gradient) / len(frame_areas_gradient)) * y_step * y_total,
        )
        for p in peaks_plat:
            plt.plot(p[0], std_grad2[p[0]], marker="o", color="red" if p[1] > 0 else "green")
        for a in frame_areas:
            plt.plot(list(range(len(a))), normalization(a) * y_step * y_total)
            # plt.plot(list(range(len(a))),a)
        ax = plt.gca()
        ratio = 0.3
        x_left, x_right = ax.get_xlim()
        y_low, y_high = ax.get_ylim()
        ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)
        # for bin in peak_bins:
        #     bin["chosen_point"] = points_max(bin["end_points"], bin["pixel_avg"])
        ip = 0
        final_chomatas = []
        final_chomatas_debug = []

        def get_all_candidates_width_norm():
            widths = []
            ip___ = 0
            while ip___ < len(peaks_plat):
                widths.append(peaks_plat[ip___ + 1][0] - peaks_plat[ip___][0])
                ip___ += 2
            return standardization(np.array(widths, dtype=np.float64))

        all_candidates_norm = get_all_candidates_width_norm()

        def get_confidence(p_st, p_ed, ip__, p_ed_before=None):
            avg_grad1 = []  # 平均一阶导
            avg_y = []  # 平均面积序列
            avg_grad_before = []
            for a in frame_areas:
                aa = normalization(a) * line_n
                avg_grad1.append((aa[p_ed[0]] - aa[p_st[0]]) / (p_ed[0] - p_st[0]))
                avg_y.append(float(aa[p_ed[0]] - aa[p_st[0]]))
                # 计算chomata存在性（如果有）
                if p_ed_before is not None:
                    avg_grad_before.append((aa[p_st[0]] - aa[p_ed_before[0]]) / (p_st[0] - p_ed_before[0]))
            conf = float(np.average(avg_y))
            # conf = conf*0.9+all_candidates_norm[ip__//2]*0.1
            if p_ed_before is not None:
                total_conf_ab = 0
                for a, b in zip(avg_grad1, avg_grad_before):
                    total_conf_ab += a / b
            return conf

        while ip < len(peaks_plat):
            if ip == 0:
                p_before = None
                if peaks_plat[ip][0] > 0:
                    p_before = (0, 1)
                c0 = get_confidence(peaks_plat[ip], peaks_plat[ip + 1], ip, p_before)
                if c0 > chomata_sensitivity:
                    x0 = (peaks_plat[ip][0] + peaks_plat[ip + 1][0]) / 2
                    T0 = x0 / line_n
                    p0 = path.point(T0)
                    final_chomatas_debug.append((int(x0), c0))
                    final_chomatas.append((math.floor(p0.real), math.floor(p0.imag), c0))
            else:
                p_before = peaks_plat[ip - 1]
                c0 = get_confidence(peaks_plat[ip], peaks_plat[ip + 1], ip, p_before)
                if c0 > chomata_sensitivity:
                    x0 = (peaks_plat[ip][0] + peaks_plat[ip + 1][0]) / 2
                    T0 = x0 / line_n
                    p0 = path.point(T0)
                    final_chomatas_debug.append((int(x0), c0))
                    final_chomatas.append((math.floor(p0.real), math.floor(p0.imag), c0))
            ip += 2
        # return [bin["chosen_point"] for bin in peak_bins]  # type: ignore
        for p in final_chomatas_debug:
            plt.plot(p[0], std_grad2[p[0]], marker="x", color="blue")
            plt.text(p[0], std_grad2[p[0]], str(round(p[1], 4)))
        plt.show()
        final_chomatas.sort(key=lambda x: x[2], reverse=True)
        return final_chomatas if len(final_chomatas) > 2 else final_chomatas

    def parsePathUsingLines(points):
        if type(points) == Path:
            return points
        path0 = []
        for i in range(len(points) - 1):
            path0.append(
                Line(complex(points[i][0], points[i][1]), complex(points[i + 1][0], points[i + 1][1]))
            )
        return Path(*path0)

    def getMaxSliceHeight(k1, k2):
        path1 = parsePathUsingLines(data[k1])
        path2 = parsePathUsingLines(data[k2])
        step = 1 / path_segment_n
        min_height = 99999
        total_height = 0
        total_intersects = 0
        has_intersects = False
        for i in range(path_segment_n):
            T = step * (i + 1)
            if T < 1 and T > 0:
                p = path1.point(T)
                cmp = path1.unit_tangent(T)
                modifier = 1 if k2 - k1 < 0 else -1
                end_vec = p + modifier * cmp * complex(0, 1) * (999)
                line0 = Line(p, end_vec)
                intersects = path2.intersect(line0)
                if len(intersects) > 0:
                    has_intersects = True
                    ans = intersects[0]
                    p2 = path2.point(ans[0][0])
                    length = Line(p, p2).length()
                    total_height += length
                    total_intersects += 1
                    if length < min_height:
                        min_height = length
        if has_intersects:
            # return int(min_height - 2)
            return int(total_height / total_intersects * 0.8)
        else:
            return default_slice_total_height

    ups = [k for k in data.keys() if k > 0]
    downs = [k for k in data.keys() if k < 0]
    ups.sort()
    downs.sort(reverse=True)
    result_dict = {}
    for i in range(len(ups)):
        k = ups[i]
        if i == len(ups) - 1:
            slice_total_height = default_slice_total_height
        else:
            slice_total_height = getMaxSliceHeight(k, ups[i + 1])
        chomatas = scan(
            parsePathUsingLines(data[k]),
            path_segment_n,
            slice_step,
            slice_total_height,
            peak_min_distance,
            peak_min_width,
            peak_min_height,
            peak_bin_num,
            k,
        )
        result_dict[k] = chomatas
    for i in range(len(downs)):
        k = downs[i]
        if i == len(downs) - 1:
            slice_total_height = default_slice_total_height
        else:
            slice_total_height = getMaxSliceHeight(k, downs[i + 1])
        chomatas = scan(
            parsePathUsingLines(data[k]),
            path_segment_n,
            slice_step,
            slice_total_height,
            peak_min_distance,
            peak_min_width,
            peak_min_height,
            peak_bin_num,
            k,
        )
        result_dict[k] = chomatas
    return result_dict
