from svgpathtools import Path, Line

import cv2
import math
import numpy as np
from scipy.signal import find_peaks


def chomatas_scan(
    data: dict,
    img_path: str,
    path_segment_n: int = 100,
    peak_bin_num: int = 10,
    peak_height_ratio=0.5,
    default_slice_total_height=20,
):
    """
    # 旋脊识别chomatas_scan

    ## 参数列表

    * data: 螺旋边缘点序列，格式`{1:[(p1x, p1y), (p2x, p2y)...], 2:[(...)...], ..., -1:[(...)...], -2:[(...)...]}`，
    注意正数索引表示上半区volution，负数表示下半区，连续数字应尽可能表示相邻的两条volution，坐标轴与opencv相同，点序列不应存在拥有相同x坐标的两个点

    * img_path: 待识别图像路径，应当使用原图，而非二值化图像

    * path_segment_n: volution细分片段总数

    * peak_bin_num: 用来排序“最黑的”部分的桶的个数，应当远小于细分片段总数

    * peak_height_ratio: 视黑色像素为255，白色像素为0情况下，求峰值，找到的峰的最小高度为`该条volution中最黑的像素值*peak_height_ratio`

    * default_slice_total_height: 最外圈识别时所扫描的宽度

    ## 返回值

    * `{1:[(p1x, p1y), (p2x, p2y)], 2:[(q1x, q1y), (q2x, q2y)], ..., -1:[(...)...], -2:[(...)...]}`，注意可能存在某一条volution只识别出一个旋脊的可能
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
            slice0 = {"start_points": [], "start_Ts": [], "end_points": [], "pixel_avg": []}
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
                    pixel_average = np.sum(masked_pixels) / len(masked_pixels)
                    slice0["start_points"].append(start_point)
                    slice0["start_Ts"].append(T)
                    slice0["end_points"].append(end_point)
                    slice0["pixel_avg"].append(pixel_average)
            return slice0

        def getOneSlice_up(path: Path, iy):
            step = 1 / line_n
            slice0 = {"start_points": [], "start_Ts": [], "end_points": [], "pixel_avg": []}
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
                    pixel_average = np.sum(masked_pixels) / len(masked_pixels)
                    slice0["start_points"].append(start_point)
                    slice0["start_Ts"].append(T)
                    slice0["end_points"].append(end_point)
                    slice0["pixel_avg"].append(pixel_average)
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
        for y in range(y_total):
            y_idx = y + 1
            slice0 = getOneSlice(path, y_idx)
            frame = 255 - np.array(slice0["pixel_avg"])
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

        if len(peak_bins) >= 2:
            peak_bins = [peak_bins[0], peak_bins[1]]
        elif len(peak_bins) >= 1:
            peak_bins = [peak_bins[0]]
        for bin in peak_bins:
            bin["chosen_point"] = points_max(bin["end_points"], bin["pixel_avg"])
        return [bin["chosen_point"] for bin in peak_bins]  # type: ignore

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
                    if length < min_height:
                        min_height = length
        if has_intersects:
            return int(min_height - 2)
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
