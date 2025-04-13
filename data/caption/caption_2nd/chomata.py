import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.signal import find_peaks

from data.caption.caption_2nd.base import BaseFeature
from data.caption.caption_2nd.custom_shapes import (
    Curve,
    CustomedShape,
    Ellipse,
    Fusiform,
    Fusiform_2,
)
from data.caption.caption_2nd.params import *


class Chomata(BaseFeature):
    def __init__(self, chomata_shapes, volution_num, volutions_table):
        self.chomata_shapes = chomata_shapes
        self.volution_num = volution_num
        self.volutions_table = volutions_table
        self.genVolutionHeightsAndWidths()
        self.genChomataMetaData()

    def my_from_dict(self, shape, chomata_center):
        if shape["type"] == "curves":
            curves = []
            for control_points in shape["control_points"]:
                curve = Curve(
                    control_points=control_points,
                    special_info=shape["special_info"],
                    fill_mode=shape["fill_mode"],
                )
                curves.append(curve)
            return CustomedShape(
                curves=curves,
                support_point=chomata_center,
                special_info=shape["special_info"],
                fill_mode=shape["fill_mode"],
            )
        elif shape["type"] == "ellipse":
            return Ellipse(
                support_point=chomata_center,
                center=shape["center"],
                major_axis=shape["major_axis"],
                minor_axis=shape["minor_axis"],
                rotation=shape["rotation"],
                special_info=shape["special_info"],
                fill_mode=shape["fill_mode"],
            )
        elif shape["type"] == "fusiform_1":
            return Fusiform(
                support_point=chomata_center,
                focal_length=shape["focal_length"],
                x_offset=shape["x_offset"],
                y_offset=shape["y_offset"],
                y_symmetric_axis=shape["y_symmetric_axis"],
                sin_params=shape["sin_params"],
                x_start=shape["x_start"],
                x_end=shape["x_end"],
                special_info=shape["special_info"],
                fill_mode=shape["fill_mode"],
            )
        elif shape["type"] == "fusiform_2":
            return Fusiform_2(
                focal_length=shape["focal_length"],
                x_offset=shape["x_offset"],
                y_offset=shape["y_offset"],
                power=shape["power"],
                x_symmetric_axis=shape["x_symmetric_axis"],
                sin_params=shape["sin_params"],
                x_start=shape["x_start"],
                x_end=shape["x_end"],
                special_info=shape["special_info"],
                fill_mode=shape["fill_mode"],
            )

    def getTangentLineFunction(self, tangent_point, curve_points_interp1d):

        curve_func = curve_points_interp1d

        slope = derivative(curve_func, tangent_point[0], dx=1e-6)

        y_target = curve_func(tangent_point[0])
        intercept = y_target - slope * tangent_point[0]

        if slope == float("inf"):
            A = 1
            B = 0
            C = -tangent_point[0]
        elif np.abs(slope) > 1e-4:
            A = slope
            B = -1
            C = intercept
        else:
            A = 0
            B = -1
            C = intercept

        return (A, B, C)

    def getPerpendicularLineFunction(self, line_ABC, point):
        A, B, C = line_ABC
        if A == 0:
            Ap = 1
            Bp = 0
            Cp = -point[0]
        elif B == 0:
            Ap = 0
            Bp = 1
            Cp = -point[1]
        else:
            perpendicular_slope = -1 / (-A / B)
            perpendicular_intercept = point[1] - perpendicular_slope * point[0]
            Ap = perpendicular_slope
            Bp = -1
            Cp = perpendicular_intercept
        return (Ap, Bp, Cp)

    def solveEllipseIntersections(self, A, B, C, a, b, c_x, c_y, r_s, r_c):
        points = []
        delta = (
            A**2 * a**2 * r_c**2
            + A**2 * b**2 * r_s**2
            - 4 * A**2 * c_x**2
            + 2 * A * B * a**2 * r_c * r_s
            - 2 * A * B * b**2 * r_c * r_s
            - 8 * A * B * c_x * c_y
            - 8 * A * C * c_x
            + B**2 * a**2 * r_s**2
            + B**2 * b**2 * r_c**2
            - 4 * B**2 * c_y**2
            - 8 * B * C * c_y
            - 4 * C**2
        )
        if delta >= 0.0:
            px1 = (
                -(
                    C
                    - (
                        B
                        * (
                            2 * B * C * a**2 * r_s**2
                            + 2 * B * C * b**2 * r_c**2
                            - 2 * A**2 * a**2 * c_y * r_c**2
                            - 2 * A**2 * b**2 * c_y * r_s**2
                            + 2 * A**2 * a**2 * c_x * r_c * r_s
                            - 2 * A**2 * b**2 * c_x * r_c * r_s
                            + A
                            * a
                            * b
                            * r_c**2
                            * (
                                A**2 * a**2 * r_c**2
                                + A**2 * b**2 * r_s**2
                                - 4 * A**2 * c_x**2
                                + 2 * A * B * a**2 * r_c * r_s
                                - 2 * A * B * b**2 * r_c * r_s
                                - 8 * A * B * c_x * c_y
                                - 8 * A * C * c_x
                                + B**2 * a**2 * r_s**2
                                + B**2 * b**2 * r_c**2
                                - 4 * B**2 * c_y**2
                                - 8 * B * C * c_y
                                - 4 * C**2
                            )
                            ** (1 / 2)
                            + A
                            * a
                            * b
                            * r_s**2
                            * (
                                A**2 * a**2 * r_c**2
                                + A**2 * b**2 * r_s**2
                                - 4 * A**2 * c_x**2
                                + 2 * A * B * a**2 * r_c * r_s
                                - 2 * A * B * b**2 * r_c * r_s
                                - 8 * A * B * c_x * c_y
                                - 8 * A * C * c_x
                                + B**2 * a**2 * r_s**2
                                + B**2 * b**2 * r_c**2
                                - 4 * B**2 * c_y**2
                                - 8 * B * C * c_y
                                - 4 * C**2
                            )
                            ** (1 / 2)
                            + 2 * A * C * a**2 * r_c * r_s
                            - 2 * A * C * b**2 * r_c * r_s
                            + 2 * A * B * a**2 * c_x * r_s**2
                            + 2 * A * B * b**2 * c_x * r_c**2
                            - 2 * A * B * a**2 * c_y * r_c * r_s
                            + 2 * A * B * b**2 * c_y * r_c * r_s
                        )
                    )
                    / (
                        2
                        * (
                            A**2 * a**2 * r_c**2
                            + A**2 * b**2 * r_s**2
                            + 2 * A * B * a**2 * r_c * r_s
                            - 2 * A * B * b**2 * r_c * r_s
                            + B**2 * a**2 * r_s**2
                            + B**2 * b**2 * r_c**2
                        )
                    )
                )
                / A
            )
            py1 = -(
                2 * B * C * a**2 * r_s**2
                + 2 * B * C * b**2 * r_c**2
                - 2 * A**2 * a**2 * c_y * r_c**2
                - 2 * A**2 * b**2 * c_y * r_s**2
                + 2 * A**2 * a**2 * c_x * r_c * r_s
                - 2 * A**2 * b**2 * c_x * r_c * r_s
                + A
                * a
                * b
                * r_c**2
                * (
                    A**2 * a**2 * r_c**2
                    + A**2 * b**2 * r_s**2
                    - 4 * A**2 * c_x**2
                    + 2 * A * B * a**2 * r_c * r_s
                    - 2 * A * B * b**2 * r_c * r_s
                    - 8 * A * B * c_x * c_y
                    - 8 * A * C * c_x
                    + B**2 * a**2 * r_s**2
                    + B**2 * b**2 * r_c**2
                    - 4 * B**2 * c_y**2
                    - 8 * B * C * c_y
                    - 4 * C**2
                )
                ** (1 / 2)
                + A
                * a
                * b
                * r_s**2
                * (
                    A**2 * a**2 * r_c**2
                    + A**2 * b**2 * r_s**2
                    - 4 * A**2 * c_x**2
                    + 2 * A * B * a**2 * r_c * r_s
                    - 2 * A * B * b**2 * r_c * r_s
                    - 8 * A * B * c_x * c_y
                    - 8 * A * C * c_x
                    + B**2 * a**2 * r_s**2
                    + B**2 * b**2 * r_c**2
                    - 4 * B**2 * c_y**2
                    - 8 * B * C * c_y
                    - 4 * C**2
                )
                ** (1 / 2)
                + 2 * A * C * a**2 * r_c * r_s
                - 2 * A * C * b**2 * r_c * r_s
                + 2 * A * B * a**2 * c_x * r_s**2
                + 2 * A * B * b**2 * c_x * r_c**2
                - 2 * A * B * a**2 * c_y * r_c * r_s
                + 2 * A * B * b**2 * c_y * r_c * r_s
            ) / (
                2
                * (
                    A**2 * a**2 * r_c**2
                    + A**2 * b**2 * r_s**2
                    + 2 * A * B * a**2 * r_c * r_s
                    - 2 * A * B * b**2 * r_c * r_s
                    + B**2 * a**2 * r_s**2
                    + B**2 * b**2 * r_c**2
                )
            )
            px2 = (
                -(
                    C
                    - (
                        B
                        * (
                            2 * B * C * a**2 * r_s**2
                            + 2 * B * C * b**2 * r_c**2
                            - 2 * A**2 * a**2 * c_y * r_c**2
                            - 2 * A**2 * b**2 * c_y * r_s**2
                            + 2 * A**2 * a**2 * c_x * r_c * r_s
                            - 2 * A**2 * b**2 * c_x * r_c * r_s
                            - A
                            * a
                            * b
                            * r_c**2
                            * (
                                A**2 * a**2 * r_c**2
                                + A**2 * b**2 * r_s**2
                                - 4 * A**2 * c_x**2
                                + 2 * A * B * a**2 * r_c * r_s
                                - 2 * A * B * b**2 * r_c * r_s
                                - 8 * A * B * c_x * c_y
                                - 8 * A * C * c_x
                                + B**2 * a**2 * r_s**2
                                + B**2 * b**2 * r_c**2
                                - 4 * B**2 * c_y**2
                                - 8 * B * C * c_y
                                - 4 * C**2
                            )
                            ** (1 / 2)
                            - A
                            * a
                            * b
                            * r_s**2
                            * (
                                A**2 * a**2 * r_c**2
                                + A**2 * b**2 * r_s**2
                                - 4 * A**2 * c_x**2
                                + 2 * A * B * a**2 * r_c * r_s
                                - 2 * A * B * b**2 * r_c * r_s
                                - 8 * A * B * c_x * c_y
                                - 8 * A * C * c_x
                                + B**2 * a**2 * r_s**2
                                + B**2 * b**2 * r_c**2
                                - 4 * B**2 * c_y**2
                                - 8 * B * C * c_y
                                - 4 * C**2
                            )
                            ** (1 / 2)
                            + 2 * A * C * a**2 * r_c * r_s
                            - 2 * A * C * b**2 * r_c * r_s
                            + 2 * A * B * a**2 * c_x * r_s**2
                            + 2 * A * B * b**2 * c_x * r_c**2
                            - 2 * A * B * a**2 * c_y * r_c * r_s
                            + 2 * A * B * b**2 * c_y * r_c * r_s
                        )
                    )
                    / (
                        2
                        * (
                            A**2 * a**2 * r_c**2
                            + A**2 * b**2 * r_s**2
                            + 2 * A * B * a**2 * r_c * r_s
                            - 2 * A * B * b**2 * r_c * r_s
                            + B**2 * a**2 * r_s**2
                            + B**2 * b**2 * r_c**2
                        )
                    )
                )
                / A
            )
            py2 = -(
                2 * B * C * a**2 * r_s**2
                + 2 * B * C * b**2 * r_c**2
                - 2 * A**2 * a**2 * c_y * r_c**2
                - 2 * A**2 * b**2 * c_y * r_s**2
                + 2 * A**2 * a**2 * c_x * r_c * r_s
                - 2 * A**2 * b**2 * c_x * r_c * r_s
                - A
                * a
                * b
                * r_c**2
                * (
                    A**2 * a**2 * r_c**2
                    + A**2 * b**2 * r_s**2
                    - 4 * A**2 * c_x**2
                    + 2 * A * B * a**2 * r_c * r_s
                    - 2 * A * B * b**2 * r_c * r_s
                    - 8 * A * B * c_x * c_y
                    - 8 * A * C * c_x
                    + B**2 * a**2 * r_s**2
                    + B**2 * b**2 * r_c**2
                    - 4 * B**2 * c_y**2
                    - 8 * B * C * c_y
                    - 4 * C**2
                )
                ** (1 / 2)
                - A
                * a
                * b
                * r_s**2
                * (
                    A**2 * a**2 * r_c**2
                    + A**2 * b**2 * r_s**2
                    - 4 * A**2 * c_x**2
                    + 2 * A * B * a**2 * r_c * r_s
                    - 2 * A * B * b**2 * r_c * r_s
                    - 8 * A * B * c_x * c_y
                    - 8 * A * C * c_x
                    + B**2 * a**2 * r_s**2
                    + B**2 * b**2 * r_c**2
                    - 4 * B**2 * c_y**2
                    - 8 * B * C * c_y
                    - 4 * C**2
                )
                ** (1 / 2)
                + 2 * A * C * a**2 * r_c * r_s
                - 2 * A * C * b**2 * r_c * r_s
                + 2 * A * B * a**2 * c_x * r_s**2
                + 2 * A * B * b**2 * c_x * r_c**2
                - 2 * A * B * a**2 * c_y * r_c * r_s
                + 2 * A * B * b**2 * c_y * r_c * r_s
            ) / (
                2
                * (
                    A**2 * a**2 * r_c**2
                    + A**2 * b**2 * r_s**2
                    + 2 * A * B * a**2 * r_c * r_s
                    - 2 * A * B * b**2 * r_c * r_s
                    + B**2 * a**2 * r_s**2
                    + B**2 * b**2 * r_c**2
                )
            )
            points.append((px1, py1))
            points.append((px2, py2))
        else:
            pass
        return points

    def decideInsideOrOnOrOutside(self, a, b, c_x, c_y, r_s, r_c, point):
        x, y = point
        ellip = (
            (4 * (c_x * r_c + c_y * r_s - r_c * x - r_s * y) ** 2) / (a**2 * (r_c**2 + r_s**2) ** 2)
            + (4 * (c_y * r_c - c_x * r_s + r_s * x - r_c * y) ** 2) / (b**2 * (r_c**2 + r_s**2) ** 2)
            - 1
        )
        if ellip > 0.0:
            return 1
        elif ellip < 0.0:
            return -1
        else:
            return 0

    def euc_dist(self, p1, p2):
        return (abs((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)) ** (1 / 2)

    def findIntersectionsAndDecideNearest(self, cho, test_point, curve_func):
        center = cho["center"]
        major_axis = cho["major_axis"]
        minor_axis = cho["minor_axis"]
        rotation = cho["rotation"]

        linet = self.getTangentLineFunction(test_point, curve_func)
        linep = self.getPerpendicularLineFunction(linet, test_point)
        intersects = self.solveEllipseIntersections(
            linep[0],
            linep[1],
            linep[2],
            major_axis,
            minor_axis,
            center[0],
            center[1],
            np.sin(rotation),
            np.cos(rotation),
        )
        if len(intersects) > 0:
            dist1 = self.euc_dist(test_point, intersects[0])
            dist2 = self.euc_dist(test_point, intersects[0])
            if dist1 < dist2:
                dist = dist1
                intersect = intersects[0]
            else:
                dist = dist2
                intersect = intersects[1]
            return [
                test_point,
                linet,
                linep,
                intersect,
                self.decideInsideOrOnOrOutside(
                    major_axis,
                    minor_axis,
                    center[0],
                    center[1],
                    np.sin(rotation),
                    np.cos(rotation),
                    test_point,
                )
                * dist,
            ]
        else:
            return []

    def genIntersectionList(self, cho_dict, volution_dict):
        vol_GS = self.my_from_dict(volution_dict, cho_dict["center"])
        curve_points = vol_GS.curve_points_recommended  # type: ignore
        cho_GS = self.my_from_dict(cho_dict, (0, 0))
        cho_bbox = cho_GS.get_bbox()  # type: ignore
        x_start = cho_bbox[0][0]
        x_end = cho_bbox[1][0]
        test_point_num = 10
        step = (x_end - x_start) / test_point_num
        x, y = zip(*curve_points)
        # Remove points with duplicate x values
        unique_x = []
        unique_y = []
        for x_val, y_val in zip(x, y):
            if x_val not in unique_x:
                unique_x.append(x_val)
                unique_y.append(y_val)
        x, y = unique_x, unique_y

        curve_func = interp1d(x, y, kind="cubic", fill_value="extrapolate")
        test_points = [(x_start + step * i, curve_func(x_start + step * i)) for i in range(test_point_num)]
        candidates = []
        temp = []
        for point in test_points:
            result = self.findIntersectionsAndDecideNearest(cho_dict, point, curve_func)
            if len(result) > 0:
                temp.append(result)
            else:
                if len(temp) > 0:
                    candidates.append(temp)
                    temp = []
        if len(temp) > 0:
            candidates.append(temp)
        min_dist = 999999.0
        min_idx = 0
        for i in range(len(candidates)):
            min_can = min(candidates[i], key=lambda x: x[-1])
            if min_can[-1] < min_dist:
                min_dist = min_can[-1]
                min_idx = i
        if len(candidates) == 0:
            print("bug")
        return candidates[min_idx]

    def solveEllipseTangentLines(self, A_0, B_0, a, b, c_x, c_y, r_s, r_c):
        C_1 = (
            -A_0 * c_x
            - B_0 * c_y
            - (
                A_0**2 * a**2 * r_c**2
                + A_0**2 * b**2 * r_s**2
                + 2 * A_0 * B_0 * a**2 * r_c * r_s
                - 2 * A_0 * B_0 * b**2 * r_c * r_s
                + B_0**2 * a**2 * r_s**2
                + B_0**2 * b**2 * r_c**2
            )
            ** (1 / 2)
            / 2
        )
        C_2 = (
            (
                A_0**2 * a**2 * r_c**2
                + A_0**2 * b**2 * r_s**2
                + 2 * A_0 * B_0 * a**2 * r_c * r_s
                - 2 * A_0 * B_0 * b**2 * r_c * r_s
                + B_0**2 * a**2 * r_s**2
                + B_0**2 * b**2 * r_c**2
            )
            ** (1 / 2)
            / 2
            - B_0 * c_y
            - A_0 * c_x
        )
        return (A_0, B_0, C_1), (A_0, B_0, C_2)

    def distance_from_point_to_line(self, center, line):
        x0, y0 = center
        A, B, C = line

        numerator = abs(A * x0 + B * y0 + C)
        denominator = (A**2 + B**2) ** (1 / 2)
        return numerator / denominator

    def distance_from_two_parallel_lines(self, line1, line2):
        A, B, C1 = line1
        C2 = line2[2]
        return abs(C1 - C2) / ((A**2 + B**2) ** (1 / 2))

    def calcChomataWidthAndHeight(self, cho_dict, volution_dict):
        inter_list = self.genIntersectionList(cho_dict, volution_dict)
        min_tangent_point = min(inter_list, key=lambda x: x[-1])
        linet = min_tangent_point[1]
        linep = min_tangent_point[2]

        center = cho_dict["center"]
        major_axis = cho_dict["major_axis"]
        minor_axis = cho_dict["minor_axis"]
        rotation = cho_dict["rotation"]
        vol_GS = self.my_from_dict(volution_dict, center)
        center_vol = vol_GS.center  # type: ignore

        line1, line2 = self.solveEllipseTangentLines(
            linet[0],
            linet[1],
            major_axis,
            minor_axis,
            center[0],
            center[1],
            np.sin(rotation),
            np.cos(rotation),
        )
        dist_c1 = self.distance_from_point_to_line(center_vol, line1)
        dist_c2 = self.distance_from_point_to_line(center_vol, line2)
        outer_linet = line1 if dist_c1 > dist_c2 else line2
        chomata_height = self.distance_from_two_parallel_lines(linet, outer_linet)

        line1, line2 = self.solveEllipseTangentLines(
            linep[0],
            linep[1],
            major_axis,
            minor_axis,
            center[0],
            center[1],
            np.sin(rotation),
            np.cos(rotation),
        )
        chomata_width = self.distance_from_two_parallel_lines(line1, line2)
        return chomata_width, chomata_height

    def genChomataMetaData(self):
        chomata_metas_by_volution = {}
        for c in self.chomata_shapes:
            vo = int(c["special_info"][len("chomata of volution ") :])
            if vo not in chomata_metas_by_volution:
                chomata_metas_by_volution[vo] = {"chomatas": [], "volution": self.volutions_table[vo]}
            chomata_metas_by_volution[vo]["chomatas"].append(c)
        self.chomata_whs = {}
        for k in chomata_metas_by_volution:
            c_l = chomata_metas_by_volution[k]
            w_t = 0
            h_t = 0
            for cho in c_l["chomatas"]:
                w, h = self.calcChomataWidthAndHeight(cho, c_l["volution"])
                w_t += w
                h_t += h
            self.chomata_whs[k] = [w_t / len(c_l["chomatas"]), h_t / len(c_l["chomatas"])]
        self.chomata_whs_relative = {}
        for k in self.chomata_whs:
            self.chomata_whs_relative[k] = [
                self.chomata_whs[k][0] / (self.volution_widths[k + 1] - self.volution_widths[k]),
                self.chomata_whs[k][1] / (self.volution_heights[k + 1] - self.volution_heights[k]),
            ]
        self.chomata_sizes = {}
        for k in chomata_metas_by_volution:
            total_sizes = 0
            for c in chomata_metas_by_volution[k]["chomatas"]:
                total_sizes += self.my_from_dict(c, (0, 0)).get_area()  # type: ignore
            self.chomata_sizes[k] = total_sizes / len(chomata_metas_by_volution[k]["chomatas"])
        self.chomata_sizes_relative = {}
        for k in self.chomata_sizes:
            self.chomata_sizes_relative[k] = self.chomata_sizes[k] / (
                (self.volution_widths[k + 1] - self.volution_widths[k])
                * (self.volution_heights[k + 1] - self.volution_heights[k])
            )

    def genChomataSize(self) -> str:
        if len(self.chomata_shapes) <= 0:
            return ""
        total = 0
        chomata_sizes_by_volution = {}
        for k in self.chomata_sizes_relative:
            chomata_sizes_by_volution[k] = self.chomata_sizes_relative[k]
        a = list(chomata_sizes_by_volution.keys())
        a.sort()
        weight_list = [chomata_sizes_by_volution[i] for i in a]
        size_inner = self.standardRangeFilter(chomata_size_classes, weight_list[0])
        size_outer = self.standardRangeFilter(chomata_size_classes, weight_list[-1])
        size_mid = -1
        if len(weight_list) >= 3:
            size_mid = self.standardRangeFilter(chomata_size_classes, weight_list[len(weight_list) // 2])
        if size_mid == -1:
            if size_inner == size_outer:
                return size_inner
            elif size_inner != "moderate" and size_outer != "moderate":
                return "{inner} in inner volutions and {outer} in outer volutions".format(
                    inner=size_inner, outer=size_outer
                )
            elif size_inner != "moderate":
                return "{inner} in inner volutions".format(inner=size_inner)
            elif size_outer != "moderate":
                return "{outer} in outer volutions".format(outer=size_outer)
        else:
            if size_outer == size_inner and size_inner == size_mid:
                return size_inner
            elif size_inner == size_outer:
                if size_mid == "moderate":
                    return "{both} in inner and outer volutions".format(both=size_inner)
                else:
                    return "{both} in inner and outer volutions but {mid} in middle volutions".format(
                        both=size_inner, mid=size_mid
                    )
            elif size_inner != "moderate" and size_outer != "moderate":
                if size_inner == "small" and size_outer == "massive":
                    return (
                        "smaller in inner volutions and become increasingly massive as the volutions progress"
                    )
                elif size_inner == "massive" and size_outer == "small":
                    return "bigger in inner volutions but become much smaller as the volutions progress"
                else:
                    return "{both} in inner and outer volutions but {mid} in middle volutions".format(
                        both=size_inner, mid=size_mid
                    )
            elif size_inner != "moderate":
                return "{inner} in inner volutions".format(inner=size_inner)
            elif size_outer != "moderate":
                return "{outer} in outer volutions".format(outer=size_outer)
        return ""

    def genChomataHeight(self):
        if len(self.chomata_shapes) <= 0:
            return ""
        chomata_heights_by_volution = {}
        for k in self.chomata_whs_relative:
            chomata_heights_by_volution[k] = self.chomata_whs_relative[k][1]
        a = list(chomata_heights_by_volution.keys())
        a.sort()
        weight_list = [chomata_heights_by_volution[i] for i in a]
        height_inner = self.standardRangeFilter(chomata_height_classes, weight_list[0])
        height_outer = self.standardRangeFilter(chomata_height_classes, weight_list[-1])
        height_mid = -1
        if len(weight_list) >= 3:
            height_mid = self.standardRangeFilter(chomata_height_classes, weight_list[len(weight_list) // 2])
        if height_mid == -1:
            if height_inner == height_outer:
                return height_inner
            elif height_inner != "moderate" and height_outer != "moderate":
                return "{inner} in inner volutions and {outer} in outer volutions".format(
                    inner=height_inner, outer=height_outer
                )
            elif height_inner != "moderate":
                return "{inner} in inner volutions".format(inner=height_inner)
            elif height_outer != "moderate":
                return "{outer} in outer volutions".format(outer=height_outer)
        else:
            if height_outer == height_inner and height_inner == height_mid:
                return height_inner
            elif height_inner == height_outer:
                if height_mid == "moderate":
                    return "{both} in inner and outer volutions".format(both=height_inner)
                else:
                    return "{both} in inner and outer volutions but {mid} in middle volutions".format(
                        both=height_inner, mid=height_mid
                    )
            elif height_inner != "moderate" and height_outer != "moderate":
                if height_inner == "low" and height_outer == "high":
                    return "lower in inner volutions and become increasingly high as the volutions progress"
                elif height_inner == "high" and height_outer == "low":
                    return "higher in inner volutions but become much lower as the volutions progress"
                else:
                    return "{both} in inner and outer volutions but {mid} in middle volutions".format(
                        both=height_inner, mid=height_mid
                    )
            elif height_inner != "moderate":
                return "{inner} in inner volutions".format(inner=height_inner)
            elif height_outer != "moderate":
                return "{outer} in outer volutions".format(outer=height_outer)
        return ""

    def genChomataWidth(self):
        if len(self.chomata_shapes) <= 0:
            return ""
        chomata_widths_by_volution = {}
        for k in self.chomata_whs_relative:
            chomata_widths_by_volution[k] = self.chomata_whs_relative[k][0]
        a = list(chomata_widths_by_volution.keys())
        a.sort()
        weight_list = [chomata_widths_by_volution[i] for i in a]
        width_inner = self.standardRangeFilter(chomata_width_classes, weight_list[0])
        width_outer = self.standardRangeFilter(chomata_width_classes, weight_list[-1])
        width_mid = -1
        if len(weight_list) >= 3:
            width_mid = self.standardRangeFilter(chomata_width_classes, weight_list[len(weight_list) // 2])
        if width_mid == -1:
            if width_inner == width_outer:
                return width_inner
            elif width_inner != "moderate" and width_outer != "moderate":
                return "{inner} in inner volutions and {outer} in outer volutions".format(
                    inner=width_inner, outer=width_outer
                )
            elif width_inner != "moderate":
                return "{inner} in inner volutions".format(inner=width_inner)
            elif width_outer != "moderate":
                return "{outer} in outer volutions".format(outer=width_outer)
        else:
            if width_outer == width_inner and width_inner == width_mid:
                return width_inner
            elif width_inner == width_outer:
                if width_mid == "moderate":
                    return "{both} in inner and outer volutions".format(both=width_inner)
                else:
                    return "{both} in inner and outer volutions but {mid} in middle volutions".format(
                        both=width_inner, mid=width_mid
                    )
            elif width_inner != "moderate" and width_outer != "moderate":
                if width_inner == "narrow" and width_outer == "broad":
                    return (
                        "narrower in inner volutions but become increasingly broad as the volutions progress"
                    )
                elif width_inner == "broad" and width_outer == "narrow":
                    return "broader in inner volutions but become much narrower as the volutions progress"
                else:
                    return "{both} in inner and outer volutions but {mid} in middle volutions".format(
                        both=width_inner, mid=width_mid
                    )
            elif width_inner != "moderate":
                return "{inner} in inner volutions".format(inner=width_inner)
            elif width_outer != "moderate":
                return "{outer} in outer volutions".format(outer=width_outer)
        return ""

    def genVolutionHeightsAndWidths(self):
        heights = []
        widths = []
        for i in range(len(self.volutions_table)):
            if self.volutions_table[i]["type"] == "ellipse":
                height = self.volutions_table[i]["minor_axis"] / 2
                width = self.volutions_table[i]["major_axis"]
                heights.append(height)
                widths.append(width)
            elif self.volutions_table[i]["type"] == "curves":
                height = (
                    abs(self.volutions_table[i]["vertices"][1][1] - self.volutions_table[i]["vertices"][3][1])
                    / 2
                )
                heights.append(height)
                width = abs(
                    self.volutions_table[i]["vertices"][0][0] - self.volutions_table[i]["vertices"][2][0]
                )
                widths.append(width)
            elif "fusiform" in self.volutions_table[i]["type"]:
                height = (
                    (self.volutions_table[i]["x_end"] - self.volutions_table[i]["x_start"])
                    / self.volutions_table[i]["ratio"]
                    / 2
                )
                width = self.volutions_table[i]["x_end"] - self.volutions_table[i]["x_start"]
                heights.append(height)
                widths.append(width)
            else:
                txt = "skip"
                break
        # for i in range(len(heights)-1):
        #     heights[i]=heights[i+1]-heights[i]
        self.volution_heights = heights
        self.volution_widths = widths

    def genChomataDevelopment(self) -> str:
        return self.standardRangeFilter(
            chomata_development_classes, len(self.chomata_shapes) / self.volution_num
        )

    def genUserInput(self):
        txt = "Chomata {block}. ".format(
            block=self.combineFeaturesPlus(
                {
                    "size": self.genChomataSize(),
                    "height": self.genChomataHeight(),
                    "width": self.genChomataWidth(),
                    "": self.genChomataDevelopment(),
                },
                prefix_cond="moderate",
            )
        )
        return txt
