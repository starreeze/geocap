import math

import numpy as np

from data.caption.caption_2nd.base import BaseFeature
from data.caption.caption_2nd.params import *


class Septa(BaseFeature):
    def __init__(self, septa_folds, volution_sizes):
        self.septa_folds = septa_folds
        self.txt = ""
        self.txt2 = ""
        self.volution_sizes = volution_sizes
        self.refineSeptaFolds()
        # self.genSeptaNum()
        self.genSeptaDescription()

    def quadrilateral_area(self, control_points):
        # 解包坐标点
        p1, p2, p3, p4 = control_points
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        # 应用鞋带公式
        area = 0.5 * abs((x1 * y2 + x2 * y3 + x3 * y4 + x4 * y1) - (y1 * x2 + y2 * x3 + y3 * x4 + y4 * x1))
        return area / 4

    def curves_center(self, control_points):
        x, y = 0, 0
        for p in control_points:
            x += p[0]
            y += p[1]
        return [x / 4, y / 4]

    def mean_absolute_difference(self, arr):
        if len(arr) < 2:
            return 0.0  # 如果数组元素少于2个，返回0

        # 对数组排序
        arr.sort()
        n = len(arr)
        total = 0.0

        # 使用数学优化计算总和：每个元素的贡献为 arr[i] * (2*i - n + 1)
        for i, num in enumerate(arr):
            total += num * (2 * i - n + 1)

        # 计算组合数
        num_pairs = n * (n - 1) // 2

        # 返回平均差值（总和除以组合数）
        return total / num_pairs

    def refineSeptaFolds(self):
        self.septa_pos = {}
        self.septa_size = {}
        refined_septa_folds = {}
        current_volution = 1
        for septa in self.septa_folds:
            if "special_info" in septa:
                current_volution = int(septa["special_info"][len("septa of volution ") :])
            if current_volution not in refined_septa_folds:
                refined_septa_folds[current_volution] = {}
            if septa["type"] not in refined_septa_folds[current_volution]:
                refined_septa_folds[current_volution][septa["type"]] = []
            refined_septa_folds[current_volution][septa["type"]].append(septa)

            if current_volution not in self.septa_pos:
                self.septa_pos[current_volution] = []
            if current_volution not in self.septa_size:
                self.septa_size[current_volution] = []

            if septa["type"] == "ellipse":
                self.septa_pos[current_volution].append(septa["center"])
                self.septa_size[current_volution].append(
                    math.pi
                    * septa["major_axis"]
                    * septa["minor_axis"]
                    / self.volution_sizes[current_volution]
                    / 4
                )
            elif septa["type"] == "curves":
                self.septa_pos[current_volution].append(self.curves_center(septa["control_points"]))
                self.septa_size[current_volution].append(
                    self.quadrilateral_area(septa["control_points"]) / self.volution_sizes[current_volution]
                )

        filtered = []
        filtered_inner = []
        filtered_outer = []
        for k in self.septa_size:
            if k <= len(refined_septa_folds) / 2:
                filtered_inner.append(self.septa_size[k])
            else:
                filtered_outer.append(self.septa_size[k])
        for size in self.septa_size.values():
            if len(size) > 0:
                filtered.append(size)
        if len(filtered) <= 0:
            self.size_diff = 0
        else:
            self.size_diff = np.average([self.mean_absolute_difference(size) for size in filtered])
        if len(filtered_inner) <= 0:
            self.size_diff_inner = 0
        else:
            self.size_diff_inner = np.average(
                [self.mean_absolute_difference(size) for size in filtered_inner]
            )
        if len(filtered_outer) <= 0:
            self.size_diff_outer = 0
        else:
            self.size_diff_outer = np.average(
                [self.mean_absolute_difference(size) for size in filtered_outer]
            )
        self.refined_septa_folds = refined_septa_folds

    def genSeptaNum(self):
        if len(self.septa_folds) == 0:
            return
        txt = "The septal counts for the {start} to {end} volution are ".format(
            start=ordinal_numbers[min(self.refined_septa_folds.keys())],
            end=ordinal_numbers[max(self.refined_septa_folds.keys())],
        )
        txt2 = "the septal counts: "
        idx = 0
        for i in self.refined_septa_folds:
            total = 0
            for v in self.refined_septa_folds[i].values():
                total += len(v)
            if idx != len(self.refined_septa_folds) - 1:
                txt += str(total)
                txt += ", "
                txt2 += str(total)
                txt2 += ", "
            else:
                txt += "and {n}, respectively. ".format(n=total)
                txt2 += str(total)
                txt2 += ".\n"
            idx += 1
        self.txt += txt
        self.txt2 += txt2

    def genSeptaDescription(self):
        if len(self.septa_folds) == 0:
            return
        inner = [0, 0, 0]
        outer = [0, 0, 0]
        tags = ["polygon", "ellipse", "curves"]
        for k in self.refined_septa_folds:
            if k <= len(self.refined_septa_folds) / 2:
                for i in range(len(tags)):
                    if tags[i] in self.refined_septa_folds[k]:
                        inner[i] += len(self.refined_septa_folds[k][tags[i]])
            else:
                for i in range(len(tags)):
                    if tags[i] in self.refined_septa_folds[k]:
                        outer[i] += len(self.refined_septa_folds[k][tags[i]])
        inner_r = [sum(inner) / (len(self.refined_septa_folds) / 2)]
        outer_r = [sum(outer) / (len(self.refined_septa_folds) / 2)]
        adj1 = self.overridedLambdaFilter(septa_shape_classes, inner_r)
        adj2 = self.overridedLambdaFilter(septa_shape_classes, outer_r)
        adj1_1 = self.overridedLambdaFilter(septa_size_difference_classes, self.size_diff_inner)
        adj2_1 = self.overridedLambdaFilter(septa_size_difference_classes, self.size_diff_outer)
        adj1 = self.overridedDescriptionByPriority(septa_shape_priority, [adj1, adj1_1])
        adj2 = self.overridedDescriptionByPriority(septa_shape_priority, [adj2, adj2_1])
        if adj1 != adj2:
            txt = "Septa {adj1} in the inner whorls and {adj2} in the outer whorls. ".format(
                adj1=adj1, adj2=adj2
            )
        else:
            txt = "Septa {adj}. ".format(adj=adj1)
        self.txt += txt

    def genUserInput(self):
        if self.txt == "":
            self.txt = "Septa straight. "
        return [f"<septa>{self.txt}</septa>"]

    def genInput(self):
        return self.txt2
