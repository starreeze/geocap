from data.caption.caption_2nd.params import *
from data.caption.caption_2nd.base import BaseFeature
from scipy.signal import find_peaks


class Chomata(BaseFeature):
    def __init__(self, chomata_shapes, volution_num, volutions_table):
        self.chomata_shapes = chomata_shapes
        self.volution_num = volution_num
        self.volutions_table = volutions_table
        self.genVolutionHeights()

    def genChomataSize(self) -> str:
        if len(self.chomata_shapes) <= 0:
            return ""
        total = 0
        chomata_sizes_by_volution = {}
        for c in self.chomata_shapes:
            vo = int(c["special_info"][len("chomata of volution ") :])
            if vo not in chomata_sizes_by_volution:
                chomata_sizes_by_volution[vo] = []
            center, weight = self.getCenterAndWeight(c)
            try:
                chomata_sizes_by_volution[vo].append(weight / self.volution_heights[vo])
            except:
                print(self.chomata_shapes)
                print(self.volution_heights)
        for k in chomata_sizes_by_volution.keys():
            chomata_sizes_by_volution[k] = sum(chomata_sizes_by_volution[k]) / len(
                chomata_sizes_by_volution[k]
            )
        a = list(chomata_sizes_by_volution.keys())
        a.sort()
        weight_list = [chomata_sizes_by_volution[i] for i in a]
        peaks, _ = find_peaks(
            weight_list, height=chomata_size_classes["massive"][0] - chomata_size_classes["small"][1]
        )
        size_inner = self.standardRangeFilter(chomata_size_classes, weight_list[0])
        size_outer = self.standardRangeFilter(chomata_size_classes, weight_list[-1])
        size_mid = -1
        if len(peaks) > 0:
            size_mid = self.standardRangeFilter(chomata_size_classes, weight_list[peaks[0]])
        if size_mid == -1:
            if size_inner == size_outer:
                return size_inner
            elif size_inner != "" and size_outer != "":
                return "{inner} in inner volutions and {outer} in outer volutions".format(
                    inner=size_inner, outer=size_outer
                )
            elif size_inner != "":
                return "{inner} in inner volutions".format(inner=size_inner)
            elif size_outer != "":
                return "{outer} in outer volutions".format(outer=size_outer)
        else:
            if size_outer == size_inner and size_inner == size_mid:
                return size_inner
            elif size_inner == size_outer and size_outer != "":
                if size_mid == "":
                    return "{both} in inner and outer volutions".format(both=size_inner)
                else:
                    return "{both} in inner and outer volutions but {mid} in middle volutions".format(
                        both=size_inner, mid=size_mid
                    )
            elif size_inner != "" and size_outer != "":
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
            elif size_inner != "":
                return "{inner} in inner volutions".format(inner=size_inner)
            elif size_outer != "":
                return "{outer} in outer volutions".format(outer=size_outer)
        return ""

    def genVolutionHeights(self):
        heights = []
        for i in range(len(self.volutions_table)):
            if self.volutions_table[i]["type"] == "ellipse":
                height = self.volutions_table[i]["minor_axis"] / 2
                heights.append(height)
            elif self.volutions_table[i]["type"] == "curves":
                height = (
                    abs(self.volutions_table[i]["vertices"][1][1] - self.volutions_table[i]["vertices"][3][1])
                    / 2
                )
                heights.append(height)
            elif "fusiform" in self.volutions_table[i]["type"]:
                height = (
                    (self.volutions_table[i]["x_end"] - self.volutions_table[i]["x_start"])
                    / self.volutions_table[i]["ratio"]
                    / 2
                )
                heights.append(height)
            else:
                txt = "skip"
                break
        # for i in range(len(heights)-1):
        #     heights[i]=heights[i+1]-heights[i]
        self.volution_heights = heights

    def genChomataDevelopment(self) -> str:
        return self.standardRangeFilter(
            chomata_development_classes, len(self.chomata_shapes) / self.volution_num
        )

    def genUserInput(self):
        txt = "Chomata {block}. ".format(
            block=self.combineFeatures([self.genChomataSize(), self.genChomataDevelopment()])
        )
        return txt
