from data.caption.caption_2nd.params import *
from data.caption.caption_2nd.base import BaseFeature
from scipy.signal import find_peaks
from data.rule.shapes import GSRule


class Chomata(BaseFeature):
    def __init__(self, chomata_shapes, volution_num, volutions_table):
        self.chomata_shapes = chomata_shapes
        self.volution_num = volution_num
        self.volutions_table = volutions_table
        self.genVolutionHeightsAndWidths()

    def genChomataSize(self) -> str:
        if len(self.chomata_shapes) <= 0:
            return ""
        total = 0
        chomata_sizes_by_volution = {}
        for c in self.chomata_shapes:
            vo = int(c["special_info"][len("chomata of volution ") :])
            if vo not in chomata_sizes_by_volution:
                chomata_sizes_by_volution[vo] = []
            # center, weight = self.getCenterAndWeight(c)
            chomata_gs = GSRule.from_dict(c)
            weight = chomata_gs.get_area()
            try:
                chomata_sizes_by_volution[vo].append(
                    weight / self.volution_widths[vo] / self.volution_heights[vo]
                )
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

    def genChomataHeight(self):
        if len(self.chomata_shapes) <= 0:
            return ""
        chomata_heights_by_volution = {}
        for c in self.chomata_shapes:
            vo = int(c["special_info"][len("chomata of volution ") :])
            if vo not in chomata_heights_by_volution:
                chomata_heights_by_volution[vo] = []
            chomata_gs = GSRule.from_dict(c)
            c_bbox = chomata_gs.get_bbox()
            try:
                chomata_heights_by_volution[vo].append(
                    abs(c_bbox[0][1] - c_bbox[1][1]) / self.volution_heights[vo]
                )
            except:
                print(self.chomata_shapes)
                print(self.volution_heights)
        for k in chomata_heights_by_volution.keys():
            chomata_heights_by_volution[k] = sum(chomata_heights_by_volution[k]) / len(
                chomata_heights_by_volution[k]
            )
        a = list(chomata_heights_by_volution.keys())
        a.sort()
        weight_list = [chomata_heights_by_volution[i] for i in a]
        peaks, _ = find_peaks(
            weight_list, height=chomata_height_classes["high"][0] - chomata_height_classes["low"][1]
        )
        height_inner = self.standardRangeFilter(chomata_height_classes, weight_list[0])
        height_outer = self.standardRangeFilter(chomata_height_classes, weight_list[-1])
        height_mid = -1
        if len(peaks) > 0:
            height_mid = self.standardRangeFilter(chomata_height_classes, weight_list[peaks[0]])
        if height_mid == -1:
            if height_inner == height_outer:
                return height_inner
            elif height_inner != "" and height_outer != "":
                return "{inner} in inner volutions and {outer} in outer volutions".format(
                    inner=height_inner, outer=height_outer
                )
            elif height_inner != "":
                return "{inner} in inner volutions".format(inner=height_inner)
            elif height_outer != "":
                return "{outer} in outer volutions".format(outer=height_outer)
        else:
            if height_outer == height_inner and height_inner == height_mid:
                return height_inner
            elif height_inner == height_outer and height_outer != "":
                if height_mid == "":
                    return "{both} in inner and outer volutions".format(both=height_inner)
                else:
                    return "{both} in inner and outer volutions but {mid} in middle volutions".format(
                        both=height_inner, mid=height_mid
                    )
            elif height_inner != "" and height_outer != "":
                if height_inner == "small" and height_outer == "massive":
                    return "lower in inner volutions and become increasingly high as the volutions progress"
                elif height_inner == "massive" and height_outer == "small":
                    return "higher in inner volutions but become much lower as the volutions progress"
                else:
                    return "{both} in inner and outer volutions but {mid} in middle volutions".format(
                        both=height_inner, mid=height_mid
                    )
            elif height_inner != "":
                return "{inner} in inner volutions".format(inner=height_inner)
            elif height_outer != "":
                return "{outer} in outer volutions".format(outer=height_outer)
        return ""

    def genChomataWidth(self):
        if len(self.chomata_shapes) <= 0:
            return ""
        chomata_widths_by_volution = {}
        for c in self.chomata_shapes:
            vo = int(c["special_info"][len("chomata of volution ") :])
            if vo not in chomata_widths_by_volution:
                chomata_widths_by_volution[vo] = []
            chomata_gs = GSRule.from_dict(c)
            c_bbox = chomata_gs.get_bbox()
            try:
                chomata_widths_by_volution[vo].append(
                    abs(c_bbox[0][0] - c_bbox[1][0]) / self.volution_widths[vo]
                )
            except:
                print(self.chomata_shapes)
                print(self.volution_widths)
        for k in chomata_widths_by_volution.keys():
            chomata_widths_by_volution[k] = sum(chomata_widths_by_volution[k]) / len(
                chomata_widths_by_volution[k]
            )
        a = list(chomata_widths_by_volution.keys())
        a.sort()
        weight_list = [chomata_widths_by_volution[i] for i in a]
        peaks, _ = find_peaks(
            weight_list, height=chomata_width_classes["broad"][0] - chomata_width_classes["narrow"][1]
        )
        width_inner = self.standardRangeFilter(chomata_width_classes, weight_list[0])
        width_outer = self.standardRangeFilter(chomata_width_classes, weight_list[-1])
        width_mid = -1
        if len(peaks) > 0:
            width_mid = self.standardRangeFilter(chomata_width_classes, weight_list[peaks[0]])
        if width_mid == -1:
            if width_inner == width_outer:
                return width_inner
            elif width_inner != "" and width_outer != "":
                return "{inner} in inner volutions and {outer} in outer volutions".format(
                    inner=width_inner, outer=width_outer
                )
            elif width_inner != "":
                return "{inner} in inner volutions".format(inner=width_inner)
            elif width_outer != "":
                return "{outer} in outer volutions".format(outer=width_outer)
        else:
            if width_outer == width_inner and width_inner == width_mid:
                return width_inner
            elif width_inner == width_outer and width_outer != "":
                if width_mid == "":
                    return "{both} in inner and outer volutions".format(both=width_inner)
                else:
                    return "{both} in inner and outer volutions but {mid} in middle volutions".format(
                        both=width_inner, mid=width_mid
                    )
            elif width_inner != "" and width_outer != "":
                if width_inner == "small" and width_outer == "massive":
                    return (
                        "narrower in inner volutions but become increasingly broad as the volutions progress"
                    )
                elif width_inner == "massive" and width_outer == "small":
                    return "broader in inner volutions but become much narrower as the volutions progress"
                else:
                    return "{both} in inner and outer volutions but {mid} in middle volutions".format(
                        both=width_inner, mid=width_mid
                    )
            elif width_inner != "":
                return "{inner} in inner volutions".format(inner=width_inner)
            elif width_outer != "":
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
                }
            )
        )
        return txt
