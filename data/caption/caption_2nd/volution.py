from data.caption.caption_2nd.base import BaseFeature
from data.caption.caption_2nd.params import *


class Volution(BaseFeature):
    def __init__(self, n_volutions, thickness_table=[], volutions_table=[], random_pixel_div_mm_offset=0):
        self.num = n_volutions
        self.thickness_table = thickness_table
        self.volutions_table = volutions_table
        self.chamber_heights = None
        self.growth_rate = None
        self.random_pixel_div_mm_offset = random_pixel_div_mm_offset

    def genWallThickness(self):
        if len(self.thickness_table) == 0:
            return ""
        txt = "Wall thickness "
        thickness_part = "{data} in {nth} whorl"
        for i in range(len(self.thickness_table)):
            txt += thickness_part.format(
                data=self.thickness_table[i][0], nth=ordinal_numbers[self.thickness_table[i][1]]
            )
            if i != len(self.thickness_table) - 1:
                txt += ", "
            else:
                txt += ". "
        return txt

    def genVolutionHeightAndGrowthRate(self):
        txt = (
            "<heights of volutions>Heights of chambers above tunnel in {start} to {end} volution are ".format(
                start=ordinal_numbers[0], end=ordinal_numbers[len(self.volutions_table) - 1]
            )
        )
        txt1 = ""
        txt2 = ""
        heights = []
        for i in range(len(self.volutions_table)):
            if self.volutions_table[i]["type"] == "ellipse":
                height = max(round(self.volutions_table[i]["minor_axis"] / 2, 3), 0.001)
                heights.append(height)
                txt1 += str(height)
            elif self.volutions_table[i]["type"] == "curves":
                height = max(
                    round(
                        abs(
                            self.volutions_table[i]["vertices"][1][1]
                            - self.volutions_table[i]["vertices"][3][1]
                        )
                        / 2
                        * shell_world_pixel
                        / (shell_pixel_div_mm + self.random_pixel_div_mm_offset),
                        3,
                    ),
                    0.001,
                )
                heights.append(height)
                txt1 += str(height)
            else:
                txt = "skip"
                break
            if i != len(self.volutions_table) - 1:
                txt1 += ", "
            else:
                txt1 += " mm. "
        if txt == "skip":
            return ""
        txt += txt1
        txt_final2 = txt + "</heights of volutions>"
        self.chamber_heights = txt1
        self.growth_rate_raw = []
        # txt += "Rates of growth of {start} to {end} volution are ".format(
        #     start=ordinal_numbers[1], end=ordinal_numbers[len(self.volutions_table) - 1]
        # )
        for i in range(len(heights) - 1):
            txt2 += str(round(heights[i + 1] / heights[i], 1))
            self.growth_rate_raw.append(heights[i + 1] - heights[i])
            if i != len(heights) - 2:
                txt2 += ", "
            else:
                txt2 += ". "
        # txt += txt2
        # self.growth_rate = txt2
        self.growth_rate = None
        txt_final1 = f"<coil tightness>Volutions {self.getCoiled()}. </coil tightness>"
        return txt_final1 + txt_final2
        # return txt

    def processTwo(self, inner, outer):
        if "moderate" in inner[0]:
            return f"{outer[0]} in outer volutions"
        elif "moderate" in outer[0]:
            return f"{inner[0]} in first {inner[1] + 2} volutions"
        else:
            return f"{inner[0]} in first {inner[1] + 2} volutions while {outer[0]} in outer volutions"

    def getCoiled(self):
        coiled = [self.standardRangeFilter(volution_coiled_classes, g) for g in self.growth_rate_raw]
        inner, outer = ["", -1], ["", -1]
        current_modify = inner
        # print("new sample:")
        # print(self.chamber_heights)
        for i, word in enumerate(coiled):
            # print(f"{i}th loop:",inner,outer)
            if current_modify[0] == "":
                current_modify[0] = word
                current_modify[1] = i
            elif current_modify[0] != word:
                current_modify = outer
                current_modify[0] = word
                current_modify[1] = i
            else:
                current_modify[1] = i
        if inner[0] != "":
            if outer[0] == "":
                return inner[0]
            elif outer[0] == inner[0]:
                return inner[0]
            else:
                return self.processTwo(inner, outer)
        else:
            return "moderately coiled"

    def genUserInput(self):
        txt = "<number of volutions>The number of volutions is {num}. </number of volutions>".format(
            num=self.num
        )
        txt += self.genVolutionHeightAndGrowthRate()
        # txt+=self.genWallThickness()
        return [f"{txt}"]

    def genInput(self):
        txt = "number of volutions(whorls): {num}\n".format(num=self.num)
        if self.chamber_heights is not None:
            txt += "heights of chambers: {nums}\n".format(nums=self.chamber_heights)
        if self.growth_rate is not None:
            txt += "rates of growth: {nums}\n".format(nums=self.growth_rate)
        return txt
