from data.caption.caption_2nd.base import BaseFeature
from data.caption.caption_2nd.params import *


class Volution(BaseFeature):
    def __init__(self, n_volutions, thickness_table=[], volutions_table=[]):
        self.num = n_volutions
        self.thickness_table = thickness_table
        self.volutions_table = volutions_table
        self.chamber_heights = None
        self.growth_rate = None

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
        txt = "Heights of chambers above tunnel in {start} to {end} volution are ".format(
            start=ordinal_numbers[0], end=ordinal_numbers[len(self.volutions_table) - 1]
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
                        / shell_pixel_div_mm,
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
        self.chamber_heights = txt1
        txt += "Rates of growth of {start} to {end} volution are ".format(
            start=ordinal_numbers[1], end=ordinal_numbers[len(self.volutions_table) - 1]
        )
        for i in range(len(heights) - 1):
            txt2 += str(round(heights[i + 1] / heights[i], 1))
            if i != len(heights) - 2:
                txt2 += ", "
            else:
                txt2 += ". "
        txt += txt2
        self.growth_rate = txt2
        return txt

    def genUserInput(self):
        txt = "The number of volutions is {num}. ".format(num=self.num)
        txt += self.genVolutionHeightAndGrowthRate()
        # txt+=self.genWallThickness()
        return txt

    def genInput(self):
        txt = "number of volutions(whorls): {num}\n".format(num=self.num)
        if self.chamber_heights != None:
            txt += "heights of chambers: {nums}\n".format(nums=self.chamber_heights)
        if self.growth_rate != None:
            txt += "rates of growth: {nums}\n".format(nums=self.growth_rate)
        return txt
