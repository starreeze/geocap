from data.caption.caption_2nd.params import *
from data.caption.caption_2nd.base import BaseFeature


class Volution(BaseFeature):
    def __init__(self, n_volutions, thickness_table=[]):
        self.num = n_volutions
        self.thickness_table = thickness_table

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

    def genUserInput(self):
        txt = "The number of volutions is {num}. ".format(num=self.num)
        # txt+=self.genWallThickness()
        return txt

    def genInput(self):
        txt = "number of volutions(whorls): {num}\n".format(num=self.num)
        return txt
