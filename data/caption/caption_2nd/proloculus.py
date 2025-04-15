import math

from data.caption.caption_2nd.base import BaseFeature
from data.caption.caption_2nd.params import *


class Proloculus(BaseFeature):
    def __init__(self, type, shape, diameter):
        self.type = type
        self.shape = shape
        self.diameter0 = diameter
        self.diameter = round(diameter * shell_world_pixel / shell_pixel_div_mm, 2)

    def getShape(self):
        type = self.type
        txt = ""
        # center, weight = self.getCenterAndWeight(self.shape)
        txt += self.standardRangeFilter(
            proloculus_size_classes, self.diameter0 * shell_world_pixel / shell_pixel_div_mm
        )
        txt += " "
        # TODO: 肾形
        txt += self.standardRangeFilter(proloculus_shape_classes, 0.1)
        return txt.strip()

    def genUserInput(self):
        txt = "Proloculus {shape}, ".format(shape=self.getShape())
        txt += "with diameter measuring {diameter} mm. ".format(diameter=self.diameter)
        return [f"<proloculus>{txt}</proloculus>"]

    def genInput(self):
        txt = "initial chamber(proloculus): {length} mm\n".format(length=self.diameter)
        return txt
