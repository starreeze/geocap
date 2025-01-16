from data.caption.caption_2nd.params import *
from data.caption.caption_2nd.base import BaseFeature


class Proloculus(BaseFeature):
    def __init__(self, type, shape, diameter):
        self.type = type
        self.shape = shape
        self.diameter = round(diameter, 2)

    def getShape(self):
        type = self.type
        txt = ""
        center, weight = self.getCenterAndWeight(self.shape)
        txt += self.standardRangeFilter(proloculus_size_classes, weight)
        txt += " "
        # TODO: 肾形
        txt += self.standardRangeFilter(proloculus_shape_classes, 0.1)
        return txt.strip()

    def genUserInput(self):
        txt = "Proloculus {shape}, ".format(shape=self.getShape())
        txt += "with diameter measuring {diameter}. ".format(diameter=self.diameter)
        return txt

    def genInput(self):
        txt = "initial chamber(proloculus): {length} mm\n".format(length=self.diameter)
        return txt
