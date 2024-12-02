from data.caption.caption_2nd.params import *
from data.caption.caption_2nd.base import BaseFeature


class Chomata(BaseFeature):
    def __init__(self, chomata_shapes, volution_num):
        self.chomata_shapes = chomata_shapes
        self.volution_num = volution_num

    def genChomataSize(self) -> str:
        total = 0
        for c in self.chomata_shapes:
            center, weight = self.getCenterAndWeight(c)
            total += weight
        return self.standardRangeFilter(chomata_size_classes, total / len(self.chomata_shapes))

    def genChomataDevelopment(self) -> str:
        return self.standardRangeFilter(chomata_development_classes, len(self.chomata_shapes) / self.volution_num)

    def genUserInput(self):
        txt = "Chomata {block}. ".format(
            block=self.combineFeatures([self.genChomataSize(), self.genChomataDevelopment()])
        )
        return txt
