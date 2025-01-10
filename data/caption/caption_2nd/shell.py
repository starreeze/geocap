from data.caption.caption_2nd.params import *
from data.caption.caption_2nd.base import BaseFeature


class Shell(BaseFeature):
    def __init__(self, type, ratio, length, width, curves, vertices):
        self.type = type
        self.ratio = round(ratio, 1)
        self.length = "{:.2f}".format(round(length, 2))
        self.width = "{:.2f}".format(round(width, 2))
        self.length_width_ratio = "{:.2f}".format(round(length / width, 2))
        self.curves = curves
        self.vertices = vertices

    def getShape(self):
        type = self.type
        ratio = self.ratio
        txt = ""
        if "fusiform" in type:
            txt = self.standardRangeFilter(fusiform_classes, ratio)
        elif "ellipse" in type:
            txt = self.standardRangeFilter(ellipse_classes, ratio)
        equ = self.getEquator()
        if equ != "":
            txt = equ + " " + txt  # type: ignore
        return txt

    def getEquator(self):
        if len(self.curves) != 4:
            return ""
        a1 = self.curves[0][2]
        top1 = self.vertices[1]
        b1 = self.curves[1][1]
        a2 = self.curves[2][2]
        top2 = self.vertices[3]
        b2 = self.curves[3][1]
        angle1 = self.calculate_angle(a1, top1, b1)
        angle2 = self.calculate_angle(a2, top2, b2)
        return self.standardRangeFilter(shell_equator_classes, (angle1 + angle2) / 2)

    def getSlope(self):
        # TODO: 侧坡凹凸性
        return self.standardRangeFilter(shell_slope_classes, 0.4)

    def getPole(self):
        if len(self.curves) != 4:
            return "bluntly rounded"
        a1 = self.curves[0][1]
        top1 = self.vertices[0]
        b1 = self.curves[3][2]
        a2 = self.curves[1][2]
        top2 = self.vertices[2]
        b2 = self.curves[2][1]
        angle1 = self.calculate_angle(a1, top1, b1)
        angle2 = self.calculate_angle(a2, top2, b2)
        return self.standardRangeFilter(shell_pole_classes, (angle1 + angle2) / 2)

    def genUserInput(self):
        txt = "Shell {shape}, ".format(shape=self.getShape())
        txt += "with {slope} slopes and {pole} ends, ".format(slope=self.getSlope(), pole=self.getPole())
        txt += "the axial length is {length} mm, ".format(length=self.length)
        txt += "and the sagittal width is {width} mm, ".format(width=self.width)
        txt += "width a ratio of length to width of {ratio}. ".format(ratio=self.length_width_ratio)
        return txt
