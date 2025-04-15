from data.caption.caption_2nd.params import *
from data.caption.caption_2nd.base import BaseFeature


class Shell(BaseFeature):
    def __init__(self, type, ratio, length, width, curves, vertices, center):
        self.type = type
        self.ratio = round(ratio, 1)
        self.w = length
        self.h = width
        self.length = "{:.2f}".format(round(length * shell_world_pixel / shell_pixel_div_mm, 2))
        self.width = "{:.2f}".format(round(width * shell_world_pixel / shell_pixel_div_mm, 2))
        self.length_width_ratio = "{:.2f}".format(round(length / width, 2))
        self.curves = curves
        self.vertices = vertices
        self.center = center

    def getShape(self):
        type = self.type
        ratio = self.ratio
        txt = ""
        txt += self.standardRangeFilter(
            shell_size_classes,
            (self.w * shell_world_pixel / shell_pixel_div_mm)
            * (self.h * shell_world_pixel / shell_pixel_div_mm),
        )
        txt += " "
        if "fusiform" in type:
            txt += self.standardRangeFilter(fusiform_classes, ratio)
        elif "ellipse" in type:
            txt += self.standardRangeFilter(ellipse_classes, ratio)
        elif "curves" in type:
            txt += self.standardRangeFilter(ellipse_classes, ratio)
        equ = self.getEquator()
        if equ != "":
            txt = equ + " in the median part, " + txt  # type: ignore
        return txt.strip()

    def getEquator(self):
        if "curves" in self.type:
            a1 = self.curves[0][2]
            top1 = self.vertices[1]
            b1 = self.curves[1][1]
            a2 = self.curves[2][2]
            top2 = self.vertices[3]
            b2 = self.curves[3][1]
            angle1 = self.calculate_angle(a1, top1, b1)
            angle2 = self.calculate_angle(a2, top2, b2)
            equator1 = self.standardRangeFilter(shell_equator_classes, angle1)
            equator2 = self.standardRangeFilter(shell_equator_classes, angle2)
            if equator1 == equator2:
                return equator1
            elif equator1 == "inflated":
                return "top inflated"
            elif equator2 == "inflated":
                return "bottom inflated"
        elif "ellipse" in self.type:
            return self.standardRangeFilter(shell_equator_classes, self.ratio * 20)
        elif "fusiform" in self.type:
            return ""

    def getSlope(self):
        if "curves" in self.type:
            return self.standardRangeFilter(shell_slope_classes, 0.4)
        elif "ellipse" in self.type:
            return self.standardRangeFilter(shell_slope_classes, 0.4)
        elif "fusiform" in self.type:
            return self.standardRangeFilter(shell_slope_classes, 0.6)

    def getPole(self):
        if "curves" in self.type:
            a1 = self.curves[0][1]
            top1 = self.vertices[0]
            b1 = self.curves[3][2]
            a2 = self.curves[1][2]
            top2 = self.vertices[2]
            b2 = self.curves[2][1]
            angle1 = self.calculate_angle(a1, top1, b1)
            angle2 = self.calculate_angle(a2, top2, b2)
            return self.standardRangeFilter(shell_pole_classes, (angle1 + angle2) / 2)
        elif "ellipse" in self.type:
            return self.standardRangeFilter(shell_pole_classes, 998)
        elif "fusiform" in self.type:
            return self.standardRangeFilter(shell_fusiform_pole_classes, self.ratio)

    def getAxis(self):
        if "curves" in self.type:
            top = self.center
            a = self.vertices[0]
            b = self.vertices[2]
            cross = self.convex_or_concave(top, a, b)
            angle = self.calculate_angle(a, top, b)
            axis = self.standardRangeFilter(shell_axis_classes, angle)
            if axis == "other":
                if cross > 0:
                    return "convex"
                elif cross < 0:
                    return "concave"
            else:
                return axis
        elif "ellipse" in self.type:
            return "straight"
        elif "fusiform" in self.type:
            return "straight"

    def genUserInput(self):
        tagged = []
        txt = "<shell>Shell {shape}, ".format(shape=self.getShape())
        txt += "with {slope} slopes and {pole} ends. </shell>".format(
            slope=self.getSlope(), pole=self.getPole()
        )
        tagged.append(txt)
        txt = "<length>The axial length is {length} mm, </length>".format(length=self.length)
        tagged.append(txt)
        txt = "<width>and the sagittal width is {width} mm, </width>".format(width=self.width)
        tagged.append(txt)
        txt = "<ratio>width a ratio of length to width of {ratio}. </ratio>".format(
            ratio=self.length_width_ratio
        )
        tagged.append(txt)
        txt = "<axis>Axis {convexity}. </axis>".format(convexity=self.getAxis())
        tagged.append(txt)
        return tagged

    def genInput(self):
        txt = "length: {length} mm, ".format(length=self.length)
        txt += "width: {width} mm, ".format(width=self.width)
        txt += "ratio: {ratio}\n".format(ratio=self.length_width_ratio)
        return txt
