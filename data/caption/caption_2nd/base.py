import math

from scipy import special


class BaseFeature:
    def __init__(self):
        pass

    def standardRangeFilter(self, classes, num):
        for k in classes.keys():
            if num >= classes[k][0] and num <= classes[k][1]:
                return k
        return ""

    def overridedLambdaFilter(self, classes, data):
        clazz = ""
        for k in classes.keys():
            if classes[k](data):
                clazz = k
        return clazz

    def combineFeatures(self, feat_list):
        txt = ""
        feat_list_filtered = [feat for feat in feat_list if feat != ""]
        for feat in feat_list_filtered:
            if txt != "":
                txt += ", "
            txt += feat
        return txt

    def combineFeaturesPlus(self, feat_dict, prefix_cond=None):
        txt = ""
        feat_dict_filtered = {}
        for k in feat_dict:
            if feat_dict[k] != "":
                feat_dict_filtered[k] = feat_dict[k]
        for feat in feat_dict_filtered:
            if txt != "":
                txt += ", "
            if prefix_cond is None:
                txt += (feat + " " + feat_dict_filtered[feat]).strip()
            elif feat_dict_filtered[feat].startswith(prefix_cond):
                txt += (feat + " " + feat_dict_filtered[feat]).strip()
            else:
                txt += (feat_dict_filtered[feat]).strip()
        return txt

    def getCenterAndWeight(self, shapez, n_digits=4):
        def euc_dist(p1, p2):
            return math.sqrt(abs((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))

        def points_center(points):
            x = 0
            y = 0
            for point in points:
                x += point[0]
                y += point[1]
            return [x / len(points), y / len(points)]

        def calculate_polygon_C(coords):
            C = 0
            for i in range(len(coords)):
                C += euc_dist(coords[i - 1], coords[i])
            return C

        def calculate_ellipse_C(a, b):
            e_sq = 1.0 - b**2 / a**2
            C = 4 * a * special.ellipe(e_sq)
            return C

        def calculate_spiral_C(initial_radius, growth_rate, max_theta, n_sample):
            a = initial_radius
            b = growth_rate
            points = []
            start_theta = max_theta - 2 * math.pi
            for i in range(n_sample):
                theta = start_theta + 2 * math.pi * i / n_sample
                r = a + b * theta
                points.append([r * math.cos(theta), r * math.sin(theta)])
            return calculate_polygon_C(points)

        shape = shapez
        if "concentric" in shapez.keys():
            shape = list(shapez["concentric"].values())[-1]
        if shape["type"] in ["segment", "polygon"]:
            p_center = points_center(shape["points"])
            return [
                round(p_center[0], ndigits=n_digits),
                round(p_center[1], ndigits=n_digits),
            ], calculate_polygon_C(shape["points"])
        elif shape["type"] in ["spiral"]:
            return [
                round(shape["center"][0], ndigits=n_digits),
                round(shape["center"][1], ndigits=n_digits),
            ], calculate_spiral_C(shape["initial_radius"], shape["growth_rate"], shape["max_theta"], 360)
        elif shape["type"] in ["ellipse"] or "fusiform" in shape["type"]:
            return [
                round(shape["center"][0], ndigits=n_digits),
                round(shape["center"][1], ndigits=n_digits),
            ], calculate_ellipse_C(shape["major_axis"], shape["minor_axis"])
        elif shape["type"] in ["segment", "line", "ray"]:
            p_center = points_center(shape["points"])
            return [round(p_center[0], ndigits=n_digits), round(p_center[1], ndigits=n_digits)], euc_dist(
                shape["points"][0], shape["points"][1]
            )
        return [0, 0], 0

    def calculate_angle(self, a, top, b):
        point_a = a
        point_b = top
        point_c = b

        a_x, b_x, c_x = point_a[0], point_b[0], point_c[0]
        a_y, b_y, c_y = point_a[1], point_b[1], point_c[1]

        if len(point_a) == len(point_b) == len(point_c) == 3:
            a_z, b_z, c_z = point_a[2], point_b[2], point_c[2]
        else:
            a_z, b_z, c_z = 0, 0, 0
        x1, y1, z1 = (a_x - b_x), (a_y - b_y), (a_z - b_z)
        x2, y2, z2 = (c_x - b_x), (c_y - b_y), (c_z - b_z)
        cos_b = (x1 * x2 + y1 * y2 + z1 * z2) / (
            math.sqrt(x1**2 + y1**2 + z1**2) * (math.sqrt(x2**2 + y2**2 + z2**2))
        )
        B = math.degrees(math.acos(round(cos_b, 8)))
        return B

    def convex_or_concave(self, top, a, b):
        vec1 = (b[0] - a[0], b[1] - a[1])
        vec2 = (top[0] - a[0], top[1] - a[1])
        cross = vec1[0] * vec2[1] - vec2[0] * vec1[1]
        if cross > 0:
            return 1
        elif cross < 0:
            return -1
        else:
            return 0

    def genInput(self):
        return ""
