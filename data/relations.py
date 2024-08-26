from typing import Any
import numpy as np
from numpy.random import uniform, randint, normal
from .shapes import Polygon, Line, Ellipse
from data.utils import (
    distance_2points,
    polar_angle,
    line_given2points,
    distance_point_to_line,
    another_2points_on_line,
    find_intersection,
    find_symmetric_point,
)


class RelationGenerator:
    def __init__(self, rule_args) -> None:
        # probability for different types of relation
        self.polygon_relation_generator = PolygonRelationGenerator(rule_args)
        self.line_relation_generator = LineRelationGenerator(rule_args)
        self.ellipse_relation_generator = EllipseRelationGenerator(rule_args)

        self.type2generator = {
            "polygon": self.polygon_relation_generator,
            "line": self.line_relation_generator,
            "ellipse": self.ellipse_relation_generator,
        }

    def __call__(self, shape: Polygon | Line | Ellipse) -> tuple[Any, str]:
        shape_type = shape.to_dict()["type"]
        if shape_type in ["line", "ray", "segment"]:
            shape_type = "line"
        generator = self.type2generator[shape_type]
        new_shape, relation_type = generator.generate_relation(shape)

        return new_shape, relation_type


class PolygonRelationGenerator:
    def __init__(self, rule_args) -> None:
        self.rule_args = rule_args

        self.base_rel = ["tangent line", "symmetric", "similar"]
        self.triangle_only_rel = ["shared edge", "circumscribed circle of triangle", "inscribed circle"]
        self.rectangle_only_rel = ["circumscribed circle of rectangle", "diagonal"]
        all_relations = self.base_rel + self.triangle_only_rel + self.rectangle_only_rel

        self.relation_level = []
        for relation_type in all_relations:
            level = eval(f"rule_args.polygon_{relation_type.replace(' ', '_')}_level")
            self.relation_level.append(level)

    def generate_relation(self, polygon: Polygon) -> tuple[Any, str]:
        self.polygon = polygon
        opt_rel = self.base_rel
        if "triangle" in self.polygon.special_info:
            opt_rel = opt_rel + self.triangle_only_rel
        elif "rectangle" in self.polygon.special_info:
            opt_rel = opt_rel + self.rectangle_only_rel

        opt_relation_level = self.relation_level[: len(opt_rel)]
        relation_prob = [x / sum(opt_relation_level) for x in opt_relation_level]
        relation_type = np.random.choice(opt_rel, p=relation_prob)
        if relation_type == "tangent line":
            new_shape = self.get_tangent_line()

        elif "symmetric" in relation_type:
            new_shape = self.get_symmetric_polygon()

        elif "similar" in relation_type:
            new_shape = self.get_similar_polygon()

        elif "shared edge" in relation_type:
            new_shape = self.get_share_edge_polygon()

        elif "circumscribed circle" in relation_type:
            new_shape = self.get_circumscribed_circle()

        elif "inscribed circle" in relation_type:
            new_shape = self.get_inscribed_circle()

        elif "diagonal" in relation_type:
            new_shape = self.get_diagonal()

        return new_shape, relation_type

    def get_tangent_line(self) -> Line:
        # choose a vertex randomly
        n = len(self.polygon.points)
        index = randint(0, n)
        x, y = self.polygon.points[index]
        # calculate tangent line
        prev_point = self.polygon.points[(index - 1) % n]
        next_point = self.polygon.points[(index + 1) % n]
        slope, _ = line_given2points([prev_point, next_point])
        intercept = y - slope * x

        p1, p2 = another_2points_on_line(line=(slope, intercept), point=(x, y))
        line = Line(type="segment", points=[p1, p2])
        return line

    def get_symmetric_polygon(self) -> Polygon | None:
        points = self.polygon.points
        idx = randint(0, len(points))
        axial = line_given2points([points[idx], points[(idx + 1) % len(points)]])
        new_points = [find_symmetric_point(axial, point) for point in points]
        new_polygon = Polygon(new_points, self.polygon.special_info)
        return new_polygon

    def get_similar_polygon(self) -> Polygon:
        scale_factor = uniform(0.5, 1.0)
        scaled_points = [(x * scale_factor, y * scale_factor) for (x, y) in self.polygon.points]

        x_noise, y_noise = uniform(-0.3, 0.3), uniform(-0.3, 0.3)
        points = [(x + x_noise, y + y_noise) for (x, y) in scaled_points]

        new_polygon = Polygon(points, self.polygon.special_info)
        new_polygon.normalize_points()
        return new_polygon

    def get_share_edge_polygon(self) -> list[Polygon]:
        assert "triangle" in self.polygon.special_info and len(self.polygon.points) == 3

        points = self.polygon.points
        polygon_list = []
        num_new_polygons = randint(1, 3)
        for _ in range(num_new_polygons):
            side_len = distance_2points(points[0], points[1])  # length of the shared edge
            mid_point = ((points[0][0] + points[1][0]) * 0.5, (points[0][1] + points[1][1]) * 0.5)
            new_point = (mid_point[0] + side_len * uniform(-0.5, 0.5), mid_point[1] + side_len * uniform(-0.5, 0.5))
            polygon_list.append(Polygon([points[0], points[1], new_point], special_info="triangle"))

        return polygon_list

    def get_circumscribed_circle(self) -> Ellipse:
        if "triangle" in self.polygon.special_info:
            assert len(self.polygon.points) == 3
            x1, y1 = self.polygon.points[0]
            x2, y2 = self.polygon.points[1]
            x3, y3 = self.polygon.points[2]

            # calculate center by solving linear system Ax=B
            A = np.array([[x1 - x2, y1 - y2], [x1 - x3, y1 - y3]])
            B = np.array([(x1**2 - x2**2 + y1**2 - y2**2) / 2, (x1**2 - x3**2 + y1**2 - y3**2) / 2])
            center = tuple(np.linalg.solve(A, B))
            radius = distance_2points(center, (x1, y1))

            circle = Ellipse(center=center)
            circle.to_circle(radius)

        elif "rectangle" in self.polygon.special_info:
            center = (
                sum(x for x, y in self.polygon.points) / 4,
                sum(y for x, y in self.polygon.points) / 4,
            )
            radius = distance_2points(center, self.polygon.points[0])
            circle = Ellipse(center=center)
            circle.to_circle(radius)

        else:
            raise ValueError("The polygon is not a triangle or rectangle.")

        return circle

    def get_inscribed_circle(self) -> Ellipse:
        assert "triangle" in self.polygon.special_info and len(self.polygon.points) == 3

        A, B, C = self.polygon.points
        a = distance_2points(B, C)
        b = distance_2points(A, C)
        c = distance_2points(A, B)

        # Calculate the area of the triangle using Heron's formula
        s = (a + b + c) / 2
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))

        # Calculate the incenter coordinates
        ix = (a * A[0] + b * B[0] + c * C[0]) / (a + b + c)
        iy = (a * A[1] + b * B[1] + c * C[1]) / (a + b + c)

        # Calculate the radius of the incircle
        radius = area / s

        circle = Ellipse(center=(ix, iy))
        circle.to_circle(radius)
        return circle

    def get_diagonal(self) -> Line:
        assert "rectangle" in self.polygon.special_info
        idx = randint(0, 2)
        p1 = self.polygon.points[idx]
        p2 = self.polygon.points[idx + 2]
        line = Line(type="segment", points=[p1, p2])
        return line


class LineRelationGenerator:
    def __init__(self, rule_args) -> None:
        self.rule_args = rule_args

        self.base_rel = ["parallel", "tangent line", "axis of ellipse"]

        self.relation_level = []
        for relation_type in self.base_rel:
            level = eval(f"rule_args.line_{relation_type.replace(' ', '_')}_level")
            self.relation_level.append(level)

        self.relation_prob = [x / sum(self.relation_level) for x in self.relation_level]

    def generate_relation(self, line: Line) -> tuple[Any, str]:
        self.line = line

        relation_type = np.random.choice(self.base_rel, p=self.relation_prob)

        if relation_type == "parallel":
            new_shape = self.get_parallel_line()

        elif relation_type == "tangent line":
            new_shape = self.get_tangent_circle()

        elif relation_type == "axis of ellipse":
            new_shape, relation_type = self.get_axis_ellipse()

        return new_shape, relation_type

    def get_parallel_line(self) -> Line:
        k, _ = line_given2points(self.line.points)

        p = (uniform(0, 1), uniform(0, 1))
        b = p[1] - k * p[0]

        points = another_2points_on_line(line=(k, b), point=p)
        line = Line(type=self.line.type, points=points)
        return line

    def get_tangent_circle(self) -> Ellipse:
        points = self.line.points
        mid_point = ((points[0][0] + points[1][0]) * 0.5, (points[0][1] + points[1][1]) * 0.5)
        center = (mid_point[0] + uniform(-0.3, 0.3), mid_point[1] + uniform(-0.3, 0.3))

        k, b = line_given2points(self.line.points)
        distance = distance_point_to_line(point=center, line=(k, b))
        while distance > 0.3:
            center = (uniform(0, 1), uniform(0, 1))
            distance = distance_point_to_line(point=center, line=(k, b))

        circle = Ellipse(center=center, rotation=0)
        circle.to_circle(radius=distance)
        return circle

    def get_axis_ellipse(self, special_axis=None) -> tuple[Ellipse, str]:
        p1, p2 = self.line.points
        center = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        angle = polar_angle(p1, p2)

        if special_axis is None:
            special_axis = np.random.choice(["major axis", "minor axis", "diameter"])

        special_info = ""
        if special_axis == "major axis":
            major_axis = self.line.get_length()
            minor_axis = uniform(0.3 * major_axis, major_axis)
            rotation = angle
        elif special_axis == "minor axis":
            minor_axis = self.line.get_length()
            major_axis = uniform(minor_axis, 1.8 * minor_axis)
            rotation = angle + np.pi / 2
        elif special_axis == "diameter":
            major_axis = self.line.get_length()
            minor_axis = major_axis
            rotation = 0
            special_info = "circle. "
        else:
            raise ValueError("invalid special axis type")

        ellipse = Ellipse(center, major_axis, minor_axis, rotation, special_info)
        return ellipse, special_axis


class EllipseRelationGenerator:
    def __init__(self, rule_args) -> None:
        self.rule_args = rule_args

        self.base_rel = [
            "tangent line",
            "tangent circle",
            "concentric",
            "circumscribed",
            "inscribed",
        ]
        self.circle_rel = []

        self.relation_level = []
        for relation_type in self.base_rel:
            level = eval(f"rule_args.ellipse_{relation_type.replace(' ', '_')}_level")
            self.relation_level.append(level)

    def generate_relation(self, ellipse: Ellipse) -> tuple[Any, str]:
        self.ellipse = ellipse
        opt_rel = self.base_rel
        if "circle" in self.ellipse.special_info:
            opt_rel = opt_rel + self.circle_rel

        opt_relation_level = self.relation_level[: len(opt_rel)]
        relation_prob = [x / sum(opt_relation_level) for x in opt_relation_level]
        relation_type = np.random.choice(opt_rel, p=relation_prob)

        if relation_type == "tangent line":
            new_shape = self.get_tangent_line()

        elif relation_type == "concentric":
            new_shape = self.get_concentric_ellipse()

        elif relation_type == "circumscribed":  # circumscribed ellipse
            new_shape = self.get_inscribed_polygon()

        elif relation_type == "inscribed":  # inscribed ellipse
            new_shape = self.get_circumscribed_polygon()

        elif relation_type == "tangent circle":
            new_shape, relation_type = self.get_tangent_circle()

        return new_shape, relation_type

    def get_tangent_line(self) -> Line:
        # choose a point on ellipse randomly and calculate tangent line
        p = self.ellipse.get_point()
        tangent_line = self.ellipse.get_tangent_line(point=p)

        p1, p2 = another_2points_on_line(line=tangent_line, point=p)
        line = Line(type="segment", points=[p1, p2])

        return line

    def get_concentric_ellipse(self) -> list[Ellipse]:
        ellipse_list = []
        num_concentric = randint(1, 4)
        for _ in range(num_concentric):
            scale_factor = uniform(0.6, 1.5)
            while 0.9 < scale_factor < 1.1:
                scale_factor = uniform(0.6, 1.5)
            scaled_major_axis = self.ellipse.major_axis * scale_factor
            scaled_minor_axis = self.ellipse.minor_axis * scale_factor
            center = self.ellipse.center
            rotation = self.ellipse.rotation

            special_info = self.ellipse.special_info
            ellipse = Ellipse(center, scaled_major_axis, scaled_minor_axis, rotation, special_info)
            ellipse_list.append(ellipse)
        return ellipse_list

    def get_inscribed_polygon(self, special_polygon="") -> Polygon:
        special_polygon = np.random.choice(["", "rectangle", "equilateral triangle"])
        if "rectangle" in special_polygon:
            rot = self.ellipse.rotation
            theta_0 = rot - np.pi * uniform(0.1, 0.4)
            theta_1 = rot + np.pi * uniform(0.1, 0.4)
            theta_list = [theta_0, theta_1, theta_0 + np.pi, theta_1 + np.pi]
        elif "equilateral" in special_polygon and "circle" in self.ellipse.special_info:
            theta_0 = uniform(0, np.pi)
            theta_list = [theta_0, theta_0 + np.pi * 2 / 3, theta_0 + np.pi * 4 / 3]
        else:
            special_polygon = ""
            # choose points on ellipse randomly
            num_points = np.random.randint(3, 7)
            theta_list = [uniform(0, 2 * np.pi) for _ in range(num_points)]

        polygon_points = [self.ellipse.get_point(theta) for theta in theta_list]
        polygon = Polygon(points=polygon_points, special_info=special_polygon)
        polygon.to_simple_polygon()
        return polygon

    def get_circumscribed_polygon(self) -> Polygon:
        special_polygon = np.random.choice(["", "rectangle", "equilateral triangle"])
        # choose tangent point on ellipse, and get intersections of tangent lines
        if "rectangle" in special_polygon and "circle" in self.ellipse.special_info:
            theta_0 = uniform(0, np.pi)
            theta_list = [theta_0 + i * np.pi / 2 for i in range(4)]
        elif "equilateral" in special_polygon and "circle" in self.ellipse.special_info:
            theta_0 = uniform(0, np.pi)
            theta_list = [theta_0, theta_0 + np.pi * 2 / 3, theta_0 + np.pi * 4 / 3]
        else:
            special_polygon = ""
            # divide the ellipse to equal sections, and choose a point on each section
            num_points = np.random.randint(3, 7)
            theta_each_section = 2 * np.pi / num_points
            theta_list = [uniform(i * theta_each_section, (i + 0.5) * theta_each_section) for i in range(num_points)]

        tangent_points = [self.ellipse.get_point(theta) for theta in theta_list]

        # get tangent line of each point
        tangent_lines = [self.ellipse.get_tangent_line(point) for point in tangent_points]

        # find intersection of two adjacent lines
        intersections = []
        for i, line1 in enumerate(tangent_lines):
            line2 = tangent_lines[(i + 1) % len(tangent_lines)]
            intersection = find_intersection(line1, line2)
            intersections.append(intersection)

        polygon = Polygon(points=intersections, special_info=special_polygon)
        return polygon

    def get_tangent_circle(self) -> tuple[Ellipse | None, str]:
        if "circle" in self.ellipse.special_info:
            x, y = self.ellipse.center
            new_center = (x + uniform(-0.5, 0.5), y + uniform(-0.5, 0.5))
            angle = polar_angle(self.ellipse.center, new_center)
            tangent_point = self.ellipse.get_point(theta=angle)
        else:  # tangent circle to a ellipse
            theta = np.random.choice([0.5 * np.pi, 1.5 * np.pi])
            tangent_point = self.ellipse.get_point(theta)  # a vertice on minor axis
            radius_vec = line_given2points([self.ellipse.center, tangent_point])
            new_center = another_2points_on_line(line=radius_vec, point=tangent_point)[0]

        d_centers = distance_2points(self.ellipse.center, new_center)
        if d_centers > 0.6:
            return None, "none"

        d_radius_vec = distance_2points(self.ellipse.center, tangent_point)

        if d_centers < d_radius_vec:
            tangent_type = "internal tangent circle"
            radius = d_radius_vec - d_centers

        else:
            tangent_type = "external tangent circle"
            radius = d_centers - d_radius_vec

        circle = Ellipse(center=new_center)
        circle.to_circle(radius)

        return circle, tangent_type
