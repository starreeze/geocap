from typing import Any, Literal, Optional
import numpy as np
from numpy.typing import NDArray
from numpy.random import uniform, randint, normal
from ..shapes import Polygon, Line, Ellipse, Fusiform, Fusiform_2, Curve, CustomedShape
from data.rule.utils import (
    distance_2points,
    polar_angle,
    line_given2points,
    distance_point_to_line,
    another_2points_on_line,
    find_intersection,
    find_symmetric_point,
    get_tangent_line,
    find_perpendicular_line,
)


class RelationGenerator:
    def __init__(self, rule_args) -> None:
        # probability for different types of relation
        self.polygon_relation_generator = PolygonRelationGenerator(rule_args)
        self.line_relation_generator = LineRelationGenerator(rule_args)
        self.ellipse_relation_generator = EllipseRelationGenerator(rule_args)
        self.fusiform_relation_generator = FusiformRelationGenerator(rule_args)

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
            new_point = (mid_point[0] + side_len * uniform(0.3, 1), mid_point[1] + side_len * uniform(0.3, 1))
            polygon = Polygon([points[0], points[1], new_point], special_info="triangle")
            if polygon.check_angle():
                polygon_list.append(polygon)

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

    def get_concentric_ellipse(self, scale_factor=None, num_concentric=None) -> list[Ellipse]:
        ellipse_list = []
        if num_concentric is None:
            num_concentric = randint(1, 4)

        for _ in range(num_concentric):
            if scale_factor is None:
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

        while not polygon.check_angle():
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

    def generate_volutions(
        self, initial_chamber: Ellipse, volution_type: Literal["concentric", "swing"]
    ) -> list[Ellipse]:
        center = (initial_chamber.center[0] + normal(0, 3e-3), initial_chamber.center[1] + normal(0, 3e-3))
        minor_axis = initial_chamber.minor_axis * normal(1.5, 0.1)
        major_axis = minor_axis * normal(2.0, 0.3)
        rotation = normal(0, 0.02)
        volution_0 = Ellipse(center, major_axis, minor_axis, rotation, special_info="volution 0. ")

        if "concentric" in volution_type:
            volutions = [volution_0]
        elif "swing" in volution_type:
            volution_0_swing = Ellipse(
                center, major_axis * 1.1, minor_axis * 1.1, rotation, special_info="volution 0. "
            )
            volutions = [volution_0, volution_0_swing]

        scale_factor = uniform(1.2, 1.5)

        max_num_volutions = randint(5, 13)
        for i in range(max_num_volutions):
            self.ellipse = volutions[-1]
            # scale_factor = scale_factor * (0.99**i)
            if "concentric" in volution_type:
                new_volutions = self.get_concentric_ellipse(scale_factor, 1)
            elif "swing" in volution_type:
                new_volutions = self.get_concentric_ellipse(np.sqrt(scale_factor), 2)

            for new_volution in new_volutions:
                # new_volution.major_axis *= normal(1, 0.02)
                new_volution.special_info = f"volution {i+1}. "
                if new_volution.major_axis < 1:
                    volutions.append(new_volution)

        if "swing" in volution_type:
            for i, volution in enumerate(volutions):
                volution.fill_mode = "white"
                volution.center = (
                    (center[0], center[1] + 0.05 * volution.minor_axis)
                    if i % 2 == 0
                    else (center[0], center[1] - 0.05 * volution.minor_axis)
                )
                volution.adjust_curve_points()

        return volutions


class FusiformRelationGenerator:
    def __init__(self, rule_args) -> None:
        self.rule_args = rule_args

    def generate_relation(self, fusiform: Fusiform | Fusiform_2) -> tuple[Any, str]:
        self.fusiform = fusiform

        return None, "none"

    def get_concentric_fusiform(
        self, scale_factor: list[float] = [], delta_y: float = 0.1, num_concentric: int = 1
    ) -> list[Fusiform]:
        assert isinstance(self.fusiform, Fusiform)

        center = self.fusiform.center
        focal_length = self.fusiform.focal_length
        x_offset = center[0]
        y_offset = self.fusiform.y_offset
        y_symmetric_axis = center[1]
        sin_params = self.fusiform.sin_params.copy()
        fusiform_list = []

        for _ in range(num_concentric):
            if not scale_factor:
                scale_factor = [uniform(0.6, 1.5), uniform(1.2, 2.0)]

            focal_length = focal_length * scale_factor[0]
            sin_params[0] = sin_params[0] * scale_factor[1]
            y_offset = y_offset - delta_y

            new_fusiform = Fusiform(focal_length, x_offset, y_offset, y_symmetric_axis, sin_params)
            fusiform_list.append(new_fusiform)

        return fusiform_list

    def get_concentric_fusiform_2(
        self, scale_factor: list[float] = [], x_delta: float = -0.1, num_concentric: int = 1
    ) -> list[Fusiform_2]:
        assert isinstance(self.fusiform, Fusiform_2)

        focal_length = self.fusiform.focal_length
        x_offset = self.fusiform.x_offset
        y_offset = self.fusiform.y_offset
        power = self.fusiform.power
        x_symmetric_axis = self.fusiform.x_symmetric_axis

        fusiform_list = []

        for _ in range(num_concentric):
            sin_params = [self.fusiform.sin_params[0], 3 * np.pi, np.pi]
            if not scale_factor:
                scale_factor = [uniform(0.6, 1.5), uniform(0.6, 1.5)]

            focal_length = focal_length * scale_factor[0]
            sin_params[0] = sin_params[0] * scale_factor[1]
            x_offset = x_offset + x_delta

            new_fusiform = Fusiform_2(focal_length, x_offset, y_offset, power, x_symmetric_axis, sin_params)
            fusiform_list.append(new_fusiform)

        return fusiform_list

    def generate_volutions(
        self, initial_chamber: Ellipse, volution_type: Literal["concentric", "swing"]
    ) -> list[Fusiform] | list[Fusiform_2]:
        fusiform_type = np.random.choice([1, 2])
        # fusiform_type = 2
        if fusiform_type == 1:
            volutions = self.generate_volutions_1(initial_chamber, volution_type)
        elif fusiform_type == 2:
            volutions = self.generate_volutions_2(initial_chamber, volution_type)

        return volutions

    def generate_volutions_1(
        self, initial_chamber: Ellipse, volution_type: Literal["concentric", "swing"]
    ) -> list[Fusiform]:
        volution_0 = self.get_random_fusiform(initial_chamber, fusiform_type=1)
        assert isinstance(volution_0, Fusiform)

        if "concentric" in volution_type:
            volutions = [volution_0]
        elif "swing" in volution_type:
            volution_0_swing = self.get_random_fusiform(initial_chamber, fusiform_type=1)
            volutions = [volution_0, volution_0_swing]

        scale_factor = [normal(0.9, 0.03), uniform(1.01, 1.03)]
        delta_y = max(0.025, normal(0.025, 0.01))

        max_num_volutions = randint(5, 13)
        for i in range(max_num_volutions):
            self.fusiform = volutions[-1]
            delta_y = delta_y * (1.05**i)
            if "concentric" in volution_type:
                new_volutions = self.get_concentric_fusiform(scale_factor, delta_y, 1)
            elif "swing" in volution_type:
                new_volutions = self.get_concentric_fusiform(np.sqrt(scale_factor).tolist(), 0.5 * delta_y, 2)

            for new_volution in new_volutions:
                if new_volution.is_closed():
                    new_volution.special_info = f"volution {i+1}. "
                    volutions.append(new_volution)

        if "swing" in volution_type:
            for i, volution in enumerate(volutions):
                volution.fill_mode = "white"
                volution.y_symmetric_axis = (
                    volution.y_symmetric_axis + 0.07 * volution.height
                    if i % 2 == 0
                    else volution.y_symmetric_axis - 0.07 * volution.height
                )
                volution.y_offset = (
                    volution.y_offset + 0.07 * volution.height
                    if i % 2 == 0
                    else volution.y_offset - 0.07 * volution.height
                )
                volution.adjust_curve_points()
        return volutions

    def generate_volutions_2(
        self, initial_chamber: Ellipse, volution_type: Literal["concentric", "swing"]
    ) -> list[Fusiform_2]:
        volution_0 = self.get_random_fusiform(initial_chamber, fusiform_type=2)
        assert isinstance(volution_0, Fusiform_2)

        if "concentric" in volution_type:
            volutions = [volution_0]
        elif "swing" in volution_type:
            volution_0_swing = self.get_random_fusiform(initial_chamber, fusiform_type=2)
            volutions = [volution_0, volution_0_swing]

        scale_factor = [normal(0.6, 0.1), uniform(1.2, 1.5)]
        x_delta = uniform(-0.05, -0.02)

        max_num_volutions = randint(5, 13)
        for i in range(max_num_volutions):
            self.fusiform = volutions[-1]

            x_delta = x_delta * (1.01**i)
            if "concentric" in volution_type:
                new_volutions = self.get_concentric_fusiform_2(scale_factor, x_delta, 1)
            elif "swing" in volution_type:
                new_volutions = self.get_concentric_fusiform_2(np.sqrt(scale_factor).tolist(), 0.5 * x_delta, 2)

            for new_volution in new_volutions:
                if new_volution.is_closed():
                    new_volution.special_info = f"volution {i+1}. "
                    volutions.append(new_volution)

        if "swing" in volution_type:
            for i, volution in enumerate(volutions):
                volution.fill_mode = "white"
                volution.y_offset = (
                    volution.y_offset + 0.05 * volution.height
                    if i % 2 == 0
                    else volution.y_offset - 0.05 * volution.height
                )
                volution.adjust_curve_points()

        return volutions

    def get_random_fusiform(self, initial_chamber: Ellipse, fusiform_type: int) -> Fusiform | Fusiform_2:
        center = (initial_chamber.center[0] + normal(0, 1e-3), initial_chamber.center[1] + normal(0, 1e-3))
        volution_0 = None
        while volution_0 is None or not volution_0.is_closed():
            if fusiform_type == 1:
                focal_length = normal(0.4, 0.02)
                x_offset = 0.5
                y_offset = center[1] - normal(0.01, 2e-3)
                y_symmetric_axis = center[1]
                epsilon = max(0.03, normal(0.03, 0.01))
                y_offset += 0.9 * epsilon
                omega = 3 * np.pi
                phi = 0
                # phi = normal(0, 0.01)
                sin_params = [epsilon, omega, phi]
                volution_0 = Fusiform(
                    focal_length, x_offset, y_offset, y_symmetric_axis, sin_params, special_info="volution 0. "
                )
            elif fusiform_type == 2:
                focal_length = normal(1200, 100)
                x_offset = center[0] - normal(1.6, 0.1) * initial_chamber.major_axis
                y_offset = center[1]
                power = max(3.0, normal(3.2, 0.2))
                x_symmetric_axis = center[0]
                epsilon = normal(2e-3, 5e-4)
                # epsilon = 0
                omega = 3 * np.pi
                phi = np.pi
                sin_params = [epsilon, omega, phi]
                volution_0 = Fusiform_2(
                    focal_length, x_offset, y_offset, power, x_symmetric_axis, sin_params, special_info="volution 0. "
                )

        return volution_0

    """
    old version: homogeneous septas with fixed num_septa
    """
    # def generate_septa(self, volutions: list[Fusiform] | list[Ellipse]) -> tuple[list[Ellipse], list[int]]:
    #     septa_list = []

    #     num_septa = [randint(3, 60) for _ in range(20)]
    #     num_septa.sort()
    #     for i, volution in enumerate(volutions[:-1]):
    #         next_volution = volutions[i + 1]

    #         angle_per_sec = (2 * np.pi) / num_septa[i]
    #         theta = uniform(0, 2 * np.pi)
    #         for _ in range(num_septa[i]):
    #             theta = theta + angle_per_sec
    #             p1 = volution.get_point(theta)
    #             p2 = next_volution.get_point(theta)
    #             interval = distance_2points(p1, p2)

    #             center = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
    #             major_axis = uniform(0.5 * interval, 0.9 * interval)
    #             minor_axis = uniform(0.6 * major_axis, major_axis)
    #             rotation = normal(theta, 0.2 * theta)
    #             septa = Ellipse(center, major_axis, minor_axis, rotation, special_info=f"septa of volution {i+1}. ")
    #             septa_list.append(septa)

    #     return septa_list, num_septa[: len(volutions) - 1]


class CustomedShapeGenerator:
    def __init__(self, rule_args) -> None:
        self.rule_args = rule_args

    def generate_relation(self, customed_shape: CustomedShape) -> tuple[Any, str]:
        self.customed_shape = customed_shape
        return None, "none"

    def get_concentric_customed_shape(
        self, scale_factor=None, num_concentric=None, randomize=True
    ) -> list[CustomedShape]:
        customed_shape_list = []
        if num_concentric is None:
            num_concentric = randint(1, 4)

        for _ in range(num_concentric):
            if scale_factor is None:
                scale_factor = uniform(0.6, 1.5)
                while 0.9 < scale_factor < 1.1:
                    scale_factor = uniform(0.6, 1.5)

            scaled_curves = []
            for curve in self.customed_shape.curves:
                new_scale_factor = scale_factor * normal(1, 0.03) if randomize else scale_factor
                scaled_control_points = []
                for i, point in enumerate(curve.control_points):
                    # Translate the point relative to the center of the customed_shape
                    translated_point = (
                        point[0] - self.customed_shape.center[0],
                        point[1] - self.customed_shape.center[1],
                    )
                    # Scale the translated point
                    if i == 0 or i == 3:  # for start and end point, no randomize
                        scaled_point = (translated_point[0] * scale_factor, translated_point[1] * scale_factor)
                    else:
                        scaled_point = (translated_point[0] * new_scale_factor, translated_point[1] * new_scale_factor)
                    # Translate the scaled point back to the original coordinate system
                    final_point = (
                        scaled_point[0] + self.customed_shape.center[0],
                        scaled_point[1] + self.customed_shape.center[1],
                    )
                    scaled_control_points.append(final_point)
                scaled_curves.append(Curve(scaled_control_points))

            special_info = self.customed_shape.special_info
            customed_shape = CustomedShape(scaled_curves, special_info)
            customed_shape_list.append(customed_shape)

        return customed_shape_list

    def generate_volutions(
        self, initial_chamber: Ellipse, volution_type: Literal["concentric", "swing"]
    ) -> list[CustomedShape]:
        volution_0 = self.get_random_customed_shape(initial_chamber)

        if "concentric" in volution_type:
            volutions = [volution_0]
        elif "swing" in volution_type:
            volution_0_swing = self.get_random_customed_shape(initial_chamber)
            volutions = [volution_0, volution_0_swing]

        max_num_volutions = randint(5, 13)
        _stop = False
        for i in range(max_num_volutions):
            self.customed_shape = volutions[-1]
            scale_factor = normal(1.7, 0.1) * (0.96**i)

            if "concentric" in volution_type:
                new_volutions = self.get_concentric_customed_shape(scale_factor, 1)
            elif "swing" in volution_type:
                new_volutions = self.get_concentric_customed_shape(np.sqrt(scale_factor), 2)

            for new_volution in new_volutions:
                new_volution.special_info = f"volution {i+1}. "
                if new_volution.width < 1 and new_volution.height < 1:
                    volutions.append(new_volution)
                else:
                    _stop = True

            if _stop:
                break

        if "swing" in volution_type:
            for i, volution in enumerate(volutions):
                swing_vector = np.array([0, 0.05 * volution.height])
                volution.fill_mode = "white"

                new_vertices = []
                for vertice in volution.vertices:
                    new_vertice = np.array(vertice) + swing_vector if i % 2 == 0 else np.array(vertice) - swing_vector
                    new_vertices.append(new_vertice.tolist())
                volution.vertices = new_vertices

                for curve in volution.curves:
                    new_points = (
                        np.array(curve.control_points) + swing_vector
                        if i % 2 == 0
                        else np.array(curve.control_points) - swing_vector
                    )
                    curve.control_points = [tuple(new_point) for new_point in new_points]
                volution.adjust_curve_points()

        return volutions

    def get_random_customed_shape(self, initial_chamber: Ellipse) -> CustomedShape:
        center = (initial_chamber.center[0] + normal(0, 1e-3), initial_chamber.center[1] + normal(0, 1e-3))
        x0, y0 = center
        major_r = initial_chamber.major_axis * normal(1.6, 0.1)
        minor_r = initial_chamber.minor_axis * normal(0.8, 0.05)

        # add random translation on left and right vertices
        y_trans_1 = normal(0, 0.2) * minor_r
        y_trans_2 = normal(0, 0.2) * minor_r

        control_points_list = []
        for i in range(4):
            angle = i * np.pi / 2

            # start control point
            p0 = [x0 + major_r * np.cos(angle), y0 + minor_r * np.sin(angle)]
            if i == 0:
                p0[1] += y_trans_1
            elif i == 2:
                p0[1] += y_trans_2

            # end control point
            p3 = [x0 + major_r * np.cos(angle + 0.5 * np.pi), y0 + minor_r * np.sin(angle + 0.5 * np.pi)]
            if i == 3:
                p3[1] += y_trans_1
            elif i == 1:
                p3[1] += y_trans_2

            # mid 2 control points
            mid_radius1 = normal(0.95, 0.1) * distance_2points(tuple(p0), center)
            mid_radius2 = normal(0.95, 0.1) * distance_2points(tuple(p3), center)
            delta_angle = normal(0.17 * np.pi, 0.03 * np.pi)
            p1 = [
                x0 + mid_radius1 * np.cos(angle + delta_angle),
                y0 + mid_radius1 * np.sin(angle + delta_angle),
            ]
            p2 = [
                x0 + mid_radius2 * np.cos(angle + 0.5 * np.pi - delta_angle),
                y0 + mid_radius2 * np.sin(angle + 0.5 * np.pi - delta_angle),
            ]

            control_points = [tuple(p) for p in [p0, p1, p2, p3]]
            control_points_list.append(control_points)

        curves = [Curve(control_points) for control_points in control_points_list]
        return CustomedShape(curves, special_info="volution 0. ")


class SeptaGenerator:
    def __init__(self, init_septa_prob=None) -> None:
        if init_septa_prob is None:
            init_septa_prob = max(0.2, uniform(0.3, 0.1))
        self.init_septa_prob = init_septa_prob

    def generate_chomata(
        self,
        volutions: list[Ellipse] | list[Fusiform] | list[Fusiform_2] | list[CustomedShape],
        tunnel_angles: list[float],
        tunnel_start_idx: int,
        volution_type: Literal["concentric", "swing"],
        num_volutions: int,
    ) -> list[Ellipse | Polygon]:
        fossil_center = volutions[0].center
        chomata_list = []

        # Add chomatas by volutions
        step = 1 if "concentric" in volution_type else 2
        for i, volution in enumerate(volutions[:-step]):
            if i / step < tunnel_start_idx:
                continue
            elif i // step >= num_volutions:
                break

            mid_angle = normal(0.5 * np.pi, 0.1)
            next_volution = volutions[i + step]
            tunnel_angle = (tunnel_angles[i // step] / 180) * np.pi

            thetas_upper = [
                mid_angle + 0.5 * tunnel_angle,
                mid_angle - 0.5 * tunnel_angle,
            ]
            thetas_lower = [
                mid_angle + np.pi + 0.5 * tunnel_angle,
                mid_angle + np.pi - 0.5 * tunnel_angle,
            ]
            # chomata_type = np.random.choice(["ellipse", "polygon"])
            chomata_type = "ellipse"
            size = "small"

            if "concentric" in volution_type:
                thetas = thetas_upper + thetas_lower
            elif "swing" in volution_type:
                thetas = thetas_upper if i % 2 == 0 else thetas_lower

            for theta in thetas:
                interval, center = self.get_interval_center(volution, next_volution, theta)

                if chomata_type == "ellipse":
                    chomata = self.one_ellipse_septa(
                        interval, center, fossil_center, theta, mode="inner", size=size, fill_mode="black"
                    )
                elif chomata_type == "polygon":
                    chomata = self.one_polygon_septa(
                        interval, center, theta, volution, next_volution, size=size, fill_mode="black"
                    )
                chomata.special_info = f"chomata of volution {i//step}. "
                chomata_list.append(chomata)

        return chomata_list

    def generate_septa(
        self,
        volutions: list[Ellipse] | list[Fusiform] | list[Fusiform_2] | list[CustomedShape],
        volution_type: Literal["concentric", "swing"],
        num_volutions: int,
        axial_filling: list[dict],
        global_gap: float = 0.5,
    ) -> tuple[list[Ellipse | Polygon], list[int]]:
        fossil_center = volutions[0].center
        septa_list = []
        # Process info about axial filling
        axial_main_angles = []
        axial_extension_angles = []
        axial_extension_volutions = []
        for axial in axial_filling:
            angle_range = [axial["start_angle"], axial["end_angle"]]
            volution_range = [axial["start_volution"], axial["end_volution"]]
            if axial["type"] == "main":
                axial_main_angles.append(angle_range)
            elif axial["type"] == "extension":
                axial_extension_angles.append(angle_range)
                axial_extension_volutions.append(volution_range)

        # Add septa between volutions
        step = 1 if "concentric" in volution_type else 2
        num_septa = [0 for _ in range(num_volutions)]
        for i, volution in enumerate(volutions[:-step]):
            if i // step >= num_volutions:
                break
            if i == 0:  # skip first volution
                continue

            # Check if volution indice in volution range of axial filling (extension)
            in_axial_extension_volution = False
            for volution_range in axial_extension_volutions:
                if volution_range[0] <= i < volution_range[1]:
                    in_axial_extension_volution = True
                    break

            # Calculate interval and angle section
            next_volution = volutions[i + step]
            _interval, _center = self.get_interval_center(volution, next_volution, theta=0.5 * np.pi)
            angle_per_sec = np.arctan2(global_gap * _interval, distance_2points(fossil_center, _center))

            thetas_upper = np.arange(0, np.pi, angle_per_sec)
            thetas_lower = np.arange(np.pi, 2 * np.pi, angle_per_sec)
            if "concentric" in volution_type:
                thetas = np.concatenate([thetas_upper, thetas_lower])
            elif "swing" in volution_type:
                thetas = thetas_upper if i % 2 == 0 else thetas_lower

            for t, theta in enumerate(thetas):
                # Check if theta in angle range of axial filling (main)
                in_axial_main_angle = False
                for angle_range in axial_main_angles:
                    if angle_range[0] < theta < angle_range[1] or angle_range[0] < theta - 2 * np.pi < angle_range[1]:
                        in_axial_main_angle = True
                if in_axial_main_angle:  # No septa for axial filling (main)
                    continue

                # Check if theta in angle range of axial filling (extension)
                in_axial_extension_angle = False
                for angle_range in axial_extension_angles:
                    if angle_range[0] < theta < angle_range[1] or angle_range[0] < theta - 2 * np.pi < angle_range[1]:
                        in_axial_extension_angle = True

                # Generate septa
                interval, center = self.get_interval_center(volution, next_volution, theta)
                if in_axial_extension_angle and in_axial_extension_volution:
                    # ellipse septa for axial filling extension area
                    septa = self.one_ellipse_septa(interval, center, fossil_center, theta, fill_mode="white")
                else:
                    # curve/polygon septa for other area
                    septa_type = np.random.choice(["curve", "polygon"])
                    if "curve" in septa_type:
                        septa = self.one_curve_septa(interval, theta, volution, next_volution, "inner", "small")

                    elif "polygon" in septa_type:
                        septa = self.one_polygon_septa(
                            interval, center, theta, volution, next_volution, fill_mode="white"
                        )
                septa.special_info = f"septa of volution {i//step}. "
                septa_list.append(septa)
                num_septa[i // step] += 1

        return septa_list, num_septa

    def generate_septa_v1(
        self,
        volutions: list[Ellipse] | list[Fusiform] | list[Fusiform_2] | list[CustomedShape],
        volution_type: Literal["concentric", "swing"],
        num_volutions: int,
        axial_filling: list[dict],
    ) -> tuple[list[Ellipse | Polygon], list[int]]:
        fossil_center = volutions[0].center
        septa_list = []

        init_septa_prob = self.init_septa_prob
        prob_scaler = normal(1.3, 0.1)

        # Add septa between volutions
        step = 1 if "concentric" in volution_type else 2
        num_septa = [0 for _ in range(num_volutions)]
        for i, volution in enumerate(volutions[:-step]):
            if i // step >= num_volutions:
                break

            next_volution = volutions[i + step]

            interval, center = self.get_interval_center(volution, next_volution, theta=0.5 * np.pi)
            angle_per_sec = np.arctan2(0.5 * interval, distance_2points(fossil_center, center))

            thetas = np.arange(0, 2 * np.pi, angle_per_sec)
            for t in range(len(thetas)):
                p1 = volution.get_point(thetas[t])
                p2 = volution.get_point(thetas[(t + 1) % len(thetas)])

                next_p1 = next_volution.get_point(thetas[t])
                next_p2 = next_volution.get_point(thetas[(t + 1) % len(thetas)])

                p1_mid = (0.3 * p1[0] + 0.7 * next_p1[0], 0.3 * p1[1] + 0.7 * next_p1[1])
                p2_mid = (0.3 * p2[0] + 0.7 * next_p2[0], 0.3 * p2[1] + 0.7 * next_p2[1])
                p3 = (p1_mid[0] + normal(0, 0.1 * interval), p1_mid[1] + normal(0, 0.1 * interval))
                p4 = (p2_mid[0] + normal(0, 0.1 * interval), p2_mid[1] + normal(0, 0.1 * interval))

                curve = Curve([p1, p3, p4, p2])
                # curve = Curve([p1,next_p1, next_p2, p2])
                septa_list.append(curve)

        return septa_list, num_septa

    def generate_septa_v0(
        self,
        volutions: list[Ellipse] | list[Fusiform] | list[Fusiform_2] | list[CustomedShape],
        volution_type: Literal["concentric", "swing"],
        num_volutions: int,
        axial_filling: list[dict],
    ) -> tuple[list[Ellipse | Polygon], list[int]]:
        fossil_center = volutions[0].center
        septa_list = []

        init_septa_prob = self.init_septa_prob
        prob_scaler = normal(1.3, 0.1)

        # Add septa between volutions
        step = 1 if "concentric" in volution_type else 2
        num_septa = [0 for _ in range(num_volutions)]
        for i, volution in enumerate(volutions[:-step]):
            if i // step >= num_volutions:
                break

            next_volution = volutions[i + step]
            prob = init_septa_prob * (prob_scaler**i)  # more septas in outer volutions

            interval, center = self.get_interval_center(volution, next_volution, theta=0.5 * np.pi)

            angle_per_sec = np.arctan2(0.5 * interval, distance_2points(fossil_center, center))
            max_num_septa = int(2 * np.pi / angle_per_sec * (0.9 ** (len(volutions) - i)))

            thetas_upper = np.arange(0, np.pi, angle_per_sec)
            thetas_lower = np.arange(np.pi, 2 * np.pi, angle_per_sec)
            if "concentric" in volution_type:
                thetas = np.concatenate([thetas_upper, thetas_lower])
            elif "swing" in volution_type:
                thetas = thetas_upper if i % 2 == 0 else thetas_lower

            continuous_septa = -1
            for theta in thetas:
                # ignore horizontal septa
                # if theta < 0.12 * np.pi or 0.88 * np.pi < theta < 1.12 * np.pi or 1.88 * np.pi < theta:
                #     continue

                if np.random.random() < prob or continuous_septa > 0:
                    septa_type = np.random.choice(["ellipse", "polygon"])
                    if continuous_septa > 0:
                        continuous_septa = continuous_septa - 1
                    else:
                        continuous_septa = randint(i // 2, 1 + i)
                        # size = np.random.choice(["big", "small"])
                        size = "small"

                    _, center = self.get_interval_center(volution, next_volution, theta)

                    if septa_type == "ellipse":
                        septa = self.one_ellipse_septa(
                            interval, center, fossil_center, theta, mode="inner", size=size, fill_mode="no"
                        )
                    elif septa_type == "polygon":
                        septa = self.one_polygon_septa(
                            interval, center, theta, volution, next_volution, size=size, fill_mode="no"
                        )
                    septa.special_info = f"septa of volution {i//step}. "
                    septa_list.append(septa)
                    num_septa[i // step] += 1

        # # Add horizontal septa
        # for i, volution in enumerate(volutions[:-1]):
        #     next_volution = volutions[i + 1]
        #     prob = 0.5 * init_septa_prob * (prob_scaler**i)

        #     interval, center = self.get_interval_center(volution, next_volution, theta=0)

        #     angle_per_sec = np.arctan2(0.7 * interval, distance_2points(fossil_center, center))
        #     num_theta_sec = int((np.pi / 3) / angle_per_sec)

        #     theta_range_1 = np.linspace(-0.1 * np.pi, 0.1 * np.pi, num_theta_sec)
        #     theta_range_2 = np.linspace(0.9 * np.pi, 1.1 * np.pi, num_theta_sec)
        #     for theta in np.concatenate((theta_range_1, theta_range_2)):
        #         _, center = self.get_interval_center(volution, next_volution, theta)

        #         # outer horizontal septa
        #         if np.random.random() < prob:
        #             outer_septa = self.one_ellipse_septa(
        #                 interval, center, fossil_center, theta + normal(0, 0.1), mode="outer", size="small"
        #             )
        #             outer_septa.special_info = f"septa of volution {i+1}. "
        #             septa_list.append(outer_septa)
        #             num_septa[i] += 1
        #         # inner horizontal septa
        #         if np.random.random() < prob:
        #             inner_septa = self.one_ellipse_septa(
        #                 interval, center, fossil_center, theta + normal(0, 0.1), mode="inner", size="small"
        #             )
        #             inner_septa.special_info = f"septa of volution {i+1}. "
        #             septa_list.append(inner_septa)
        #             num_septa[i] += 1

        return septa_list, num_septa

    def get_interval_center(
        self,
        volution: Ellipse | Fusiform | Fusiform_2 | CustomedShape,
        next_volution: Ellipse | Fusiform | Fusiform_2 | CustomedShape,
        theta: float,
    ) -> tuple[float, tuple[Any, Any]]:
        p1 = volution.get_point(theta)
        p2 = next_volution.get_point(theta)

        interval = distance_2points(p1, p2)
        center = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
        return interval, center

    def one_ellipse_septa(
        self,
        interval: float,
        center: tuple[float, float],
        fossil_center: tuple[float, float],
        theta: float,
        mode: Literal["outer", "inner"] = "inner",
        size: Literal["big", "small"] = "small",
        fill_mode: Literal["no", "white", "black"] = "no",
    ) -> Ellipse:
        if size == "big":
            major_axis = uniform(0.6 * interval, 0.8 * interval)
            minor_axis = uniform(0.6 * major_axis, major_axis)
        elif size == "small":
            major_axis = uniform(0.4 * interval, 0.6 * interval)
            minor_axis = uniform(0.6 * major_axis, major_axis)
        rotation = theta * normal(1, 0.1)

        # margin = 0.5 * (interval - major_axis) * np.cos(rotation - theta)
        margin = 0.5 * (interval - minor_axis)
        vec_centers = np.array(fossil_center) - np.array(center)
        if mode == "inner":
            septa_center = np.array(center) + vec_centers * (margin / distance_2points(fossil_center, center))
        elif mode == "outer":
            septa_center = np.array(center) - vec_centers * (margin / distance_2points(fossil_center, center))

        ellipse = Ellipse(tuple(septa_center), major_axis, minor_axis, rotation, fill_mode=fill_mode)
        return ellipse

    def one_polygon_septa(
        self,
        interval: float,
        center: tuple[float, float],
        theta: float,
        volution: Ellipse | Fusiform | Fusiform_2 | CustomedShape,
        next_volution: Ellipse | Fusiform | Fusiform_2 | CustomedShape,
        mode: Literal["outer", "inner"] = "inner",
        size: Literal["big", "small"] = "small",
        fill_mode: Literal["no", "white", "black"] = "no",
    ) -> Polygon:
        p1 = volution.get_point(theta + uniform(0.05, 0.1))
        p2 = volution.get_point(theta - uniform(0.05, 0.1))

        next_p1 = next_volution.get_point(theta + uniform(0.05, 0.1))
        next_p2 = next_volution.get_point(theta - uniform(0.05, 0.1))

        # num_edges = np.random.choice([3, 4, 5, 6], p=[0.1, 0.3, 0.3, 0.3])
        num_edges = np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])
        if num_edges == 3:
            if size == "big":
                p3 = next_volution.get_point(theta + normal(0, 0.05))
            elif size == "small":
                p3 = (center[0] + normal(0, 0.25 * interval), center[1] + normal(0, 0.25 * interval))
            points = [p1, p2, p3]
        elif num_edges == 4:
            if size == "big":
                p3 = next_p1
                p4 = next_p2
            elif size == "small":
                p1_mid = (0.5 * (p1[0] + next_p1[0]), 0.5 * (p1[1] + next_p1[1]))
                p2_mid = (0.5 * (p2[0] + next_p2[0]), 0.5 * (p2[1] + next_p2[1]))
                p3 = (p1_mid[0] + normal(0, 0.1 * interval), p1_mid[1] + normal(0, 0.1 * interval))
                p4 = (p2_mid[0] + normal(0, 0.1 * interval), p2_mid[1] + normal(0, 0.1 * interval))
            points = [p1, p2, p3, p4]
        elif num_edges == 5:
            p_mid = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]))
            next_p5 = next_volution.get_point(theta + normal(0, 0.05))
            if size == "big":
                ratio = max(0.5, uniform(0.6, 0.1))
                p5 = next_p5
            elif size == "small":
                ratio = max(0.2, normal(0.3, 0.05))
                center_ratio = normal(0.5, 0.1)
                p5 = (
                    (center_ratio * next_p5[0] + (1 - center_ratio) * p_mid[0]),
                    (center_ratio * next_p5[1] + (1 - center_ratio) * p_mid[1]),
                )
            p3 = (ratio * next_p1[0] + (1 - ratio) * p1[0]), (ratio * next_p1[1] + (1 - ratio) * p1[1])
            p4 = (ratio * next_p2[0] + (1 - ratio) * p2[0]), (ratio * next_p2[1] + (1 - ratio) * p2[1])
            points = [p1, p2, p3, p4, p5]
        elif num_edges == 6:
            if size == "big":
                ratio = max(0.5, uniform(0.6, 0.1))
                p5 = next_p1
                p6 = next_p2
            elif size == "small":
                ratio = max(0.2, normal(0.3, 0.05))
                center_ratio = normal(0.5, 0.1)
                p5 = (
                    (center_ratio * next_p1[0] + (1 - center_ratio) * p1[0]),
                    (center_ratio * next_p1[1] + (1 - center_ratio) * p1[1]),
                )
                p6 = (
                    (center_ratio * next_p2[0] + (1 - center_ratio) * p2[0]),
                    (center_ratio * next_p2[1] + (1 - center_ratio) * p2[1]),
                )
            p3 = (ratio * next_p1[0] + (1 - ratio) * p1[0]), (ratio * next_p1[1] + (1 - ratio) * p1[1])
            p3 = (p3[0] + normal(0, 0.1 * interval), p3[1] + normal(0, 0.1 * interval))
            p4 = (ratio * next_p2[0] + (1 - ratio) * p2[0]), (ratio * next_p2[1] + (1 - ratio) * p2[1])
            p4 = (p4[0] + normal(0, 0.1 * interval), p4[1] + normal(0, 0.1 * interval))
            points = [p1, p2, p3, p4, p5, p6]

        polygon = Polygon(points, fill_mode=fill_mode)
        polygon.to_simple_polygon()
        return polygon

    def one_curve_septa(
        self,
        interval: float,
        theta: float,
        volution: Ellipse | Fusiform | Fusiform_2 | CustomedShape,
        next_volution: Ellipse | Fusiform | Fusiform_2 | CustomedShape,
        mode: Literal["outer", "inner"] = "inner",
        size: Literal["big", "small"] = "small",
        fill_mode: Literal["no", "white", "black"] = "no",
    ):
        n = len(volution.curve_points)
        p1 = volution.get_point(theta + uniform(0.02, 0.05))
        p2 = volution.get_point(theta - uniform(0.02, 0.05))
        # assert isinstance(p1, tuple) and isinstance(p2, tuple)

        points_next_volution = []
        # curve_points = [tuple(point) for point in volution.curve_points.tolist()]
        for p in [p1, p2]:
            p_array = np.array(p)
            p_index = np.where(np.all(p_array == volution.curve_points, axis=1))[0][0]
            index_range = [i % n for i in range(p_index - 3, p_index + 4)]

            # Get normal line of volution at p
            points = [tuple(p) for p in volution.curve_points[index_range]]
            # points = list(dict.fromkeys(points))
            points = list(set(points))  # remove duplicates
            tangent_line = get_tangent_line(p, points)

            normal_line = find_perpendicular_line(tangent_line, p)

            # Get intersections of normal line and next volution
            quarter_size = len(volution.curve_points) // 4
            quarter_idx = p_index // quarter_size  # i_th curve
            start_idx = int(quarter_size * quarter_idx)
            end_idx = int(quarter_size * (quarter_idx + 1))

            distances = [
                distance_point_to_line(point, normal_line) for point in next_volution.curve_points[start_idx:end_idx]
            ]
            min_distance_idx = np.argmin(distances)

            next_point = next_volution.curve_points[start_idx + min_distance_idx]
            points_next_volution.append(next_point)

        next_p1, next_p2 = points_next_volution
        p1_mid = (0.3 * p1[0] + 0.7 * next_p1[0], 0.3 * p1[1] + 0.7 * next_p1[1])
        p2_mid = (0.3 * p2[0] + 0.7 * next_p2[0], 0.3 * p2[1] + 0.7 * next_p2[1])
        p3 = (p1_mid[0] + normal(0, 0.1 * interval), p1_mid[1] + normal(0, 0.1 * interval))
        p4 = (p2_mid[0] + normal(0, 0.1 * interval), p2_mid[1] + normal(0, 0.1 * interval))

        curve = Curve([p1, p3, p4, p2], fill_mode=fill_mode)
        return curve
