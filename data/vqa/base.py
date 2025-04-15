# -*- coding: utf-8 -*-
# @Date    : 2024-12-10 18:21:21
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

from collections import Counter
from typing import Any

from tqdm import tqdm

from common.args import logger
from data.rule.shapes import GSRule
from data.rule.utils import round_floats


class GeneratorBase:
    data: list[dict[str, Any]] = []  # [{shapes: [{type, center, box, area}], relations, counts}]
    total_shapes = [
        "line",
        "ellipse",
        "circle",
        "triangle",  # order cannot be changed; this will be indexed
        "quadrilateral",
        "pentagon",
        "hexagon",
        "rectangle",
        "square",
        "spiral",
    ]
    # parent -> [(child_type, child_desc)]: format to `parent (not/excluding child_desc)`
    shape_hierarchy = {
        "ellipse": [("circle", "circle")],
        "rectangle": [("square", "square")],
        "quadrilateral": [("rectangle", "rectangle"), ("square", "square")],
        "line": [
            ("triangle", "polygon edge"),
            ("quadrilateral", "polygon edge"),
            ("pentagon", "polygon edge"),
            ("hexagon", "polygon edge"),
            ("rectangle", "polygon edge"),
            ("square", "polygon edge"),
        ],
    }
    relation_reverse = {
        "tangent": "tangent",
        "parallel": "parallel",
        "circumscribed": "inscribed",
        "inscribed": "circumscribed",
        "shared edge": "shared edge",
        "diagonal": "circumscribed",
        "major axis": "circumscribed",
        "minor axis": "circumscribed",
        "diameter": "circumscribed",
    }
    total_relations = list(relation_reverse.keys())

    @staticmethod
    def get_relation(relation: str) -> str:
        if "tangent" in relation:
            return "tangent"
        if "circumscribed" in relation:
            return "circumscribed"
        if "inscribed" in relation:
            return "inscribed"
        # these will not appear as they require multiple shapes, causing ambiguity
        if relation in ["concentric", "similar", "symmetric"]:
            raise NotImplementedError(f"Relation {relation} not implemented")
        return relation

    @classmethod
    def get_type(cls, shape_dict: dict[str, Any]) -> str:
        special_info = shape_dict.get("special_info", "").strip(" .").split(" ")[-1]
        if special_info:
            return special_info
        if shape_dict["type"] in ["segment", "ray"]:
            return "line"
        if shape_dict["type"] == "polygon":
            return cls.total_shapes[len(shape_dict["points"])]
        return shape_dict["type"]

    def __init__(self, rules: list[dict[str, Any]]):
        if self.data:
            return
        logger.info("Loading VQA data from rules")
        for figure in tqdm(rules):
            info: dict[str, Any] = {"shapes": []}
            for shape_dict in figure["shapes"]:
                shape = GSRule.from_dict(shape_dict)
                shape_info: dict[str, Any] = shape_dict | {
                    "type": self.get_type(shape_dict),
                    "center": shape.get_centroid(),
                    "box": shape.get_bbox(),
                    "area": shape.get_area(),
                }
                shape_info.pop("special_info", None)
                shape_info.pop("fill_mode", None)
                info["shapes"].append(round_floats(shape_info))
            info["relations"] = figure["relations"]
            info["counts"] = dict(Counter(shape["type"] for shape in info["shapes"]))
            self.data.append(info)

    @classmethod
    def clarify_hierarchical_choices(cls, qa: dict[str, Any], image_types: list[str]):
        """
        Add clarification to choices to clarify hierarchical types when child types appear in either choices or figure.
        Clarification is performed by manual call in corresponding perspective.
        QA will be modified in place.
        """
        choices: list[str] = qa["choices"]
        for parent, children in cls.shape_hierarchy.items():
            try:
                parent_idx = choices.index(parent)
            except ValueError:
                continue
            # Check if any child type exists in choices or figure
            overlapping_child_descs = set([c[1] for c in children if c[0] in choices + image_types])
            if not overlapping_child_descs:
                continue
            # Add clarification to parent
            choices[parent_idx] = f"{parent} ({', '.join(d + 's' for d in overlapping_child_descs)} excluded)"
            # Update answer if needed
            if qa["answer"] == parent:
                qa["answer"] = choices[parent_idx]

    @classmethod
    def clarify_hierarchical_text(cls, type: str, image_types: list[str], add_s: bool = False) -> str:
        if type not in cls.shape_hierarchy:
            return type + "s" if add_s else type
        overlapping_child_descs = set([c[1] for c in cls.shape_hierarchy[type] if c[0] in image_types])
        if not overlapping_child_descs:
            return type
        if add_s:
            child_desc = ", ".join(c + "s" for c in overlapping_child_descs)
            return f"{type}s (excluding {child_desc})"
        else:
            child_desc = ", ".join(overlapping_child_descs)
            return f"{type} (not {child_desc})"
