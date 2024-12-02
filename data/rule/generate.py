"generate rules for producing geometry shapes"
import json
import os

import numpy as np
from numpy.random import choice, normal, randint
from tqdm import trange

from common.args import data_args, rule_args
from data.rule.relations import (
    RelationGenerator,
    SeptaGenerator,
    EllipseRelationGenerator,
    FusiformRelationGenerator,
    CustomedShapeGenerator,
)
from data.shapes import ShapeGenerator
from data.rule.utils import overlap_area


def generate_fossil_rules(data_args, rule_args) -> list[dict[str, list]]:
    shape_generator = ShapeGenerator(rule_args)
    results = []

    for _ in trange(data_args.num_fossil_samples):
        shapes = []
        numerical_info = {}

        # Generate initial chamber(a small ellipse in the center of the canvas)
        initial_chamber = shape_generator.generate_initial_chamber()
        fossil_center = initial_chamber.center
        numerical_info["center"] = fossil_center
        shapes.append(initial_chamber)

        # Generate volutions/whorls(a set of concentric ellipses or fusiforms)

        # volution_shape = choice(["ellipse", "fusiform", "customed_shape"], p=[0.1, 0.2, 0.7])
        volution_shape = choice(["fusiform", "customed_shape"], p=[0.3, 0.7])
        # volution_type = choice(["concentric", "swing"])
        # volution_shape = "ellipse"
        volution_type = "concentric"
        if volution_shape == "ellipse":
            volution_generator = EllipseRelationGenerator(rule_args)
        elif volution_shape == "fusiform":
            volution_generator = FusiformRelationGenerator(rule_args)
        elif volution_shape == "customed_shape":
            volution_generator = CustomedShapeGenerator(rule_args)

        volutions = volution_generator.generate_volutions(initial_chamber, volution_type)

        num_volutions = len(volutions) - 1 if "concentric" in volution_type else len(volutions) / 2 - 1
        numerical_info["num_volutions"] = float(num_volutions)

        fossil_bbox = volutions[-1].get_bbox()
        numerical_info["fossil_bbox"] = fossil_bbox

        shapes.extend(volutions)
        shapes.reverse()  # reverse for overlap in 'swing' volution_type

        # Set tunnel angles for each volution
        tunnel_angle = normal(18, 2)  # initialize
        tunnel_angles = []
        for _ in range(int(num_volutions)):
            scale_factor = normal(1.1, 0.1)
            tunnel_angle *= scale_factor
            tunnel_angles.append(tunnel_angle)

        tunnel_start_idx = randint(0, 4)
        numerical_info["tunnel_start_idx"] = tunnel_start_idx
        numerical_info["tunnel_angles"] = tunnel_angles[tunnel_start_idx:]

        # Generate chomata
        # septa_generator = SeptaGenerator()
        septa_generator = SeptaGenerator(init_septa_prob=1)
        chomata_list = septa_generator.generate_chomata(
            volutions, tunnel_angles, tunnel_start_idx, volution_type, int(num_volutions)
        )
        shapes.extend(chomata_list)

        # Generate axial filling
        if np.random.rand() < rule_args.prob_has_axial_filling:
            axial_filling = shape_generator.generate_axial_filling(int(num_volutions), rule_args)
        else:
            axial_filling = []

        # Generate septa folds at poles
        poles_folds = shape_generator.generate_poles_folds(int(num_volutions), axial_filling, rule_args)

        # Generate other septa folds
        have_septa_folds = choice([True, False])
        # have_septa_folds = True
        if have_septa_folds:
            global_gap = normal(0.8, 0.1)
            septa_folds, num_septa = septa_generator.generate_septa(
                volutions, volution_type, int(num_volutions), axial_filling, global_gap
            )
            septa_folds = [shape.to_dict() for shape in septa_folds]
        else:
            septa_folds = []
            num_septa = [0 for _ in range(int(num_volutions))]
        numerical_info["num_septa"] = num_septa

        shapes_dict = [shape.to_dict() for shape in shapes]
        results.append(
            {
                "shapes": shapes_dict,
                "axial_filling": axial_filling,
                "septa_folds": septa_folds,
                "poles_folds": poles_folds,
                "numerical_info": numerical_info,
            }
        )

    assert len(results) == data_args.num_fossil_samples
    return results


def generate_rules(data_args, rule_args) -> list[dict[str, list]]:
    """
    Generate random rules across different types and shapes. Then mix them together.
    Returns: a list of samples where each consists a list of shapes and a list of relations.
    """
    shape_generator = ShapeGenerator(rule_args)
    relation_generator = RelationGenerator(rule_args)
    results = []

    num_init_shapes = 0
    total_shapes = 0
    for _ in trange(data_args.num_basic_geo_samples):
        shapes = []
        num_shapes = randint(2, rule_args.max_num_shapes // 2)  # leave space for special relations
        for _ in range(num_shapes):
            new_shape = shape_generator()
            # shapes.append(new_shape)
            if no_overlap(shapes, new_shape):
                shapes.append(new_shape)

        num_init_shapes = num_init_shapes + len(shapes)

        relations = []
        for head_idx in range(len(shapes)):
            head_shape = shapes[head_idx]
            if head_shape.to_dict()["type"] == "spiral":
                continue

            tail_shape, relation_type = relation_generator(head_shape)

            if isinstance(tail_shape, list):
                exclude_shape = [head_shape]
                for t_shape in tail_shape:
                    if no_overlap(shapes, t_shape, exclude_shape=exclude_shape):
                        tail_idx = len(shapes)
                        relations.append((head_idx, tail_idx, relation_type))
                        shapes.append(t_shape)
                        exclude_shape.append(t_shape)
            else:  # tail_shape is a GSRule instance
                if no_overlap(shapes, tail_shape, exclude_shape=[head_shape]):
                    tail_idx = len(shapes)
                    relations.append((head_idx, tail_idx, relation_type))
                    shapes.append(tail_shape)

        total_shapes += len(shapes)
        shapes_dict = [shape.to_dict() for shape in shapes]
        sample = {"shapes": shapes_dict, "relations": relations}
        results.append(sample)

    print(f"number of initial shapes = {num_init_shapes}")
    print(f"total shapes = {total_shapes}")
    assert len(results) == data_args.num_basic_geo_samples
    return results


def no_overlap(shapes, new_shape, exclude_shape=None, thres=0.2) -> bool:
    if new_shape is None:
        return False

    if exclude_shape is None:
        exclude_shape = []

    iou_sum = 0
    for cur_shape in shapes:
        if cur_shape not in exclude_shape:
            cur_bbox = cur_shape.get_bbox()
            new_bbox = new_shape.get_bbox()
            cur_area = (cur_bbox[1][0] - cur_bbox[0][0]) * (cur_bbox[0][1] - cur_bbox[1][1])
            new_area = (new_bbox[1][0] - new_bbox[0][0]) * (new_bbox[0][1] - new_bbox[1][1])
            intersection = overlap_area(cur_bbox, new_bbox)
            union = cur_area + new_area - intersection
            iou_sum += intersection / union

    if iou_sum > thres:
        return False
    return True


def round_floats(obj, precision=2):
    if isinstance(obj, float):
        return round(obj, precision)
    if isinstance(obj, dict):
        return {k: round_floats(v, precision) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(x, precision) for x in obj]
    return obj


def save_rules(rules: list[dict[str, list]], output_file: str):
    with open(output_file, "w") as f:
        json.dump(round_floats(rules, rule_args.output_fp_precision), f)


def main():
    if data_args.stage == 1:
        samples = generate_rules(data_args, rule_args)
    elif data_args.stage == 2:
        samples = generate_fossil_rules(data_args, rule_args)

    os.makedirs(os.path.dirname(data_args.rules_path), exist_ok=True)
    save_rules(samples, data_args.rules_path)


if __name__ == "__main__":
    main()
