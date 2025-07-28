"generate rules for producing geometry shapes"

import json
import os
from random import shuffle

import numpy as np
from iterwrap import iterate_wrapper
from numpy.random import choice, normal, randint
from tqdm import tqdm, trange

from common.args import data_args, rule_args, run_args
from data.rule.relations import (
    CustomedShapeGenerator,
    EllipseRelationGenerator,
    FusiformRelationGenerator,
    RelationGenerator,
    SeptaGenerator,
)
from data.rule.shapes import ShapeGenerator
from data.rule.utils import no_overlap, overlap_area, round_floats, valid_intersection


def generate_fossil_rules() -> list[dict[str, list]]:
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

        # Generate volutions/whorls
        volution_shape = choice(["fusiform", "customed_shape"], p=[0.2, 0.8])
        # volution_shape = "fusiform"
        volution_type = "concentric"
        if volution_shape == "ellipse":
            volution_generator = EllipseRelationGenerator(rule_args)
        elif volution_shape == "fusiform":
            volution_generator = FusiformRelationGenerator(rule_args)
        elif volution_shape == "customed_shape":
            volution_generator = CustomedShapeGenerator(rule_args)

        volutions = volution_generator.generate_volutions(initial_chamber, volution_type)

        num_volutions = len(volutions) if "concentric" in volution_type else len(volutions) // 2
        numerical_info["num_volutions"] = float(num_volutions)

        fossil_bbox = volutions[-1].get_bbox()
        numerical_info["fossil_bbox"] = fossil_bbox

        shapes.extend(volutions)
        # shapes.reverse()  # reverse for overlap in 'swing' volution_type

        # Set tunnel angles for each volution
        tunnel_angle = max(5, normal(18, 4))  # initialize
        tunnel_angles = []
        for _ in range(num_volutions - 1):
            scale_factor = normal(1.1, 0.05)
            tunnel_angle *= scale_factor
            tunnel_angles.append(tunnel_angle)

        num_visible_chomata = randint(0, num_volutions)
        visible_chomata_idx = sorted(choice(num_volutions - 1, num_visible_chomata, replace=False))
        numerical_info["visible_chomata_idx"] = [int(idx) for idx in visible_chomata_idx]
        numerical_info["tunnel_angles"] = np.array(tunnel_angles)[visible_chomata_idx].tolist()

        # Generate chomata
        septa_generator = SeptaGenerator(rule_args)
        chomata_list = septa_generator.generate_chomata(
            volutions, tunnel_angles, visible_chomata_idx, volution_type, num_volutions
        )
        shapes.extend(chomata_list)

        # Generate axial filling
        if np.random.rand() < rule_args.prob_has_axial_filling:
            axial_filling = shape_generator.generate_axial_filling(num_volutions, rule_args)
        else:
            axial_filling = []

        # Generate septa folds at poles
        poles_folds = shape_generator.generate_poles_folds(num_volutions, axial_filling, rule_args)

        # Generate other septa folds
        have_septa_folds = choice([True, False], p=[0.7, 0.3])
        if have_septa_folds:
            global_gap = normal(0.7, 0.1)
            septa_folds, num_septa = septa_generator.generate_septa(
                volutions, volution_type, int(num_volutions), axial_filling, poles_folds, global_gap
            )
            # shapes.extend(septa_folds)
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


def generate_rules(idx_target: tuple[int, dict[int, int]]) -> list[dict[str, list]]:
    """
    Generate random rules across different types and shapes. Then mix them together.
    Input:
    target_num_samples (dict): Dictionary of expected number of samples where:
        * Key (int): num_shapes
        * Value (int): num_samples
    Returns:
    a list of samples where each consists a list of shapes and a list of relations.
    """
    idx, target_num_samples = idx_target
    # Set random seed for each process
    seed = os.getpid() + 1
    np.random.seed(seed)

    results = []
    shape_generator = ShapeGenerator(rule_args)
    relation_generator = RelationGenerator(rule_args)

    max_num_shapes = max([key for key in target_num_samples])
    progress_bar = tqdm(total=sum(target_num_samples.values()), desc="Generating rules", position=idx)
    cur_num_samples = {k: 0 for k in target_num_samples.keys()}

    # Generate samples until reaching the target number of samples for each shape count
    while not all(cur_num_samples.get(k, 0) >= v for k, v in target_num_samples.items()):
        shapes = []
        num_shapes = randint(1, max_num_shapes // 2 + 1)  # leave space for special relations
        for _ in range(num_shapes):
            new_shape = shape_generator()
            new_shape_bbox = new_shape.get_bbox()
            area_in_canvas = overlap_area(new_shape_bbox, [[0, 1], [1, 0]])
            area_new_shape = (new_shape_bbox[1][0] - new_shape_bbox[0][0]) * (
                new_shape_bbox[0][1] - new_shape_bbox[1][1]
            )
            if area_in_canvas / area_new_shape < rule_args.in_canvas_area_thres:
                continue

            if no_overlap(shapes, new_shape) and valid_intersection(shapes, new_shape):
                shapes.append(new_shape)

        relations = []
        for head_idx in range(len(shapes)):
            head_shape = shapes[head_idx]
            if head_shape.to_dict()["type"] == "spiral":
                continue

            tail_shape, relation_type = relation_generator(head_shape)
            assert isinstance(tail_shape, list) and isinstance(relation_type, str)

            exclude_shape = [head_shape]
            for i, t_shape in enumerate(tail_shape):
                if not no_overlap(shapes, t_shape, exclude_shape=exclude_shape):
                    continue

                intersection_exclude_shape = [head_shape] if relation_type == "shared edge" else None
                if not valid_intersection(shapes, t_shape, exclude_shape=intersection_exclude_shape):
                    continue

                # Check if each tail_shape is in the canvas
                tail_bbox = t_shape.get_bbox()
                area_in_canvas = overlap_area(tail_bbox, [[0, 1], [1, 0]])
                area_tail_bbox = (tail_bbox[1][0] - tail_bbox[0][0]) * (tail_bbox[0][1] - tail_bbox[1][1])
                if area_in_canvas / area_tail_bbox < rule_args.in_canvas_area_thres:
                    continue

                # if len(shapes) >= rule_args.max_num_shapes:
                #     break

                # Add each tail_shape to shapes
                tail_idx = len(shapes)

                # Keep 'ellipse-polygon/star-relation' order in inscribed or circumscribed relation
                if head_shape.to_dict()["type"] in ["polygon", "star"] and "cribed" in relation_type:
                    relations.append((tail_idx, head_idx, relation_type))
                # Keep 'line-polygon-relation' order in diagonal relation
                elif "polygon" in head_shape.to_dict()["type"] and "diagonal" in relation_type:
                    relations.append((tail_idx, head_idx, relation_type))
                elif i > 0:  # multiple tail shapes (e.g. concentric ellipses or adjacent sectors)
                    relations.append((tail_idx - 1, tail_idx, relation_type))
                else:
                    relations.append((head_idx, tail_idx, relation_type))

                shapes.append(t_shape)
                exclude_shape.append(t_shape)

        if cur_num_samples.get(len(shapes), 0) < target_num_samples.get(len(shapes), 0):
            shapes_dict = [shape.to_dict() for shape in shapes]
            sample = {"shapes": shapes_dict, "relations": relations}
            cur_num_samples[len(shapes)] += 1
            results.append(sample)
            progress_bar.update(1)

    return results


def generate_rules_multiprocess(num_workers: int = 2) -> list[dict[str, list]]:
    """Multiprocessing version"""

    seed = os.getpid()
    np.random.seed(seed)

    target_num_samples = {}
    for i, num_samples in enumerate(data_args.num_samples_per_num_shapes):
        target_num_samples[i + rule_args.min_num_shapes] = num_samples

    target_samples_list = []
    base_samples = {k: v // num_workers for k, v in target_num_samples.items()}
    remainder_samples = {k: v % num_workers for k, v in target_num_samples.items()}

    # Create target samples for all workers except last
    for _ in range(num_workers - 1):
        target_samples_list.append(base_samples.copy())

    # Create last target sample with base + remainders
    last_sample = base_samples.copy()
    for k, v in remainder_samples.items():
        last_sample[k] += v
    target_samples_list.append(last_sample)

    results_list = iterate_wrapper(
        generate_rules,
        list(enumerate(target_samples_list)),
        num_workers=num_workers,
        run_name="generate_rules",
        bar=False,
        restart=True,
    )
    assert results_list is not None
    # Flatten the list of lists into a single list
    results = [item for sublist in results_list for item in sublist]
    shuffle(results)
    return results


def save_rules(rules: list[dict[str, list]], output_file: str):
    with open(output_file, "w") as f:
        json.dump(round_floats(rules, rule_args.output_fp_precision), f, indent=4)


def main():
    if data_args.stage == 1:
        samples = generate_rules_multiprocess(run_args.num_workers)
    elif data_args.stage == 2:
        samples = generate_fossil_rules()

    os.makedirs(os.path.dirname(data_args.rules_path), exist_ok=True)
    save_rules(samples, data_args.rules_path)


if __name__ == "__main__":
    main()
