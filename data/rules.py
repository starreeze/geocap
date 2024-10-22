"generate rules for producing geometry shapes"
import json
import os

from numpy.random import randint, choice, normal
from tqdm import trange

from common.args import data_args, rule_args
from data.relations import RelationGenerator, SeptaGenerator, CustomedShapeGenerator
from data.shapes import ShapeGenerator
from data.utils import overlap_area


def generate_fossil_rules(data_args, rule_args) -> list[dict[str, list]]:
    shape_generator = ShapeGenerator(rule_args)
    relation_generator = RelationGenerator(rule_args)
    results = []

    for _ in trange(data_args.num_fossil_samples):
        shapes = []
        numerical_info = {}

        # Generate initial chamber(a small ellipse in the center of the canvas)
        initial_chamber = shape_generator.generate_initial_chamber()
        shapes.append(initial_chamber)

        # Generate volutions/whorls(a set of concentric ellipses or fusiforms)
        # volution_shape = choice(["ellipse", "fusiform", "customed_shape"], p=[0.2, 0.8])
        # volution_type = choice(["concentric", "swing"])
        volution_shape = "customed_shape"
        volution_type = "swing"
        if volution_shape == "ellipse":
            volution_generator = relation_generator.ellipse_relation_generator
        elif volution_shape == "fusiform":
            volution_generator = relation_generator.fusiform_relation_generator
        elif volution_shape == "customed_shape":
            cs_generator = CustomedShapeGenerator(rule_args)
            volutions = cs_generator.generate_volutions(initial_chamber, volution_type)

        #### volutions = volution_generator.generate_volutions(initial_chamber, volution_type)

        numerical_info["num_volutions"] = (
            len(volutions) - 1 if "concentric" in volution_type else len(volutions) // 2 - 1
        )
        shapes.extend(volutions)
        shapes.reverse()

        # Set tunnel angles for each volution
        tunnel_angle = normal(12, 3)  # initialize
        tunnel_angles = []
        for _ in range(numerical_info["num_volutions"]):
            scale_factor = normal(1.1, 0.1)
            tunnel_angle *= scale_factor
            tunnel_angles.append(tunnel_angle)

        tunnel_start_idx = randint(0, 4)
        numerical_info["tunnel_start_idx"] = tunnel_start_idx
        numerical_info["tunnel_angles"] = tunnel_angles[tunnel_start_idx:]

        # Generate chomata
        septa_generator = SeptaGenerator()
        chomata_list = septa_generator.generate_chomata(volutions, tunnel_angles, volution_type)
        shapes.extend(chomata_list)

        # Generate septa
        have_septa_folds = choice([True, False])
        # have_septa_folds = False
        if have_septa_folds:
            septa_folds, num_septa = septa_generator.generate_septa(volutions, volution_type)
            shapes.extend(septa_folds)
            numerical_info["num_septa"] = num_septa

        shapes_dict = [shape.to_dict() for shape in shapes]

        img_width = normal(5.0, 1.0)
        img_height = img_width * normal(0.6, 0.1)
        img_size = [img_width, img_height]
        numerical_info["img_size"] = img_size
        results.append({"shapes": shapes_dict, "numerical_info": numerical_info})

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


def save_rules(rules: list[dict[str, list]], output_file: str):
    with open(output_file, "w") as f:
        json.dump(rules, f, default=vars)


def main():
    if "basic" in rule_args.mode:
        samples = generate_rules(data_args, rule_args)
    elif "fossil" in rule_args.mode:
        samples = generate_fossil_rules(data_args, rule_args)

    os.makedirs(os.path.dirname(data_args.rules_path), exist_ok=True)
    save_rules(samples, data_args.rules_path)


if __name__ == "__main__":
    main()
