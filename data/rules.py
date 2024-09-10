"generate rules for producing geometry shapes"
import json
import os

from numpy.random import randint
from tqdm import trange

from common.args import data_args, rule_args
from data.relations import RelationGenerator
from data.shapes import ShapeGenerator
from data.utils import overlap_area


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
    samples = generate_rules(data_args, rule_args)
    os.makedirs(os.path.dirname(data_args.rules_path), exist_ok=True)
    save_rules(samples, data_args.rules_path)


if __name__ == "__main__":
    main()
