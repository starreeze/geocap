import random
from itertools import product
from typing import Any, cast

import numpy as np
from tqdm import tqdm

from common.args import logger, vqa_args
from data.rule.utils import overlap_area
from data.vqa.base import GeneratorBase


class RuleBasedQAGenerator(GeneratorBase):
    def __call__(self, perspective: str) -> list[dict[str, Any]]:
        logger.info(f"Generating {perspective} questions")
        qa_pairs: list[dict[str, Any]] = []
        for i, figure in tqdm(enumerate(self.data), total=len(self.data)):
            for j, qa in enumerate(getattr(self, perspective)(figure)):
                self.clarify_hierarchical_choices(qa)
                qa_pairs.append({"image_id": i, "question_id": j} | qa)
        return qa_pairs

    @classmethod
    def counting(cls, figure: dict[str, Any]) -> list[dict[str, Any]]:
        "how many [type] are there?"
        # Step 1: Generate probability distribution based on counts
        counts: dict[str, int] = figure["counts"]
        total_count = sum(counts.values())
        prob_dist = {type: count / total_count for type, count in counts.items()}

        # Step 2: Generate questions based on observed shapes (without duplicates but weighted)
        types = list(counts.keys())
        probs = [prob_dist[type] for type in types]
        num_questions = min(len(types), vqa_args.max_q_ip - 1)
        selected_types = list(np.random.choice(types, size=num_questions, p=probs, replace=False))

        # Step 3: Add question for unobserved shape (count = 0)
        unobserved_types = [t for t in cls.total_shapes if t not in counts]
        zero_type = random.choice(unobserved_types) if unobserved_types else random.choice(types)

        # Step 4: Generate questions with choices
        qa_pairs: list[dict[str, Any]] = []
        for type in selected_types + [zero_type]:
            correct_answer = counts.get(type, 0)
            # Generate choices with uniform spacing around the correct answer
            position = random.choices(range(4), weights=vqa_args.gt_choice_w)[0]
            choices = [max(0, correct_answer + i - position) for i in range(4)]
            if len(set(choices)) < 4:  # Handle cases where some choices are 0
                choices = list(range(4))
            clarified_type = cls.clarify_hierarchical_text(type, list(figure["counts"].keys()), "counting")
            question = f"How many {clarified_type} are there in the image?"

            qa_pairs.append({"question": question, "choices": choices, "answer": correct_answer})
        return qa_pairs

    @classmethod
    def relation(cls, figure: dict[str, Any]) -> list[dict[str, Any]]:
        "what's the relation of A to B?"
        # ask relations only on shapes with counting = 1 to avoid ambiguity
        types = [k for k, v in figure["counts"].items() if v == 1]

        # sample relation pairs
        relations: dict[tuple[str, str], str] = {}
        for i, j, r in figure["relations"]:
            type_i, type_j = figure["shapes"][i]["type"], figure["shapes"][j]["type"]
            if type_i in types and type_j in types:
                relations[(type_i, type_j)] = cls.get_relation(r)
        relation_pairs = dict(random.sample(list(relations.items()), min(len(relations), vqa_args.max_q_ip - 1)))

        # reverse half of the relation pairs to avoid bias
        reversed_pairs = {}
        for key, relation in relation_pairs.items():
            if random.random() < 0.5:
                reversed_pairs[(key[1], key[0])] = cls.relation_reverse[relation]
            else:
                reversed_pairs[key] = relation
        relation_pairs = reversed_pairs

        # sample no-relation pairs
        none_desc = "none of the above"
        if random.random() < vqa_args.nrel_q_prob:
            total = cast(list[tuple[str, str]], list(product(types, repeat=2)))
            no_relations = [t for t in total if t[0] != t[1] and t not in relations and t[::-1] not in relations]
            no_relation_pairs = {k: none_desc for k in random.sample(no_relations, min(len(no_relations), 1))}
        else:
            no_relation_pairs = {}
        sampled_pairs = relation_pairs | no_relation_pairs

        # generate questions
        qa_pairs: list[dict[str, Any]] = []
        for pair, relation in sampled_pairs.items():
            type1 = cls.clarify_hierarchical_text(pair[0], list(figure["counts"].keys()), "relation")
            type2 = cls.clarify_hierarchical_text(pair[1], list(figure["counts"].keys()), "relation")
            question = f"What is the relationship of the {type1} to the {type2} in the image?"
            if pair in relation_pairs:
                compliment = [r for r in cls.total_relations if r != relation]
                choices = [relation] + random.sample(compliment, 2)
                random.shuffle(choices)
                choices.append(none_desc)
                answer = relation
            else:
                choices = random.sample(cls.total_relations, 3) + [none_desc]
                answer = none_desc
            qa_pairs.append({"question": question, "choices": choices, "answer": answer})
        return qa_pairs

    @classmethod
    def size(cls, figure: dict[str, Any]) -> list[dict[str, Any]]:
        "what's the width, height, area of [shape]?"

        qa_pairs: list[dict[str, Any]] = []
        dimensions = ["horizontal span", "vertical span", "area"]
        questions_per_dim = vqa_args.max_q_ip // 3
        remaining = vqa_args.max_q_ip % 3

        for i, dim in enumerate(dimensions):
            # exclude line as it has no area
            if dim == "area":
                types = [k for k, v in figure["counts"].items() if v == 1 and k != "line"]
            else:
                types = [k for k, v in figure["counts"].items() if v == 1]
            num_questions = questions_per_dim + (1 if remaining > 0 else 0)
            remaining -= 1
            sampled_types = random.sample(types, min(len(types), num_questions))

            for type in sampled_types:
                clarified_type = cls.clarify_hierarchical_text(type, list(figure["counts"].keys()), "size")
                shape = next(s for s in figure["shapes"] if s["type"] == type)
                if dim == "area":
                    correct_value = shape["area"]
                    question = f"which of the following is closest to the area of the {clarified_type}?"
                else:
                    # ensure the box is in range
                    box = np.array(shape["box"]).clip(0, 1)
                    correct_value = abs(box[1, i] - box[0, i])
                    question = f"which of the following is closest to the {dim} of the {clarified_type}?"
                question = "Suppose that the width and height of the image is 1, " + question

                factor = vqa_args.size_diff
                # Try different positions according to gt_choice_w until all choices are in range
                valid_choices = None
                candidates = list(range(4))
                weights = vqa_args.gt_choice_w.copy()
                while candidates:
                    pos = random.choices(candidates, weights=weights)[0]
                    idx = candidates.index(pos)
                    candidates.pop(idx)
                    weights.pop(idx)
                    test_choices = [correct_value + factor * (i - pos) for i in range(4)]
                    # ensure all choices are in range
                    if all(0 <= v <= 1 for v in test_choices):
                        position = pos
                        valid_choices = test_choices
                        break
                # we should always have valid choices as the gt is in range and factor is not too large
                assert valid_choices is not None

                choices = [str(round(v, vqa_args.vqa_digits)) for v in valid_choices]
                answer = choices[position]
                qa_pairs.append({"question": question, "choices": choices, "answer": answer})
        return qa_pairs

    @classmethod
    def location(cls, figure: dict[str, Any]) -> list[dict[str, Any]]:
        "where is A located (relative to B)?"
        types = [k for k, v in figure["counts"].items() if v == 1]
        qa_pairs: list[dict[str, Any]] = []

        rel_questions = vqa_args.max_q_ip // 2
        abs_questions = vqa_args.max_q_ip - rel_questions

        def get_position(x: float, y: float, ref_x: float = 0.5, ref_y: float = 0.5) -> str | None:
            "if the shapes are very close to each other, return None"
            if min(abs(x - ref_x), abs(y - ref_y)) < vqa_args.location_type_t:
                return None
            return ("upper " if y > ref_y else "lower ") + ("left" if x < ref_x else "right")

        def generate_choices(correct_pos: str) -> list[str]:
            positions = ["upper left", "upper right", "lower left", "lower right"]
            positions.remove(correct_pos)
            choices = [correct_pos] + random.sample(positions, 3)
            random.shuffle(choices)
            return choices

        # Generate absolute position questions
        sampled_types = random.sample(types, len(types))
        for type in sampled_types:
            clarified_type = cls.clarify_hierarchical_text(type, list(figure["counts"].keys()), "location")
            shape = next(s for s in figure["shapes"] if s["type"] == type)
            x, y = shape["center"]
            correct_pos = get_position(x, y)
            if correct_pos is None:
                continue
            question = f"Considering the centroid, where is the {clarified_type} located in the image?"
            choices = generate_choices(correct_pos)
            qa_pairs.append({"question": question, "choices": choices, "answer": correct_pos})
            if len(qa_pairs) >= abs_questions:
                break

        # Generate relative position questions
        type_pairs = [p for p in product(types, repeat=2) if p[0] != p[1]]
        sampled_pairs = random.sample(type_pairs, len(type_pairs))
        relative_qa_pairs: list[dict[str, Any]] = []
        for type_a, type_b in sampled_pairs:
            type_a_clear = cls.clarify_hierarchical_text(type_a, list(figure["counts"].keys()), "location")
            type_b_clear = cls.clarify_hierarchical_text(type_b, list(figure["counts"].keys()), "location")
            shape_a = next(s for s in figure["shapes"] if s["type"] == type_a)
            shape_b = next(s for s in figure["shapes"] if s["type"] == type_b)
            x_a, y_a = shape_a["center"]
            x_b, y_b = shape_b["center"]
            correct_pos = get_position(x_a, y_a, x_b, y_b)
            if correct_pos is None:
                continue
            question = f"Considering the centroid, where is the {type_a_clear} located relative to the {type_b_clear}?"
            choices = generate_choices(correct_pos)
            relative_qa_pairs.append({"question": question, "choices": choices, "answer": correct_pos})
            if len(relative_qa_pairs) >= rel_questions:
                break
        return qa_pairs + relative_qa_pairs

    @classmethod
    def reference(cls, figure: dict[str, Any]) -> list[dict[str, Any]]:
        qa_pairs: list[dict[str, Any]] = []
        shapes = figure["shapes"]
        counts = figure["counts"]

        # size-based questions
        attr = random.choice(["larger", "smaller"])
        # filter out line as it has no area
        shapes_n_line = [s for s in shapes if s["type"] != "line"]
        sorted_shapes = sorted(shapes_n_line, key=lambda s: s["area"], reverse=(attr == "larger"))
        appearance_shapes = {}
        for j in range(len(sorted_shapes)):
            shape_type = sorted_shapes[j]["type"]
            if shape_type not in appearance_shapes:
                appearance_shapes[shape_type] = []
            appearance_shapes[shape_type].append(j)
        appearance_shapes_keys = list(appearance_shapes.keys())
        size_qa_shapes = []
        for j in range(len(appearance_shapes_keys)):
            for i in range(len(appearance_shapes_keys)):
                if i == j:
                    continue
                shape_i_rank = appearance_shapes[appearance_shapes_keys[i]][-1]
                shape_j_rank = appearance_shapes[appearance_shapes_keys[j]][0]
                if (
                    shape_i_rank >= shape_j_rank
                    or abs(sorted_shapes[shape_i_rank]["area"] - sorted_shapes[shape_j_rank]["area"]) < area_type_t
                ):
                    continue
                answer_type = appearance_shapes_keys[i]
                anchor_type = appearance_shapes_keys[j]
                choices_types = []
                for k in range(len(appearance_shapes_keys)):
                    if k == i or k == j:
                        continue
                    shape_j_rank = appearance_shapes[appearance_shapes_keys[j]][-1]
                    shape_k_rank = appearance_shapes[appearance_shapes_keys[k]][0]
                    if (
                        shape_j_rank >= shape_k_rank
                        or abs(sorted_shapes[shape_j_rank]["area"] - sorted_shapes[shape_k_rank]["area"]) < area_type_t
                    ):
                        continue
                    choices_types.append(appearance_shapes_keys[k])
                size_qa_shapes.append((answer_type, anchor_type, choices_types))
        size_freq_qa_pairs = []
        if len(size_qa_shapes) > 0:
            answer_type, anchor_type, choices_types = random.choice(size_qa_shapes)
            answer_type = cls.clarify_hierarchical_text(answer_type, figure["counts"], "size")
            if counts[anchor_type] == 1:
                anchor_type = "the " + cls.clarify_hierarchical_text(anchor_type, figure["counts"], "size")
            else:
                anchor_type = cls.clarify_hierarchical_text(anchor_type, figure["counts"], "size")
                if " (" in anchor_type:
                    anchor_type = anchor_type.replace(" (", "s (")
                else:
                    anchor_type += "s"
                anchor_type = "all " + anchor_type
            choices_types = (
                [
                    cls.clarify_hierarchical_text(choices_type, figure["counts"], "size")
                    for choices_type in choices_types
                ]
                + [
                    cls.clarify_hierarchical_text(t, figure["counts"], "size")
                    for t in filter(lambda x: x != "line", cls.total_shapes)
                    if t not in counts
                ]
            )[:3]
            choices_types.append(answer_type)
            random.shuffle(choices_types)
            size_freq_qa_pairs.append(
                {
                    "question": f"Which of the following shapes presented in the image is {attr} than {anchor_type} in the image?",
                    "choices": choices_types,
                    "answer": answer_type,
                }
            )

        # Frequency-based questions
        attr: str = random.choice(["more", "less"])
        sorted_types = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=(attr == "more"))
        freq_qa_shapes = []
        for _j, (shape_type_j, freq_j) in enumerate(sorted_types):
            for shape_type_i, freq_i in sorted_types[:_j]:
                if freq_i == freq_j:
                    break
                answer_type = shape_type_i
                anchor_type = shape_type_j
                choices_types = [x[0] for x in sorted_types[_j + 1 :]]
                freq_qa_shapes.append((answer_type, anchor_type, choices_types))
        if len(freq_qa_shapes) > 0:
            answer_type, anchor_type, choices_types = random.choice(freq_qa_shapes)
            answer_type = cls.clarify_hierarchical_text(answer_type, figure["counts"], "size")
            anchor_type = cls.clarify_hierarchical_text(anchor_type, figure["counts"], "size")
            if " (" in anchor_type:
                anchor_type = anchor_type.replace(" (", "s (")
            else:
                anchor_type += "s"
            choices_types = (
                [
                    cls.clarify_hierarchical_text(choices_type, figure["counts"], "counting")
                    for choices_type in choices_types
                ]
                + [
                    cls.clarify_hierarchical_text(t, figure["counts"], "counting")
                    for t in cls.total_shapes
                    if t not in counts
                ]
            )[:3]
            choices_types.append(answer_type)
            random.shuffle(choices_types)
            size_freq_qa_pairs.append(
                {
                    "question": f"Which of the following shapes presented in the image appears {attr} frequently than {anchor_type} in the image?",
                    "choices": choices_types,
                    "answer": answer_type,
                }
            )

        # Location-based questions (relatively & absolutely comparison)
        distinguish_threshold = (
            vqa_args.distinguish_threshold_of_relative_direction
        )  # The minimum distance between two shapes.
        deviation_threshold = (
            vqa_args.deviation_threshold_of_relative_direction
        )  # The maximum deviation angle between the direction of the anchor shape related to the answer shape && the direction of vec.
        exclusiv_deviation_threshold = (
            vqa_args.exclusiv_deviation_threshold_of_relative_direction
        )  # The minimum deviation angle between the direction of the shape (excluding the answer shape) related to the anchor shape && the direction of vec.
        #     acquire all shapes that appears only once
        candidate_types = []
        for shape, freq in counts.items():
            if shape in ["line"]:
                continue
            if freq == 1:
                candidate_types.append(shape)
        candidate_shapes = [shape for shape in figure["shapes"] if shape["type"] in candidate_types]
        #     acquire the circles and ellipses (not circle) if they all are concentric (and counts more than one)
        _circle_specs = {}
        _ellipse_specs = {}
        for shape in figure["shapes"]:
            if shape["type"] == "circle":
                _spec = tuple(shape["center"])
                if _spec not in _circle_specs:
                    _circle_specs[_spec] = []
                _circle_specs[_spec].append(shape)
            elif shape["type"] == "ellipse":
                _spec = (tuple(shape["center"]), shape["rotation"])
                if _spec not in _ellipse_specs:
                    _ellipse_specs[_spec] = []
                _ellipse_specs[_spec].append(shape)
        if len(_circle_specs) == 1:
            _circles = list(_circle_specs.values())[0]
            if len(_circles) > 1:
                candidate_shapes.append(max(_circles, key=lambda x: x["major_axis"]))
        if len(_ellipse_specs) > 1:
            _ellipses = list(_ellipse_specs.values())[0]
            if len(_ellipses) > 1:
                candidate_shapes.append(max(_ellipses, key=lambda x: x["major_axis"]))
        #     generate qa pairs
        direction_qa_pairs = []
        if len(candidate_shapes) >= 3:
            direction_qa_pairs = []
            proj_para_func = lambda shape: shape["center"][0] * vec[0] + shape["center"][1] * vec[1]
            proj_perp_func = lambda shape: shape["center"][0] * vec[1] - shape["center"][1] * vec[0]
            for direction, vec in vqa_args.relative_direction_text_and_vector_dict.items():
                sorted_shapes_data = sorted(
                    map(lambda shape: (proj_para_func(shape), proj_perp_func(shape), shape), candidate_shapes),
                    key=lambda info: (info[0], info[1]),
                    reverse=True,
                )
                answer_proj_para_length, answer_proj_perp_length, answer_shape = sorted_shapes_data[0]
                answer_shape_box = answer_shape["box"]
                answer_shape_box_area = (answer_shape_box[1][0] - answer_shape_box[0][0]) * (
                    answer_shape_box[0][1] - answer_shape_box[1][1]
                )
                anchor_shape = None
                for _j, (proj_para_length, proj_perp_length, shape) in enumerate(sorted_shapes_data):
                    if _j == 0:
                        continue
                    if (answer_proj_para_length - proj_para_length) < distinguish_threshold:
                        continue
                    _angle = np.arctan2(
                        abs((answer_proj_perp_length - proj_perp_length)),
                        abs((answer_proj_para_length - proj_para_length)),
                    )
                    if _angle > deviation_threshold:
                        continue
                    __flag = False
                    for shape1_id, shape2_id, rel in figure["relations"]:
                        shape1 = figure["shapes"][shape1_id]["type"]
                        shape2 = figure["shapes"][shape2_id]["type"]
                        if (shape1 == answer_shape["type"] and shape2 == shape["type"]) or (
                            shape2 == answer_shape["type"] and shape1 == shape["type"]
                        ):
                            __flag = True
                            break
                    if __flag:
                        continue
                    for _tmp_proj_para_length, _tmp_proj_perp_length, _tmp_shape in sorted_shapes_data[1:_j]:
                        _angle = np.arctan2(
                            abs((_tmp_proj_perp_length - proj_perp_length)),
                            abs((_tmp_proj_para_length - proj_para_length)),
                        )
                        if _angle < exclusiv_deviation_threshold:
                            __flag = True
                    if __flag:
                        continue
                    shape_box = shape["box"]
                    shape_box_area = (shape_box[1][0] - shape_box[0][0]) * (shape_box[0][1] - shape_box[1][1])
                    overlap_box_area = overlap_area(answer_shape_box, shape_box)
                    _ratio_1 = overlap_box_area / shape_box_area
                    _ratio_2 = overlap_box_area / answer_shape_box_area
                    if _ratio_1 > 0.5 or _ratio_2 > 0.5:
                        continue
                    # print(_ratio_1, _ratio_2)
                    anchor_shape = shape
                    break
                else:
                    continue
                if anchor_shape is None:
                    continue
                answer_type = cls.clarify_hierarchical_text(
                    answer_shape["type"], list(figure["counts"].keys()), "location"
                )
                anchor_type = cls.clarify_hierarchical_text(
                    anchor_shape["type"], list(figure["counts"].keys()), "location"
                )
                candidate_types_1 = [
                    cls.clarify_hierarchical_text(shape["type"], list(figure["counts"].keys()), "location")
                    for shape in candidate_shapes
                ]
                candidate_types_2 = [
                    cls.clarify_hierarchical_text(t, list(figure["counts"].keys()), "location")
                    for t in cls.total_shapes
                    if t not in counts
                ]
                candidate_types = (candidate_types_1 + candidate_types_2)[:5]
                if answer_type in candidate_types:
                    candidate_types.remove(answer_type)
                if anchor_type in candidate_types:
                    candidate_types.remove(anchor_type)
                candidate_types = candidate_types[:3]
                candidate_types.append(answer_type)
                random.shuffle(candidate_types)
                direction_qa_pairs.append(
                    {
                        "question": f"Which of the following shapes is {direction} the {anchor_type}?",
                        "choices": candidate_types,
                        "answer": answer_type,
                    }
                )

        distinguish_threshold = vqa_args.distinguish_threshold_of_absolute_direction
        for direction, part_box in vqa_args.absolute_direction_text_and_box_dict.items():
            exclusiv_part_box = (
                (part_box[0][0] - 2 * distinguish_threshold, part_box[0][1] + 2 * distinguish_threshold),
                (part_box[1][0] + 2 * distinguish_threshold, part_box[1][1] - 2 * distinguish_threshold),
            )
            candidates_types = set()
            nonchoices_types = set()
            exclusiv_types = set()
            for shape in figure["shapes"]:
                shape_box = (
                    (shape["box"][0][0] - 0.0001, shape["box"][0][1] + 0.0001),
                    (shape["box"][1][0] + 0.0001, shape["box"][1][1] - 0.0001),
                )
                shape_box_area = abs((shape_box[1][0] - shape_box[0][0]) * (shape_box[0][1] - shape_box[1][1]))
                inclusiv_ratio = overlap_area(part_box, shape_box) / shape_box_area
                exclusiv_ratio = overlap_area(exclusiv_part_box, shape_box) / shape_box_area
                if (
                    exclusiv_ratio >= 0.9999
                    and inclusiv_ratio >= vqa_args.inclusiv_overlapping_threshold_of_absolute_direction
                ):
                    candidates_types.add(shape["type"])
                if exclusiv_ratio >= 0.5:
                    nonchoices_types.add(shape["type"])
                else:
                    exclusiv_types.add(shape["type"])
            for shape_type in exclusiv_types:
                if shape_type in candidates_types:
                    candidates_types.remove(shape_type)
            if len(candidates_types) == 0:
                continue
            candidates_types = list(candidates_types)
            random.shuffle(candidates_types)
            answer_type = candidates_types.pop(0)
            choices_types = (
                [
                    shape_type
                    for shape_type in cls.total_shapes
                    if shape_type in counts and shape_type not in nonchoices_types
                ]
                + [shape_type for shape_type in cls.total_shapes if shape_type not in counts]
            )[:3]
            choices_types = [
                cls.clarify_hierarchical_text(shape_type, list(figure["counts"].keys()), "location")
                for shape_type in choices_types
            ]
            answer_type = cls.clarify_hierarchical_text(answer_type, list(figure["counts"].keys()), "location")
            choices_types.append(answer_type)
            random.shuffle(choices_types)
            direction_qa_pairs.append(
                {
                    "question": f"Which of the following shapes is {direction}?",
                    "choices": choices_types,
                    "answer": answer_type,
                }
            )

        different_answer_qa_pairs = {}
        for _qa_pair in direction_qa_pairs:
            if _qa_pair["answer"] not in different_answer_qa_pairs:
                different_answer_qa_pairs[_qa_pair["answer"]] = []
            different_answer_qa_pairs[_qa_pair["answer"]].append(_qa_pair)
        if "line" in different_answer_qa_pairs:
            del different_answer_qa_pairs["line"]
        if len(different_answer_qa_pairs) <= 2:
            direction_qa_pairs = list(different_answer_qa_pairs.values())
        else:
            direction_qa_pairs = []
            for key in random.sample(list(different_answer_qa_pairs.keys()), 2):
                direction_qa_pairs.append(different_answer_qa_pairs[key])
        direction_qa_pairs = [random.choice(_qa_pairs) for _qa_pairs in direction_qa_pairs]

        # Generate choices prioritizing types in the image and ensuring no other correct answers are included
        for qa in qa_pairs:
            image_types = [t for t in counts.keys() if t not in qa["exclude_types"]]
            remaining_types = [t for t in cls.total_shapes if t not in counts]
            wrong_choices = (image_types[:3] + random.sample(remaining_types, max(0, 3 - len(image_types))))[:3]
            choices = [qa["answer"]] + wrong_choices
            random.shuffle(choices)
            qa["choices"] = choices
            del qa["exclude_types"]

        qa_pairs.extend(direction_qa_pairs + size_freq_qa_pairs)
        return random.sample(qa_pairs, k=min(vqa_args.max_q_ip, len(qa_pairs)))

    @classmethod
    def existence(cls, figure: dict[str, Any]) -> list[dict[str, Any]]:
        "is there a [shape] in the image?"
        qa_pairs: list[dict[str, Any]] = []
        counts = figure["counts"]

        all_present_types: list[str] = list(counts.keys())
        all_absent_types: list[str] = [t for t in cls.total_shapes if t not in counts]

        # 1. Ask whether two shapes exist. Only preserve 2 questions.
        question_types = random.sample(("TT", "TF", "FT", "FF"), k=2)
        # 1-1. Ask about two shape that both exist
        if "TT" in question_types and len(all_present_types) >= 2:
            present_types = random.sample(all_present_types, k=2)
            clarified_types = [
                cls.clarify_hierarchical_text(present_type, list(figure["counts"].keys()), "existence")
                for present_type in present_types
            ]
            _idx: int = random.randint(0, 1)
            qa = {
                "question": f"Is there a {clarified_types[0]} and a {clarified_types[1]} in the image?",
                "choices": [
                    "Yes, both exist.",
                    f"No, only the {clarified_types[_idx]} exists.",
                    f"No, only the {clarified_types[1 - _idx]} exists.",
                    "No, neither exists.",
                ],
                "answer": "Yes, both exist.",
            }
            qa_pairs.append(qa)

        # 1-2. Ask about two shape that neither exists
        if "FF" in question_types and len(all_absent_types) >= 2:
            absent_types = random.sample(all_absent_types, k=2)
            clarified_types = [
                cls.clarify_hierarchical_text(absent_type, list(figure["counts"].keys()), "existence")
                for absent_type in absent_types
            ]
            _idx: int = random.randint(0, 1)
            qa = {
                "question": f"Is there a {clarified_types[0]} and a {clarified_types[1]} in the image?",
                "choices": [
                    "Yes, both exist.",
                    f"No, only the {clarified_types[_idx]} exists.",
                    f"No, only the {clarified_types[1 - _idx]} exists.",
                    "No, neither exists.",
                ],
                "answer": "No, neither exists.",
            }
            qa_pairs.append(qa)

        # 1-3. Ask about two shape that only one exists
        if (
            ("TF" in question_types or "FT" in question_types)
            and len(all_present_types) >= 1
            and len(all_absent_types) >= 1
        ):
            present_type = random.choice(all_present_types)
            absent_type = random.choice(all_absent_types)
            clarified_types = [
                cls.clarify_hierarchical_text(present_type, list(figure["counts"].keys()), "existence"),
                cls.clarify_hierarchical_text(absent_type, list(figure["counts"].keys()), "existence"),
            ]
            _present_idx = random.randint(0, 1)
            if _present_idx:
                clarified_types.reverse()
            _idx: int = random.randint(0, 1)
            qa = {
                "question": f"Is there a {clarified_types[0]} and a {clarified_types[1]} in the image?",
                "choices": [
                    "Yes, both exist.",
                    f"No, only the {clarified_types[_idx]} exists.",
                    f"No, only the {clarified_types[1 - _idx]} exists.",
                    "No, neither exists.",
                ],
                "answer": f"No, only the {clarified_types[_present_idx]} exists.",
            }
            qa_pairs.append(qa)

        # 2. Multiple choice question about present or absent shape
        absent_types = all_absent_types
        can_ask_absent = len(counts) >= 3 and len(absent_types) >= 1  # Need 3 present + 1 absent
        can_ask_present = len(counts) >= 1 and len(absent_types) >= 3  # Need 1 present + 3 absent
        if can_ask_absent or can_ask_present:
            if can_ask_absent and can_ask_present:
                ask_absent = random.random() < 0.5
            else:
                ask_absent = can_ask_absent
            if ask_absent:
                present_types = random.sample(list(counts.keys()), 3)
                absent_type = random.choice(absent_types)
                choices = present_types + [absent_type]
                random.shuffle(choices)
                qa = {
                    "question": "Which of the following is absent in the image?",
                    "choices": choices,
                    "answer": absent_type,
                }
            else:
                present_type = random.choice(list(counts.keys()))
                absent_choices = random.sample(absent_types, 3)
                choices = [present_type] + absent_choices
                random.shuffle(choices)
                qa = {
                    "question": "Which of the following is present in the image?",
                    "choices": choices,
                    "answer": present_type,
                }
            qa_pairs.append(qa)

        return qa_pairs
