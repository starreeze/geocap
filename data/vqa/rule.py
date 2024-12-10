# -*- coding: utf-8 -*-
# @Date    : 2024-12-08 10:54:09
# @Author  : Shangyu.Xing (starreeze@foxmail.com)

import random
from collections import Counter
from itertools import product
from typing import Any, cast

import numpy as np
from tqdm import tqdm

from common.args import logger, vqa_args
from data.rule.shapes import GSRule


class RuleBasedQAGenerator:
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
    shape_hierarchy = {
        "ellipse": ["circle"],
        "rectangle": ["square"],
        "quadrilateral": ["rectangle", "square"],
    }
    relation_reverse = {
        "tangent": "tangent",
        "parallel": "parallel",
        "circumscribed": "inscribed",
        "inscribed": "circumscribed",
        "shared edge": "shared edge",
        "diagonal": None,
        "major axis": None,
        "minor axis": None,
        "diameter": None,
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
                shape_info = {
                    "type": self.get_type(shape_dict),
                    "center": shape.get_centroid(),
                    "box": shape.get_bbox(),
                    "area": shape.get_area(),
                }
                info["shapes"].append(shape_info)
            info["relations"] = figure["relations"]
            info["counts"] = dict(Counter(shape["type"] for shape in info["shapes"]))
            self.data.append(info)

    @classmethod
    def clarify_hierarchical_choices(cls, qa: dict[str, Any]):
        "Post-process choices to clarify hierarchical types when parent and child types appear together."
        # however, for existence, we need to handle in the question generation
        if "choices" not in qa or not isinstance(qa["choices"][0], str):
            return
        choices: list[str] = qa["choices"]
        for parent, children in cls.shape_hierarchy.items():
            try:
                parent_idx = choices.index(parent)
            except ValueError:
                continue
            # Check if any child type exists in choices
            overlapping_children = [c for c in children if c in choices]
            if not overlapping_children:
                continue
            # Add clarification to parent
            choices[parent_idx] = f"{parent} ({', '.join(overlapping_children)} excluded)"
            # Update answer if needed
            if qa["answer"] == parent:
                qa["answer"] = choices[parent_idx]

    @classmethod
    def clarity_hierarchical_text(cls, type: str, image_types: list[str], perspective: str = "counting") -> str:
        if type not in cls.shape_hierarchy:
            return type
        overlapping_children = [c for c in cls.shape_hierarchy[type] if c in image_types]
        if not overlapping_children:
            return type
        child_desc = ", ".join(overlapping_children)
        return f"{type} ({'excluding' if perspective == 'counting' else 'not'} {child_desc})"

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
            clarified_type = cls.clarity_hierarchical_text(type, list(figure["counts"].keys()), "counting")
            question = f"How many {clarified_type}(s) are there in the image?"

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
            type1 = cls.clarity_hierarchical_text(pair[0], list(figure["counts"].keys()), "relation")
            type2 = cls.clarity_hierarchical_text(pair[1], list(figure["counts"].keys()), "relation")
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
        # exclude line as it has no area
        types = [k for k, v in figure["counts"].items() if v == 1 and k != "line"]

        qa_pairs: list[dict[str, Any]] = []
        dimensions = ["horizontal span", "vertical span", "area"]
        questions_per_dim = vqa_args.max_q_ip // 3
        remaining = vqa_args.max_q_ip % 3

        for i, dim in enumerate(dimensions):
            num_questions = questions_per_dim + (1 if remaining > 0 else 0)
            remaining -= 1
            sampled_types = random.sample(types, min(len(types), num_questions))

            for type in sampled_types:
                clarified_type = cls.clarity_hierarchical_text(type, list(figure["counts"].keys()), "size")
                shape = next(s for s in figure["shapes"] if s["type"] == type)
                if dim == "area":
                    correct_value = shape["area"]
                    question = f"which of the following is closest to the area of the {clarified_type}?"
                else:
                    correct_value = abs(shape["box"][1][i] - shape["box"][0][i])
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
                    if all(0 <= v <= 1 for v in test_choices):
                        position = pos
                        valid_choices = test_choices
                        break
                # If no valid position found, adjust factor to fit range
                if valid_choices is None:
                    factor = min(correct_value / 2, (1 - correct_value) / 2)
                    logger.warning(f"Adjusting factor to {factor} to fit range for {dim} of {clarified_type}")
                    logger.info(f"In image: {figure}")
                    position = random.randint(0, 3)
                    valid_choices = [correct_value + factor * (i - position) for i in range(4)]

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
            if abs(x - ref_x) < vqa_args.location_type_t and abs(y - ref_y) < vqa_args.location_type_t:
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
            clarified_type = cls.clarity_hierarchical_text(type, list(figure["counts"].keys()), "location")
            shape = next(s for s in figure["shapes"] if s["type"] == type)
            x, y = shape["center"]
            correct_pos = get_position(x, y)
            if correct_pos is None:
                continue
            question = f"Where is the {clarified_type} located in the image?"
            choices = generate_choices(correct_pos)
            qa_pairs.append({"question": question, "choices": choices, "answer": correct_pos})
            if len(qa_pairs) >= abs_questions:
                break

        # Generate relative position questions
        type_pairs = [p for p in product(types, repeat=2) if p[0] != p[1]]
        sampled_pairs = random.sample(type_pairs, len(type_pairs))
        relative_qa_pairs: list[dict[str, Any]] = []
        for type_a, type_b in sampled_pairs:
            type_a_clear = cls.clarity_hierarchical_text(type_a, list(figure["counts"].keys()), "location")
            type_b_clear = cls.clarity_hierarchical_text(type_b, list(figure["counts"].keys()), "location")
            shape_a = next(s for s in figure["shapes"] if s["type"] == type_a)
            shape_b = next(s for s in figure["shapes"] if s["type"] == type_b)
            x_a, y_a = shape_a["center"]
            x_b, y_b = shape_b["center"]
            correct_pos = get_position(x_a, y_a, x_b, y_b)
            if correct_pos is None:
                continue
            question = f"Where is the {type_a_clear} located relative to the {type_b_clear}?"
            choices = generate_choices(correct_pos)
            relative_qa_pairs.append({"question": question, "choices": choices, "answer": correct_pos})
            if len(relative_qa_pairs) >= rel_questions:
                break
        return qa_pairs + relative_qa_pairs

    @classmethod
    def reference(cls, figure: dict[str, Any]) -> list[dict[str, Any]]:
        "what's the type of [attribute] shape?"
        qa_pairs: list[dict[str, Any]] = []
        shapes = figure["shapes"]
        counts = figure["counts"]

        # Size-based questions (largest/smallest area)
        attr = random.choice(["largest", "smallest"])
        sorted_shapes = sorted(shapes, key=lambda s: s["area"], reverse=(attr == "largest"))
        max_area = sorted_shapes[0]["area"]
        all_correct_types = {s["type"] for s in sorted_shapes if abs(s["area"] - max_area) < vqa_args.area_type_t}
        correct_type = sorted_shapes[0]["type"]
        question = f"What type of shape presented in the image has the {attr} area?"
        qa_pairs.append({"question": question, "answer": correct_type, "exclude_types": all_correct_types})

        # Location-based questions
        loc_attrs = {
            "leftmost": lambda s: s["center"][0],
            "rightmost": lambda s: -s["center"][0],
            "uppermost": lambda s: s["center"][1],
            "lowermost": lambda s: -s["center"][1],
        }
        attr, key_func = random.choice(list(loc_attrs.items()))
        sorted_shapes = sorted(shapes, key=key_func)
        extreme_val = key_func(sorted_shapes[0])
        all_correct_types = {
            s["type"] for s in sorted_shapes if abs(key_func(s) - extreme_val) < vqa_args.location_type_t
        }
        correct_type = sorted_shapes[0]["type"]
        question = f"What type of shape presented in the image has the {attr} centroid?"
        qa_pairs.append({"question": question, "answer": correct_type, "exclude_types": all_correct_types})

        # Frequency-based questions
        attr = random.choice(["most", "least"])
        sorted_types = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=(attr == "most"))
        max_count = sorted_types[0][1]
        all_correct_types = {t for t, c in counts.items() if c == max_count}
        correct_type = sorted_types[0][0]
        question = f"What type of shape presented in the image appears {attr} frequently?"
        qa_pairs.append({"question": question, "answer": correct_type, "exclude_types": all_correct_types})

        # Generate choices prioritizing types in the image and ensuring no other correct answers are included
        for qa in qa_pairs:
            image_types = [t for t in counts.keys() if t not in qa["exclude_types"]]
            remaining_types = [t for t in cls.total_shapes if t not in counts]
            wrong_choices = (image_types[:3] + random.sample(remaining_types, max(0, 3 - len(image_types))))[:3]
            choices = [qa["answer"]] + wrong_choices
            random.shuffle(choices)
            qa["choices"] = choices
            del qa["exclude_types"]

        return qa_pairs

    @classmethod
    def existence(cls, figure: dict[str, Any]) -> list[dict[str, Any]]:
        "is there a [shape] in the image?"
        qa_pairs: list[dict[str, Any]] = []
        counts = figure["counts"]

        # 1. Ask about a shape that exists
        present_type = random.choice(list(counts.keys()))
        clarified_type = cls.clarity_hierarchical_text(present_type, list(figure["counts"].keys()), "existence")
        qa = {"question": f"Is there a {clarified_type} in the image?", "choices": ["yes", "no"], "answer": "yes"}
        qa_pairs.append(qa)

        # 2. Ask about a shape that doesn't exist
        absent_types = [t for t in cls.total_shapes if t not in counts]
        if absent_types:
            absent_type = random.choice(absent_types)
            clarified_type = cls.clarity_hierarchical_text(absent_type, list(figure["counts"].keys()), "existence")
            qa = {"question": f"Is there a {clarified_type} in the image?", "choices": ["yes", "no"], "answer": "no"}
            qa_pairs.append(qa)

        # 3. Multiple choice question about present or absent shape
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
