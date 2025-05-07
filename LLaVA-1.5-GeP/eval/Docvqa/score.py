# -*- coding: utf-8 -*-
import argparse
import json
import re
from collections import Counter

import Levenshtein


def normalize_text(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^\w\s]", "", sentence)
    return sentence


def calculate_anls(candidate, reference, tau=0.5):
    candidate = normalize_text(candidate)
    reference = normalize_text(reference)

    if not reference:
        return 0

    levenshtein_distance = Levenshtein.distance(candidate, reference)
    max_length = max(len(reference), 1)
    normalized_score = 1 - (levenshtein_distance / max_length)

    return normalized_score if normalized_score >= tau else 0


def process_file(file_path):
    yes_no_score = 0
    yes_no_count = 0
    non_binary_score = 0
    non_binary_count = 0

    with open(file_path, "r") as f:
        data = json.load(f)

        for entry in data:
            label = entry.get("label", "").strip()
            response = entry.get("response", "").strip()

            anls_score = calculate_anls(response, label)

            non_binary_score += anls_score
            non_binary_count += 1

    yes_no_average_score = yes_no_score / yes_no_count if yes_no_count > 0 else 0
    non_binary_average_score = non_binary_score / non_binary_count if non_binary_count > 0 else 0

    return yes_no_average_score, non_binary_average_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()

    yes_no_score, non_binary_score = process_file(args.input_file)
    average_score = (yes_no_score + non_binary_score) / 2
    print(f"Score: {average_score:.4f}")


if __name__ == "__main__":
    main()
