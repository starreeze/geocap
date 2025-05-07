# -*- coding: utf-8 -*-
import argparse
import json
import re
from collections import Counter


def split_sentence(sentence):
    """Normalize and split sentence into words"""
    sentence = sentence.lower().strip()
    sentence = re.sub(r"[^\w\s]", "", sentence)
    words = sentence.split()
    return Counter(words)


def calculate_recall(candidate, reference):
    """Calculate recall between candidate and reference text"""
    candidate_words = split_sentence(candidate)
    reference_words = split_sentence(reference)

    count = sum(
        min(reference_words[word], candidate_words[word])
        for word in reference_words
        if word in candidate_words
    )
    total = sum(reference_words.values())

    return count / total if total > 0 else 0


def extract_first_word(response):
    """Extract and clean the first word from response"""
    response = response.strip().lower()
    response = re.sub(r"[^\w\s]", "", response)
    words = response.split()
    return words[0] if words else ""


def process_file(file_path):
    """Process evaluation file and calculate scores"""
    yes_no_score = 0
    yes_no_count = 0
    non_binary_score = 0
    non_binary_count = 0

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

        for entry in data:
            label = str(entry.get("label", "")).strip()
            response = str(entry.get("response", "")).strip()

            if label.lower() in {"yes", "no"}:
                first_word = extract_first_word(response)
                if first_word == label.lower():
                    yes_no_score += 1
                yes_no_count += 1
            else:
                recall = calculate_recall(response, label)
                non_binary_score += recall
                non_binary_count += 1

    yes_no_avg = yes_no_score / yes_no_count if yes_no_count > 0 else 0
    non_binary_avg = non_binary_score / non_binary_count if non_binary_count > 0 else 0

    return yes_no_avg, non_binary_avg


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    # Process the evaluation file
    yes_no_score, non_binary_score = process_file(args.input_file)

    # Print results
    print("\nEvaluation Results:")
    print(f"Closed Questions Accuracy: {yes_no_score:.4f}")
    print(f"Open Questions  Score: {non_binary_score:.4f}")


if __name__ == "__main__":
    main()
