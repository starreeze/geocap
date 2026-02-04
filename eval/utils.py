import re

characteristics = [
    "size",
    "shape",
    "equator",
    "lateral_slopes",
    "poles",
    "length",
    "width",
    "ratio",
    "axis_shape",
    "number_of_volutions",
    "coil_tightness",
    "height_of_volution",
    "thickness_of_spirotheca",
    # "endothyroid",
    "septa",
    "proloculus",
    "tunnel_shape",
    "tunnel_angles",
    "chomata",
    "axial_filling",
]
rule_based_eval_features = [
    "length",
    "width",
    "ratio",
    "axis_shape",
    "number_of_volutions",
    "proloculus",
    # "tunnel_shape",
    "tunnel_angles",
    "chomata",
]


def find_first_json_block(text: str) -> tuple[str, str]:
    """Find the first complete JSON block in the text.

    Args:
        text (str): Input text containing JSON block(s)

    Returns:
        tuple[str, str]: (matched JSON block, remaining text)

    Raises:
        ValueError: If no valid JSON block is found or text is malformed
    """
    stack = []
    start = -1

    for i, char in enumerate(text):
        if char in "[{":
            if not stack:  # First opening bracket
                start = i
            stack.append(char)
        elif char in "]}":
            if not stack:
                raise ValueError("Unmatched closing bracket")

            # Check if brackets match
            if (char == "]" and stack[-1] == "[") or (char == "}" and stack[-1] == "{"):
                stack.pop()
                if not stack:  # Found complete block
                    return text[start : i + 1], text[i + 1 :]
            else:
                raise ValueError("Mismatched brackets")

    if stack:
        raise ValueError("Unclosed brackets")
    raise ValueError("No JSON block found")


def extract_range_or_num(text: str) -> list[float] | str:
    """
    Extract numerical ranges from text strings, with unit conversion.

    Examples:
    - "3.1-5.5 mm" -> [3.1, 5.5]
    - "120 to 340 μm" -> [0.12, 0.34]  # converted to mm
    - "2.8-3.5, more frequently 3.0" -> [2.8, 3.5]
    - "near 1.5 in the first volution, increasing to near 2.5 at maturity" -> [1.5, 2.5]

    Returns a list of floats representing the range in mm, or an empty list if no range is found.
    """
    # Check if the text contains micrometer units
    is_micrometer = bool(re.search(r"(?:μ|\u03bc|microns?)", text))

    # Pattern for ratio format (e.g., "1:2.0, 1:2.5")
    ratio_pattern = r"1:(\d+\.?\d*)"
    match = re.search(ratio_pattern, text)
    if match:
        values = re.findall(ratio_pattern, text)
        return [float(v) for v in values]

    # Pattern for ratio format (e.g., "2.0:1, 2.5:1")
    ratio_pattern = r"(\d+\.?\d*):1"
    match = re.search(ratio_pattern, text)
    if match:
        values = re.findall(ratio_pattern, text)
        return [float(v) for v in values]

    # Pattern for ranges with hyphen (e.g., "3.1-5.5 mm")
    hyphen_pattern = r"(\d+\.?\d*)\s*-\s*(\d+\.?\d*)"
    match = re.search(hyphen_pattern, text)
    if match:
        values = [float(match.group(1)), float(match.group(2))]
        return [v * 0.001 if is_micrometer else v for v in values]

    # Pattern for ranges with "to" (e.g., "3.8 to 5.7 mm")
    to_pattern = r"(\d+\.?\d*)\s+to\s+(\d+\.?\d*)"
    match = re.search(to_pattern, text)
    if match:
        values = [float(match.group(1)), float(match.group(2))]
        return [v * 0.001 if is_micrometer else v for v in values]

    # Pattern for ranges with "or" (e.g., "6 or 7 mm")
    or_pattern = r"(\d+\.?\d*)\s+or\s+(\d+\.?\d*)"
    match = re.search(or_pattern, text)
    if match:
        values = [float(match.group(1)), float(match.group(2))]
        return [v * 0.001 if is_micrometer else v for v in values]

    # Pattern for "near X, increasing to near Y" type ranges
    near_pattern = r"near\s+(\d+\.?\d*).*?(?:increasing|decreasing)\s+to\s+(?:near\s+)?(\d+\.?\d*)"
    match = re.search(near_pattern, text)
    if match:
        values = [float(match.group(1)), float(match.group(2))]
        return [v * 0.001 if is_micrometer else v for v in values]

    # If no range is found, try to extract single numbers
    numbers = re.findall(r"(\d+\.?\d*)", text)
    if len(numbers):
        values = [float(number) for number in numbers]
        return [v * 0.001 if is_micrometer else v for v in values]

    return "no number found"


def calculate_score(ref_range: list[float] | str, pred_range: list[float] | str) -> int:
    if ref_range == "no number found":
        return -1

    # Process reference and prediction list
    if isinstance(ref_range, list):
        if pred_range == "no number found":
            return 0

        ref_low, ref_high = min(ref_range), max(ref_range)

    if isinstance(pred_range, list):
        # assert (
        #     len(pred_range) == 1
        # ), f"Invalid prediction: {pred_range}"  # assume the prediction is a accurate value
        # pred = pred_range[0]
        if len(pred_range) == 1:
            pred = pred_range[0]
        else:
            pred = sum(pred_range) / len(pred_range)

    def score_func(ref_low, ref_high, pred, thres_high=0.3, thres_low=0.05) -> int:
        # Calculate relative error
        if pred < ref_low:
            error = (ref_low - pred) / ref_low
        elif pred > ref_high:
            error = (pred - ref_high) / ref_high
        else:
            error = 0
        # Apply threshold
        if error > thres_high:
            error = 1
        elif error < thres_low:
            error = 0
        else:
            # Project error to [0, 1]
            error = (error - thres_low) / (thres_high - thres_low)
        return round(10 * (1 - error))

    return score_func(ref_low, ref_high, pred)


def extract_chomata(text, default_develop_value="absent") -> tuple[str, str, str]:
    develop_template = (
        r"absent|indistinct|weakly|poorly|faint|slight|observed|present|heavy|well|distinct|strongly"
    )
    develop_map = {
        "absent": "absent",
        "indistinct": "absent",
        "weakly": "weakly",
        "poorly": "weakly",
        "faint": "weakly",
        "slight": "weakly",
        "observed": "weakly",
        "present": "weakly",
        "heavy": "well",
        "well": "well",
        "distinct": "well",
        "strongly": "well",
    }
    develop = re.search(develop_template, text)
    if develop:
        develop = develop_map[develop.group(0)]
    else:
        develop = default_develop_value
    # Check for massive/small first as they affect both height and width
    is_massive = bool(re.search(r"\bmassive\b", text))
    is_small = bool(re.search(r"\b(small|tiny|minute)\b", text))

    height_template = r"\b(high|medium height|moderate height|low)\b"
    height_map = {"high": "high", "medium height": "moderate", "moderate height": "moderate", "low": "low"}
    height = re.search(height_template, text)
    if height:
        height = height_map[height.group(0)]
    elif is_massive:
        height = "high"
    elif is_small:
        height = "low"
    else:
        height = "moderate"

    width_template = r"\b(wide|broad|medium width|moderate width|narrow)\b"
    width_map = {
        "wide": "wide",
        "broad": "wide",
        "medium width": "moderate",
        "moderate width": "moderate",
        "narrow": "narrow",
    }
    width = re.search(width_template, text)
    if width:
        width = width_map[width.group(0)]
    elif is_massive:
        width = "wide"
    elif is_small:
        width = "narrow"
    else:
        width = "moderate"
    return develop, height, width


def calculate_chomata_rating(output, reference) -> float:
    develop_output, height_output, width_output = extract_chomata(output, default_develop_value="none")
    develop_reference, height_reference, width_reference = extract_chomata(
        reference, default_develop_value="none"
    )
    rating = 0
    # develop score
    if develop_reference == develop_output:
        rating += 4
    elif develop_output == "absent" or develop_reference == "absent":
        return 0
    elif develop_reference == "none":
        rating += 4
    # height score
    if height_output == height_reference:
        rating += 3
    elif height_output == "moderate" or height_reference == "moderate":
        rating += 1
    # width score
    if width_output == width_reference:
        rating += 3
    elif width_output == "moderate" or width_reference == "moderate":
        rating += 1
    return rating


def extract_tunnel_shape(text, default_value="moderate"):
    height_template = r"\b(high|medium height|moderate height|low)\b"
    height_map = {"high": "high", "medium height": "moderate", "moderate height": "moderate", "low": "low"}
    width_template = r"\b(wide|broad|medium width|moderate width|narrow)\b"
    width_map = {
        "wide": "wide",
        "broad": "wide",
        "medium width": "moderate",
        "moderate width": "moderate",
        "narrow": "narrow",
    }
    height = re.search(height_template, text)
    width = re.search(width_template, text)
    height = height_map[height.group(0)] if height else default_value
    width = width_map[width.group(0)] if width else default_value
    return height, width


def extract_axis_shape(text, default_value="straight"):
    axis_template = r"\b(straight|convex|concave|irregular|curved|sinuous)\b"
    axis = re.search(axis_template, text)
    axis = axis.group(0) if axis else default_value
    return axis


def calculate_axis_shape_rating(output, reference) -> float:
    axis_output = extract_axis_shape(output, default_value="none")
    axis_reference = extract_axis_shape(reference, default_value="straight")
    if axis_reference == axis_output:
        return 10
    elif axis_output in ["convex", "concave", "curved"] and axis_reference in ["convex", "concave", "curved"]:
        return 10
    elif axis_output in ["irregular", "sinuous"] and axis_reference in ["irregular", "sinuous"]:
        return 10
    else:
        return 0


# test extract_range_or_num
if __name__ == "__main__":
    print(extract_range_or_num("20\u00b0 in the second whorl to 30\u00b0-40\u00b0in the fifth"))
    print(extract_range_or_num("about 22 degrees"))
    print(extract_range_or_num("20\u00b0 and 25\u00b0"))
    print(extract_range_or_num("27, 27, and 31 degrees"))
    print(extract_range_or_num("moderately large but increases at a relatively slow rate"))
