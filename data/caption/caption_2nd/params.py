fusiform_classes = {"thickly fusiform": [0, 1.9], "fusiform": [2, 3], "elongate fusiform": [3.1, 99]}

ellipse_classes = {
    "ellipsoidal": [1.5, 2.1],
    "sub-ellipsoidal": [2.2, 4],
    "cylindrical": [4.1, 6],
    "sub-cylindrical": [6.1, 99],
}

shell_world_pixel = 1280

shell_pixel_div_mm = 136

shell_size_classes = {"small": [0, 10], "medium": [10, 20], "large": [20, 999]}

shell_equator_classes = {"inflated": [90, 9999], "normal": [0, 90]}

shell_slope_classes = {"convex": [0, 0.5], "concave": [0.5, 1]}

shell_pole_classes = {"bluntly rounded": [100, 999], "bluntly pointed": [80, 100], "elongated": [0, 80]}

shell_fusiform_pole_classes = {"bluntly rounded": [0, 6], "elongated": [6.1, 99]}

shell_axis_classes = {"straight": [178, 180], "other": [0, 178]}

proloculus_size_classes = {
    "very small": [0, 0.1],
    "small": [0.1, 0.2],
    "medium": [0.2, 0.3],
    "large": [0.3, 100],
}

proloculus_shape_classes = {"spherical": [0, 0.33], "normal": [0.33, 0.67], "kidney-shaped": [0.67, 1]}

volution_coiled_classes = {
    "tightly coiled": [0, 0.150],
    "moderately coiled": [0.150, 0.2],
    "lossely expanded": [0.2, 999],
}

tunnel_angle_classes = {"narrow": [0, 20], "moderate": [21, 30], "broad": [31, 99]}

chomata_size_classes = {"small": [0, 0.06], "moderate": [0.06, 0.1], "massive": [0.1, 9999]}

chomata_height_classes = {"low": [0, 0.4], "moderate": [0.4, 0.6], "high": [0.6, 999]}

chomata_width_classes = {"narrow": [0, 0.1], "moderate": [0.1, 0.2], "broad": [0.2, 999]}

chomata_development_classes = {
    "absence": [-1, 0.1],
    "absent in most volutions": [0.1, 1],
    "weakly developed": [1, 2],
    "present only in some volutions": [2, 3],
    "well developed": [3, 999],
}

deposit_development_classes = {"absence": [-1, 0.2], "normal": [0.2, 0.5], "well developed": [0.5, 999]}

septa_shape_classes = {
    "slightly fluted": lambda x: True,
    "straight": lambda x: x[0] >= 0.7,
    "undulant": lambda x: x[1] >= 0.7,
    "fluted": lambda x: x[1] + x[2] >= 0.7 and x[2] >= 0.35,
    "strongly fluted": lambda x: x[2] >= 0.7,
    "irregularly fluted": lambda x: x[2] >= 0.9,
}

ordinal_numbers = [
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth",
    "twentieth",
    "twenty-first",
    "twenty-second",
    "twenty-third",
    "twenty-fourth",
    "twenty-fifth",
    "twenty-sixth",
    "twenty-seventh",
    "twenty-eighth",
    "twenty-ninth",
    "thirtieth",
    "thirty-first",
    "thirty-second",
    "thirty-third",
    "thirty-fourth",
    "thirty-fifth",
    "thirty-sixth",
    "thirty-seventh",
    "thirty-eighth",
    "thirty-ninth",
    "fortieth",
    "forty-first",
    "forty-second",
    "forty-third",
    "forty-fourth",
    "forty-fifth",
    "forty-sixth",
    "forty-seventh",
    "forty-eighth",
    "forty-ninth",
    "fiftieth",
    "fifty-first",
    "fifty-second",
    "fifty-third",
    "fifty-fourth",
    "fifty-fifth",
    "fifty-sixth",
    "fifty-seventh",
    "fifty-eighth",
    "fifty-ninth",
    "sixtieth",
    "sixty-first",
    "sixty-second",
    "sixty-third",
    "sixty-fourth",
    "sixty-fifth",
    "sixty-sixth",
    "sixty-seventh",
    "sixty-eighth",
    "sixty-ninth",
    "seventieth",
    "seventy-first",
    "seventy-second",
    "seventy-third",
    "seventy-fourth",
    "seventy-fifth",
    "seventy-sixth",
    "seventy-seventh",
    "seventy-eighth",
    "seventy-ninth",
    "eightieth",
    "eighty-first",
    "eighty-second",
    "eighty-third",
    "eighty-fourth",
    "eighty-fifth",
    "eighty-sixth",
    "eighty-seventh",
    "eighty-eighth",
    "eighty-ninth",
    "ninetieth",
    "ninety-first",
    "ninety-second",
    "ninety-third",
    "ninety-fourth",
    "ninety-fifth",
    "ninety-sixth",
    "ninety-seventh",
    "ninety-eighth",
    "ninety-ninth",
    "one hundredth",
]
