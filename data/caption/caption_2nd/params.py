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

shell_pole_classes = {
    "bluntly rounded": [130, 999],
    "bluntly pointed": [110, 130],
    "sharply pointed": [0, 110],
}

shell_ellipse_pole_classes = {"bluntly rounded": [0, 4], "bluntly pointed": [4, 6], "elongated": [6, 99]}

shell_fusiform_pole_classes = {
    "bluntly rounded": [0, 6],
    "elongated": [6.1, 99],
    "sharply pointed": [100, 114],
}

shell_axis_classes = {"straight": [175, 180], "slightly": [165, 175], "": [155, 165], "strongly":[0, 155]}

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
    "loosely expanded": [0.2, 999],
}

tunnel_angle_classes = {"narrow": [0, 20], "moderate": [21, 30], "broad": [31, 99]}

tunnel_shape_regular_classes = {"regular":[0,0.5],"irregular":[0.5,99]}

tunnel_shape_regular_priority = {
    "regular": 0,
    "irregular": 1
}

chomata_size_classes = {
    "absent": [0, 0.0001],
    "small": [0.0001, 0.03],
    "moderate": [0.03, 0.05],
    "massive": [0.05, 9999],
}

chomata_height_classes = {
    "absent": [0, 0.0001],
    "low": [0.0001, 0.4],
    "moderate": [0.4, 0.6],
    "high": [0.6, 999],
}

chomata_width_classes = {
    "absent": [0, 0.0001],
    "narrow": [0.0001, 0.1],
    "moderate": [0.1, 0.2],
    "broad": [0.2, 999],
}

chomata_development_classes = {
    "absence": [-1, 0.1],
    "absent in most volutions": [0.1, 1],
    "weakly developed": [1, 2],
    "present only in some volutions": [2, 3],
    "well developed": [3, 999],
}

deposit_development_classes = {"absence": [-1, 0.2], "normal": [0.2, 0.5], "well developed": [0.5, 999]}

septa_shape_classes = {
    "straight": lambda x: x[0] <= 1,
    "slightly fluted": lambda x: x[0] > 1 and x[0] <= 6,
    "fluted": lambda x: x[0] > 6 and x[0] <= 10,
    "strongly fluted": lambda x: (x[0] > 10),
}

septa_size_difference_classes = {
    "straight": lambda x: x <= 0.003,
    "slightly fluted": lambda x: x > 0.003 and x <= 0.004,
    "fluted": lambda x: x > 0.004 and x <= 0.005,
    "strongly fluted": lambda x: (x > 0.005),
}

septa_shape_priority = {
    "straight": 0,
    "slightly fluted": 1,
    "fluted": 2,
    "strongly fluted": 3,
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
