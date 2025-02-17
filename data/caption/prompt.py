# 1st start
context = """
The user will provide you with all the necessary information to describe a picture, which includes several geometric shapes. You need to generate a smooth and detailed descriptive text to depict the image, including the total number of shapes in the picture, what they are individually, their positions on the picture, and their relationships with each other. You should utilize all the information in the “relations” attribute in the results you generate. Your response should not include attributes that the user has not mentioned. Below are a few examples:

Example 1:
User Input:
{"total":3,"canvas":{"top":[{"name":"spiral"}],"bottom":[{"name":"right-angled triangle"}],"bottom-left":[{"name":"pentagon"}]}}
The descriptive text you should generate:
There are a total of 3 shapes in this image: a spiral at the top of the image, a right-angled triangle at the bottom of the image, and a pentagon in the bottom left corner of the image.

Example 2:
User Input:
{"total":3,"canvas":{"top-left":[{"name":"triangle"}],"bottom":[{"name":"circle","diameter":0.29}],"right":[{"name":"line"}]},"relations":["canvas::right::line is tangent to canvas::bottom::circle"]}
The descriptive text you should generate:
There are a total of 3 shapes in this image. In the top left corner, there is a triangle. Below the image, there is a circle with a diameter of 0.29 that is tangent to a line on the right side of the image.

Example 3:
User Input:
{"total":2,"canvas":{"bottom-right":[{"name":"rectangle"},{"name":"circle","diameter":0.14}]},"relations":["canvas::bottom-right::rectangle is inscribed within canvas::bottom-right::circle"]}
The descriptive text you should generate:
There are a total of 2 shapes in this image. In the bottom right corner, a rectangle is inscribed within a circle with a diameter of 0.14.

Example 4:
User Input:
{"total":7,"canvas":{"center":{"top-right":[{"name":"line"}],"center":[{"name":"line"}]},"top":[{"name":"segment","length":0.22}],"left":[{"name":"ellipse","majoraxis":0.34,"minoraxis":0.17},{"name":"square","side length":0.45}],"bottom-right":[{"concentric":{"the first shape from the inside,counting outward":{"name":"circle","diameter":0.1},"the second shape from the inside,counting outward":{"name":"circle","diameter":0.2}}}]},"relations":["canvas::center::top-right::line is parallel to canvas::top::segment","canvas::center::center::line is tangent to canvas::bottom-right::concentric::the second shape from the inside,counting outward::circle"]}
The descriptive text you should generate:
This image contains a total of 7 shapes: In the center of the image, there are two straight lines, with the one skewing towards the upper right being parallel to a line segment located at the top of the image, which has a length of 0.22. On the left side of the image, there is an ellipse with major and minor axes of 0.34 and 0.17, respectively, also there is a square measuring 0.45 in side length. Finally, in the bottom right corner of the image, there are two concentric circles with diameters of 0.1 and 0.2, respectively, where the outer circle is tangent to the straight line in the exact middle of the image.
"""
head_start_no_param_pool = [
    "Please provide a smooth and detailed description of the shapes in this image and their relationships. ",
    "Kindly furnish a coherent and elaborate account of the forms present in this picture and how they interact. ",
    "Please supply a fluid and comprehensive depiction of the figures within this image and their interconnections. ",
    "I would appreciate a well-structured and intricate narrative of the various shapes visible in the image and the nature of their associations. ",
    "Could you please offer a seamless and thorough explanation of the shapes seen in this image and the relationship between them? ",
    "It is requested that you deliver a contiguous and detailed description of the shapes in this image and their respective relationships. ",
    "Please render a lucid and elaborate description of the shapes that compose this image and the dynamics between them. ",
    "I need a clear and detailed account of the shapes in this image and how they relate to one another. ",
    "Offer a flowing and detailed depiction of the shapes within this image and the relationships that they share. ",
    "We require a well-ordered and explicit description of the shapes present in the image and the manner in which they relate. ",
    "Could you supply a contiguous and detailed account of the forms in this image and the connections between them? ",
]
head_start_with_param_part1_pool = [
    "I invite you to elaborate a fluid and precise narrative of the shapes in the image, detailing their interplay, ",
    "Please be so kind as to craft a comprehensive and clear description of the shapes within the image, ",
    "We ask that you generate a detailed and coherent explanation of the shapes observed in the image and the nature of their interactions, ",
    "It is imperative that you describe the shapes in the image and their relationships in a smooth and detailed manner, ",
    "Could you create a meticulous and well-flowing description of the shapes in the image, along with their interactions, ",
    "We would like a seamless and detailed depiction of the shapes in the image and how they relate to one another, ",
    "Please offer a nuanced and comprehensive description of the image's shapes and their relationships, ",
    "It is requested that you provide a clear and elaborate account of the shapes in the image and their interconnectedness, ",
    "We require a detailed and coherent description of the shapes within the image, including their relationships, ",
    "Please furnish a detailed and integrated description of the shapes in the image and the dynamics between them, ",
]
head_start_with_param_part2_pool = [
    "with numerical specifics for each shape based on its position. ",
    "including their relationships, as outlined by the numerical details that define their placements. ",
    "using the provided numerical data that categorizes them by their positions. ",
    "utilizing the numerical details given for each shape based on its position. ",
    "using the numerical details provided that distinguish them by their positioning. ",
    "considering the numerical information that characterizes each shape by its position. ",
    "with reference to the numerical details that classify them based on their placements. ",
    "as informed by the numerical specifics that mark their positions. ",
    "taking into account the numerical details that identify each shape by its location. ",
]
head_with_param_part1_pool = [
    "Here are some numerical specifics about certain shapes, identified by their positioning on the image. ",
    "The following numbers provide details about individual shapes, categorized by their locations. ",
    "Supplied here are some numerical specifics of individual shapes, noted by their position. ",
    "There are some numerical data points for particular shapes that are defined by their placement. ",
    "Enclosed are numerical specifics of several shapes, which are recognized by their position. ",
    "Here are some numerical details about select shapes, identified by their positions. ",
    "The following are numerical details for certain shapes that are characterized by their locations. ",
    "The numerical data provided here pertains to specific shapes, distinguished by their placement. ",
    "Below, there are numerical specifics for individual shapes, which are set apart by their positions. ",
    "Here are numerical details of some shapes, which are classified by their positioning. ",
]
head_end_pool = [
    "Your narrative should accurately portray the precise placement of the shapes on the canvas. ",
    "Your description should reflect the exact locations of the shapes on the artboard. ",
    "Be sure to illustrate the exact positioning of the shapes on the canvas in your description. ",
    "Your task is to represent the exact layout of the shapes on the canvas in your description. ",
    "Your description should render the actual arrangement of the shapes on the canvas. ",
    "Your goal is to delineate the exact positions of the shapes on the canvas. ",
    "Please ensure that your description accurately details the positions of the shapes on the canvas. ",
    "In your description, be sure to accurately convey where the shapes are situated on the canvas. ",
    "Your description should accurately map out the placement of the shapes on the canvas. ",
    "Be certain to capture the precise placement of the shapes on the canvas in your description. ",
    "Ensure your description reflects the exact positioning on the canvas. ",
    "Your depiction should accurately represent the shapes' locations on the canvas. ",
    "Your description should pinpoint the exact locations of the shapes on the canvas. ",
    "Your account must accurately convey the shapes' arrangement on the canvas. ",
    "Your aim should be to accurately depict the placement of the shapes on the canvas. ",
    "Your description should map out the precise locations of the shapes on the canvas. ",
    "Your narrative should reflect the accurate positioning of the shapes on the canvas. ",
    "Your description should accurately capture the shapes' positions on the canvas. ",
    "Your depiction should correctly represent where each shape is placed on the canvas.",
    "Your narrative should accurately detail the exact positioning of the shapes on the canvas. ",
]
# 1st end
# 2nd start
head_start_no_param_2nd = "The following is an image of a paleontological fossil, please provide a detailed description for the fossil image."
head_start_2nd = "The following is an image of a paleontological fossil, please provide a detailed description for the fossil image. Here is some information about the fossil that must be included in the description:"
# 2nd end
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
# head_not_full_part1_pool = [
#     "Nevertheless, the numerical data listed below is incomplete, ",
#     "Yet, the figures supplied here are not exhaustive, ",
#     "Nonetheless, the numerical details given below are lacking, ",
#     "However, the information presented below in numerical form is not comprehensive, ",
#     "In spite of this, the numerical content below is not fully detailed, ",
#     "Yet, the numerical data provided is insufficient, ",
#     "However, the numerical details here are not exhaustive, ",
#     "Despite this, the numerical information supplied is not thorough, ",
#     "Nonetheless, the numerical data listed is not complete, ",
# ]
# head_not_full_part2_pool = [
#     "and certain values for the graphs must still be extracted from the images. ",
#     "and additional numerical details from the graphs need to be interpreted from the pictures. ",
#     "with further data from the graphs required to be extracted from the visuals. ",
#     "and it is necessary to derive missing values from the graphical images. ",
#     "and some values for the graphs have to be obtained from the images. ",
#     "necessitating the extraction of remaining values from the depicted graphs. ",
#     "and more data from the graphs will need to be interpreted based on the images. ",
#     "and certain graph values must still be deduced from the visual representations. ",
#     "and there is a need to derive the remaining graph values from the images provided. ",
# ]
