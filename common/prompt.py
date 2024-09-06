context = """
The user will provide you with all the necessary information to describe a picture, which includes several geometric shapes. You need to generate a smooth and detailed descriptive text to depict the image, including the total number of shapes in the picture, what they are individually, their positions on the picture, and their relationships with each other. Your response should not include attributes that the user has not mentioned. Below are a few examples:

Example 1:
User Input:
TOTAL SHAPES: 3
BEGIN SHAPES
top_left triangle
bottom circle | diameter: 0.29
right line
END SHAPES
BEGIN RELATIONS
bottom circle, right line | tangent
END RELATIONS
The descriptive text you should generate:
This image contains three shapes: a triangle, a circle, and a straight line. The triangle is in the upper left corner. Below, there is a circle with a diameter of 0.29 that is tangent to a straight line on the right.

Example 2:
User Input:
TOTAL SHAPES: 2
BEGIN SHAPES
top_right circle | diameter: 0.14
bottom_right triangle
END SHAPES
BEGIN RELATIONS
bottom_right triangle, top_right circle | inscribed
END RELATIONS
The descriptive text you should generate:
This image contains two shapes: a triangle and a circle. The circle has a diameter of 0.14, and the triangle is inscribed within the circle.

Example 3:
User Input:
TOTAL SHAPES: 4
BEGIN SHAPES
top ray
top_left segment | length: 0.71
bottom ellipse | major_axis: 0.95 | minor_axis: 0.63
center spiral
END SHAPES
BEGIN RELATIONS
END RELATIONS
The descriptive text you should generate:
This image contains four shapes: in the upper left corner, there is a line segment with a length of 0.71; above, there is a ray; in the center, there is a spiral; and below, there is an ellipse with a major axis of 0.95 and a minor axis of 0.63.
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
    "Please offer a nuanced and comprehensive description of the image’s shapes and their relationships, ",
    "It is requested that you provide a clear and elaborate account of the shapes in the image and their interconnectedness, ",
    "We require a detailed and coherent description of the shapes within the image, including their relationships, ",
    "Please furnish a detailed and integrated description of the shapes in the image and the dynamics between them, ",
]
head_start_with_param_part2_pool = [
    "with numerical specifics for each shape based on its relative position. ",
    "including their relationships, as outlined by the numerical details that define their relative placements. ",
    "using the provided numerical data that categorizes them by their relative positions. ",
    "utilizing the numerical details given for each shape based on its relative position. ",
    "taking into account the numerical specifics that identify them by their relative locations? ",
    "using the numerical details provided that distinguish them by their relative positioning. ",
    "considering the numerical information that characterizes each shape by its relative position. ",
    "with reference to the numerical details that classify them based on their relative placements. ",
    "as informed by the numerical specifics that mark their relative positions. ",
    "taking into account the numerical details that identify each shape by its relative location. ",
]
head_with_param_part1_pool = [
    "Here are some numerical specifics about certain shapes, identified by their positioning in relation to one another. ",
    "The following numbers provide details about individual shapes, categorized by their comparative locations. ",
    "Supplied here are some numerical specifics of individual shapes, noted by their position relative to the others. ",
    "There are some numerical data points for particular shapes that are defined by their relative placement. ",
    "Enclosed are numerical specifics of several shapes, which are recognized by their position in relation to one another. ",
    "Here are some numerical details about select shapes, identified by their relative positions. ",
    "The following are numerical details for certain shapes that are characterized by their relative locations. ",
    "The numerical data provided here pertains to specific shapes, distinguished by their relative placement. ",
    "Below, there are numerical specifics for individual shapes, which are set apart by their comparative positions. ",
    "Here are numerical details of some shapes, which are classified by their relative positioning. ",
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
    "Your depiction should accurately represent the shapes’ locations on the canvas. ",
    "Your description should pinpoint the exact locations of the shapes on the canvas. ",
    "Your account must accurately convey the shapes’ arrangement on the canvas. ",
    "Your aim should be to accurately depict the placement of the shapes on the canvas. ",
    "Your description should map out the precise locations of the shapes on the canvas. ",
    "Your narrative should reflect the accurate positioning of the shapes on the canvas. ",
    "Your description should accurately capture the shapes’ positions on the canvas. ",
    "Your depiction should correctly represent where each shape is placed on the canvas.",
    "Your narrative should accurately detail the exact positioning of the shapes on the canvas. ",
]
head_not_full_part1_pool = [
    "Nevertheless, the numerical data listed below is incomplete, ",
    "Yet, the figures supplied here are not exhaustive, ",
    "Nonetheless, the numerical details given below are lacking, ",
    "However, the information presented below in numerical form is not comprehensive, ",
    "In spite of this, the numerical content below is not fully detailed, ",
    "Yet, the numerical data provided is insufficient, ",
    "However, the numerical details here are not exhaustive, ",
    "Despite this, the numerical information supplied is not thorough, ",
    "Nonetheless, the numerical data listed is not complete, ",
]
head_not_full_part2_pool = [
    "and certain values for the graphs must still be extracted from the images. ",
    "and additional numerical details from the graphs need to be interpreted from the pictures. ",
    "with further data from the graphs required to be extracted from the visuals. ",
    "and it is necessary to derive missing values from the graphical images. ",
    "and some values for the graphs have to be obtained from the images. ",
    "necessitating the extraction of remaining values from the depicted graphs. ",
    "and more data from the graphs will need to be interpreted based on the images. ",
    "and certain graph values must still be deduced from the visual representations. ",
    "and there is a need to derive the remaining graph values from the images provided. ",
]
