import json
import jsonlines

from tqdm import tqdm

# from common.llm import generator_mapping, model_path_mapping
# from common.llm import APIGenerator  # sry but now we do this
from common.llm import APIGenerator


def main():
    sys_prompt = """You are a helpful assistant. You will be provided with a fossil picture input and you should depict this picture. The user will provide you with the pixel information of the image and its actual length and width. You need to infer the real data and description based on the image and the user's input. Do not use sub-titles and lists to devide the output; instead, merge the paragraph into a whole. You do not need a conclusion, and you should try to mimic the example. The user will provide you with two examples, namely Example1 and Example2. You need to complete the output of "Your turn" based on the examples.
    """  # Test: No Pre-info & With Pre-info
    img01 = "Schwagerina_furoni_1_5.png"
    img02 = "Fusulinella_clarki_1_1.png"
    user_prompt0 = [
        (
            "text",
            """Example1:
User: The following is an image of a paleontological fossil, please provide a detailed description for the fossil image. The resolution of the fossil image is 517\u00d7302, and the width and height of the image are 5.687 mm and 3.322 mm, respectively.""",
        ),
        ("image", f"/home/nfs04/xingsy/foscap/dataset/common/images/{img01}"),
        (
            "text",
            """Assistant: Shell large, inflated fusiform, with sharply-pointed poles, steep slightly concave lateral slopes, and straight axis of coiling. Mature specimens of seven volutions measure 7.4 mm. in length and 3.75 mm, in width, giving a form ratio of 1:2.0. The shell retains essentially the same shape throughout growth of the individual. The form ratio is about 1:2 for the first volution to maturity. However, the lateral slopes of the first three to four volutions are convex, but the lateral slopes of the outer volutions of mature specimens become slightly concave. The proloculum is large and spherical in shape. Its outside diameter measures about 465 microns. The shell expands essentially uniformly up to about the fifth volution. The average heights of the chambers in the first to the seventh volution are 108, 171, 242, 314, 342, 363, and 324 microns, respectively. The septa are relatively thick. They are distinctly pendant-shaped in cross-section, but the lower margins of the septa are distinctly thicker immediately over the tunnel than poleward from the tunnel. The septa are narrowly fluted throughout. The spirotheca is thick and composed of a tectum and keriotheca with coarse alveoli. Secondary deposits continuous with the chomata and axial fillings cover the upper surfaces of the spirotheca in most parts of the inner volutions. The average thicknesses of the spirotheca in the first to the seventh volution measure about 50, 70, 90, 110, 150, 150, and 150 microns, respectively. The tunnel is low and of medium width, and its path is straight. The average tunnel angles of the second to the sixth volutions measure 20\u00b0, 20\u00b0, 22\u00b0, 24\u00b0, and 30\u00b0, respectively. Discontinuous irregular chomata border the tunnel in the second volution to the beginning part of the last volution. Dense calcite essentially fills the chambers in the polar regions of the second to the next to outermost volution.""",
        ),
        (
            "text",
            """Example2:
User: The following is an image of a paleontological fossil, please provide a detailed description for the fossil image. The resolution of the fossil image is 184\u00d791, and the width and height of the image are 1.012 mm and 0.500 mm, respectively.""",
        ),
        ("image", f"/home/nfs04/xingsy/foscap/dataset/common/images/{img02}"),
        (
            "text",
            """Assistant: The shell is small in size and inflated fusiform in shape, with almost straight axis of coiling, convex lateral slopes, and bluntly pointed poles. Larger specimens of six volutions measure about 2.6 mm. long and 1.05 mm. wide, giving form ratios of 2.48. The form ratios of the first to the sixth volution of the holotype specimen are 1.1, 1.5, 1.8, 2.2, 2.4, and 2.5, respectively.\\nThe proloculus is relatively large and has an outside diameter of about 79\u00b5. The shell expands almost uniformly. The heights of the chambers above the tunnel in the first to the sixth volution of the holotype are 38, 53, 70, 97, 131, and 160\u00b5, respectively. The chambers are lowest above the tunnel where the above measurements were made, and they increase in height rather uniformly toward the poles and very sharply in the extreme polar ends. The spirotheca is composed of the tectum and upper and lower tectoria in the inner four and one-half volutions and has a thin diaphanotheca below the tectum in the tunnel area of the outer volutions. The tectum and diaphanotheca \u00b5 about 10, 13, and 18 in the fourth to sixth volution, respectively. The septa are composed of the tectum and thick deposits on both sides. They are slightly fluted in the extreme polar ends but are about plane across the central part of the shell. The tunnel is narrow and of medium height, and its path is irregular. Averages of the tunnel angles are 14, 20, 25, and 24\u00b0, respectively. The chomata are very massive throughout the shell. They are more than half as high as the chambers midway between the septa but extend to the tops of the chambers immediately adjacent to the septa. Their tunnel sides are vertical to slightly overhanging, and their poleward slopes are very low, with their lateral edges ending near the poles in the second to fifth volution. In the outer one or two volutions the chomata are more narrow and have steep poleward slopes.""",
        ),
    ]
    generator = APIGenerator("gpt-4o", sys_prompt=sys_prompt)
    pics, questions, references = [], [], []
    with open("dataset/instructions_no_vis_gpt4o.jsonl", "r") as f:
        processed = 900
    with open("dataset/instructions_no_vis_tools.jsonl", "r") as f:
        with jsonlines.open(
            "dataset/instructions_no_vis_gpt4o_precisely.jsonl", mode="a", flush=True
        ) as writer:
            index = 0
            for line in tqdm(f):
                data = json.loads(line)
                if index < processed:
                    pass
                else:
                    inputs = user_prompt0 + [
                        ("text", "Your turn:"),
                        ("text", data["input"]),
                        ("image", f"/home/nfs04/xingsy/foscap/dataset/common/images/{data['image']}"),
                        ("text", "Assistant:"),
                    ]
                    resp = generator.get_one_response(inputs)
                    jsonls = {"image": data["image"], "question": data["input"], "response": resp}
                    print(jsonls)
                    writer.write(jsonls)
                index += 1


if __name__ == "__main__":
    main()
