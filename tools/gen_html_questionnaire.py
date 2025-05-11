import argparse
import base64
import json
import os
import random


def gen_html_questionnaire(path, prefix, id_range, dst_path):
    id_list = list(id_range)
    random.shuffle(id_list)
    with open(dst_path, "w", encoding="utf-8") as fpw, open(
        dst_path.removesuffix(".html") + "_answer.md", "w", encoding="utf-8"
    ) as fpw_a:
        print("<!DOCTYPE html>", file=fpw)
        print('<html lang="en">', file=fpw)
        print("<head>", file=fpw)
        print('<meta charset="UTF-8">', file=fpw)
        print(f"<title> Image Test ({prefix}) </title>", file=fpw)
        perspectives = [name.removesuffix(".jsonl") for name in os.listdir(path) if name.endswith(".jsonl")]
        listeners = ["all"]
        print(
            f"""
        <script>
            document.addEventListener('DOMContentLoaded', function() {{
                {listeners}.forEach(function(perspective) {{
                    var form = document.getElementById('form-' + perspective);
                    document.getElementById('submit-' + perspective).addEventListener('click', function(event) {{
                        event.preventDefault();
                        var fieldsets = form.querySelectorAll('fieldset');
                        var correct = 0;
                        var count = 0;
                        var message = ''; // 创建一个空字符串用于构建对话框消息
                        fieldsets.forEach(function(fieldset) {{
                            var radios = fieldset.querySelectorAll('input[type="radio"]:checked');
                            if (radios.length > 0) {{
                                var selectedValue = radios[0].value;
                                message += fieldset.name + ': ' + radios[0].id.slice(0, 1) + ' ' + selectedValue + '\\n';
                                correct += selectedValue === 'true' ? 1 : 0;
                                count += 1;
                            }} else {{
                                message += fieldset.name + ': -\\n';
                                count += 1;
                            }}
                        }});
                        var acc = (correct / count * 100).toFixed(2);
                        alert('Your accuracy in ' + perspective + 'is' + acc + '%');
                        message += 'total acc.: ' + acc + '%';
                        var blob = new Blob([message], {{type: 'text/plain'}});
                        var url = window.URL.createObjectURL(blob);
                        var a = document.createElement('a');
                        a.href = url;
                        a.download = 'humanresult-{prefix}-' + perspective + '.txt';
                        document.body.appendChild(a);
                        a.click();
                        window.URL.revokeObjectURL(url);
                    }});
                }});
            }});
        </script>
        """,
            file=fpw,
        )
        print("<style>\nbody{font-size: 24px;}", file=fpw)
        for image_id in id_list:
            with open(os.path.join(figures_path, f"{prefix}_{image_id:08}.jpg"), "rb") as fpr:
                base64code = base64.b64encode(fpr.read())
            print(
                f".img_{prefix}_{image_id} {{background-image: url(data:image/jpg;base64,{base64code.decode('ascii')}); width: 640px; height: 640px; background-size: contain; border: 2px solid skyblue; flex-shrink: 0;}}",
                file=fpw,
            )
        print("</style>", file=fpw)
        print("</head>", file=fpw)
        print("<body>", file=fpw)
        print('<form id="form-all">', file=fpw, end="\n\n")
        for __i, perspective in enumerate(perspectives):
            print(f"<h2> Perspective: {perspective} </h2>", file=fpw, end="\n\n")
            print(f"## {perspective}", file=fpw_a, end="\n\n")
            last_image_id = -1
            with open(os.path.join(path, perspective + ".jsonl"), "r", encoding="utf-8") as fpr:
                _cnt = 0
                while line := fpr.readline():
                    sample = json.loads(line.strip())
                    image_id = sample["image_id"]
                    if not (image_id in id_list[__i % len(perspectives) :: len(perspectives)]):
                        continue
                    question_id = sample["question_id"]
                    if image_id != last_image_id:
                        if last_image_id != -1:
                            print("</div></div>", file=fpw)
                        print(f"<h3> Image {image_id} ({prefix}) </h3>", file=fpw)
                        print('<div style="display: flex; align-items: flex-start; gap: 20px;">', file=fpw)
                        print(f'<div class="img_{prefix}_{image_id}"> </div><div>', file=fpw)
                    print(f'<fieldset name="question-{image_id}-{perspective}-{question_id}"> ', file=fpw)
                    print(f"<legend> {sample['question']} </legend>", file=fpw)
                    for i in range(4):
                        letter = chr(65 + i)
                        choice = sample["choices"][i]
                        is_answer = str(choice == sample["answer"]).lower()
                        print(
                            f'<div><input type="radio" id="{letter}-choice-{image_id}-{perspective}-{question_id}" name="question-{image_id}-{perspective}-{question_id}" value={is_answer} /> <label for="choice-{image_id}-{perspective}-{question_id}-{letter}">{letter}.{choice}</label></div>',
                            file=fpw,
                        )
                    print("</fieldset>", file=fpw, end="\n")
                    print(chr(65 + sample["choices"].index(sample["answer"])), file=fpw_a, end="\n")
                    last_image_id = image_id
                    _cnt += 1
                    if _cnt >= questions_per_perspective:  # At most 10 perspective
                        break
                print("</div></div>", file=fpw)
        print(
            '<button style="width: 300px; height: 200px; font-size: 32;" type="submit" id="submit-all"> Submit </button>',
            file=fpw,
        )
        print("</form>", file=fpw)
        print("</body>", file=fpw)


parser = argparse.ArgumentParser(description="Generate an HTML questionnaire.")
parser.add_argument("--perspectives", type=str, nargs="*", default=(), help="Which perspectives to be used")
parser.add_argument("--questions_path", type=str, required=True, help="Path to questions directory")
parser.add_argument(
    "--questions_per_perspective", type=str, default=10, help="The number of questions for each perspective"
)
parser.add_argument("--figures_path", type=str, required=True, help="Path to figures directory")
parser.add_argument("--figure_prefix", type=str, required=True, help="Prefix for figures (easy or hard)")
parser.add_argument("--start_idx", type=int, default=0, help="Start index for figure range")
parser.add_argument("--end_idx", type=int, required=True, help="End index for figure range")
parser.add_argument("--step_idx", type=int, default=1, help="Step size for figure range")
parser.add_argument("--dst_path", type=str, default=None, help="Step size for figure range")
args = parser.parse_args()

perspectives = args.perspectives
if perspectives is None:
    perspectives = ["counting", "existing", "location", "reference", "relation", "size"]
questions_per_perspective = 10
figures_path = args.figures_path
dst_path = args.dst_path
if dst_path is None:
    dst_path = f"./human-{args.figure_prefix}-questionaire.html"
if not dst_path.endswith("html"):
    dst_path += ".html"
gen_html_questionnaire(
    args.questions_path, args.figure_prefix, range(args.start_idx, args.end_idx, args.step_idx), dst_path
)
