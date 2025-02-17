from common.args import eval_stage3_args
import json


def main():
    mode = eval_stage3_args.manual_fix_mode
    assert mode == "output" or mode == "reference"
    with open(f"extracted_{mode}_info.json", "r") as f:
        data: list[dict] = json.load(f)
    assert isinstance(data, list)
    index = eval_stage3_args.manual_fix_index
    assert index != -1
    content = eval_stage3_args.manual_fix_content
    try:
        patch = json.loads(content)
        assert isinstance(patch, dict)
    except Exception as e:
        print(e)
        return
    data.insert(index, patch)
    with open(f"extracted_{mode}_info.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    main()
