import json
import re


def reorder_tag(input_path, output_path):
    with open(input_path, "r") as f:
        captions = [json.loads(line) for line in f]

    pre_defined_tags = [
        "<size>",
        "<shape>",
        "<equator>",
        "<lateral slopes>",
        "<poles>",
        "<axis>",
        "<length>",
        "<width>",
        "<ratio>",
        "<number of volutions>",
        "<coil tightness>",
        "<heights of volutions>",
        "<endothyroid>",
        "<spirotheca>",
        "<septa>",
        "<proloculus>",
        "<tunnel shape>",
        "<tunnel angle>",
        "<chomata>",
        "<axial filling>",
    ]

    for caption in captions:
        output = caption["output"]

        # 解析所有的标记句子
        # 使用正则表达式匹配 <tag>content</tag> 格式
        pattern = r"<([^>]+)>([^<]*?)</\1>"
        matches = re.findall(pattern, output)

        # 创建标签到内容列表的映射，处理重复标签
        tag_content_map = {}
        for tag, content in matches:
            tag_key = f"<{tag}>"
            if tag_key in tag_content_map:
                # 如果标签已存在，合并内容
                tag_content_map[tag_key].append(content.strip())
            else:
                tag_content_map[tag_key] = [content.strip()]

        # 按照预定义顺序重新排序
        reordered_output = ""
        for predefined_tag in pre_defined_tags:
            if predefined_tag in tag_content_map:
                tag_name = predefined_tag[1:-1]  # 去掉 < 和 >
                # 合并所有相同标签的内容
                combined_content = "".join(tag_content_map[predefined_tag])
                if combined_content:  # 只有当内容不为空时才添加
                    reordered_output += f"{predefined_tag}{combined_content}</{tag_name}> "

        # # 添加不在预定义列表中的其他标签
        # for tag_key, content_list in tag_content_map.items():
        #     if tag_key not in pre_defined_tags:
        #         tag_name = tag_key[1:-1]  # 去掉 < 和 >
        #         combined_content = "".join(content_list)
        #         if combined_content:  # 只有当内容不为空时才添加
        #             reordered_output += f"{tag_key}{combined_content}</{tag_name}> "

        # 更新caption的output
        caption["output"] = reordered_output.strip()

    # 写入输出文件
    with open(output_path, "w", encoding="utf-8") as f:
        for caption in captions:
            f.write(json.dumps(caption, ensure_ascii=False) + "\n")
