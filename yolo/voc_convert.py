import os
import xmltodict


def voc_to_yolo(annotation_dir, my_annotation_dir, classes):
    """把 VOC XML 标注转成自定义模型使用的 txt 文件 (cx cy w h one_hot...)"""
    os.makedirs(my_annotation_dir, exist_ok=True)

    xml_files = sorted(f for f in os.listdir(annotation_dir) if f.endswith(".xml"))

    for name in xml_files:
        xml_path = os.path.join(annotation_dir, name)
        my_file_path = os.path.join(
            my_annotation_dir, os.path.splitext(name)[0] + ".txt"
        )
        with open(xml_path, "r", encoding="utf-8") as f:
            data = xmltodict.parse(f.read())

        annotation = data["annotation"]
        file_width = float(annotation["size"]["width"])
        file_height = float(annotation["size"]["height"])
        objects = annotation.get("object", [])
        if isinstance(objects, dict):
            objects = [objects]

        my_line = []
        for obj in objects:
            if obj["name"] not in classes:
                continue
            label_index = classes.index(obj["name"])
            xmin = float(obj["bndbox"]["xmin"])
            xmax = float(obj["bndbox"]["xmax"])
            ymin = float(obj["bndbox"]["ymin"])
            ymax = float(obj["bndbox"]["ymax"])

            x_center = (xmin + xmax) / 2.0 / file_width
            y_center = (ymin + ymax) / 2.0 / file_height
            width = (xmax - xmin) / file_width
            height = (ymax - ymin) / file_height

            one_hot = [0] * len(classes)
            one_hot[label_index] = 1
            one_hot_str = " ".join(str(h) for h in one_hot)

            my_line.append(
                f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {one_hot_str}"
            )
        if my_line:
            with open(my_file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(my_line))
        elif os.path.exists(my_file_path):
            os.remove(my_file_path)
