#!/usr/bin/env python3

import os
import sys
import json
import glob

def clean_nested_lines(lines):
    # Remove only empty sub-arrays, keep outer list even if empty
    return [line for line in lines if isinstance(line, list) and len(line) > 0]

def convert_ras_to_lps(annotations: list):
    for frame in annotations:
        if frame.get("coordinate_space", "RAS") == "RAS":
            for line_group in ["pleura_lines", "b_lines"]:
                for entry in frame.get(line_group, []):
                    points = entry["line"]["points"]
                    for point in points:
                        point[0] = -point[0]  # Negate X (Right → Left)
                        point[1] = -point[1]  # Negate Y (Anterior → Posterior)
            frame["coordinate_space"] = "LPS"  # Update coordinate_space

def transform_annotations(data, rater):
    annotations = data.get("frame_annotations")
    if isinstance(annotations, dict):
        new_list = []
        for k in sorted(annotations.keys(), key=lambda x: int(x)):
            frame = annotations[k]
            if "annotation" in frame:
                frame.update(frame.pop("annotation"))

            pleura = clean_nested_lines(frame.get("pleura_lines", []))
            b_lines = clean_nested_lines(frame.get("b_lines", []))

            frame["pleura_lines"] = [
                {"rater": rater, "line": {"points": coords}} for coords in pleura
            ]
            frame["b_lines"] = [
                {"rater": rater, "line": {"points": coords}} for coords in b_lines
            ]

            frame["frame_number"] = int(k) if k.isdigit() else k
            frame["coordinate_space"] = "RAS"
            new_list.append(frame)

        data["frame_annotations"] = new_list
        # write out as LPS
        convert_ras_to_lps(data.get("frame_annotations", []))

    return data

def extract_rater_from_path(path):
    rater = os.path.basename(os.path.normpath(path))
    if rater:
        return rater.lower()
    else:
        return None

def convert_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    json_paths = glob.glob(os.path.join(input_dir, "*", "*", "*.json"))

    for in_path in json_paths:
        if os.path.basename(in_path).startswith("._"):
            continue

        rater = extract_rater_from_path(os.path.dirname(os.path.dirname(in_path)))
        if not rater:
            continue

        rel_path = os.path.relpath(in_path, input_dir)
        base_name = os.path.splitext(os.path.basename(in_path))[0]
        out_file = f"{base_name}.{rater}.json"
        out_path = os.path.join(output_dir, os.path.dirname(rel_path), out_file)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            with open(in_path, "r") as f:
                data = json.load(f)

            transformed = transform_annotations(data, rater)

            with open(out_path, "w") as f:
                json.dump(transformed, f, indent=2)

            print(f"✅ Transformed and saved {out_file}")
        except Exception as e:
            print(f"❌ Failed to process {in_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_annotated_format_with_rater.py <input_dir> <output_dir>")
        sys.exit(1)

    convert_directory(sys.argv[1], sys.argv[2])