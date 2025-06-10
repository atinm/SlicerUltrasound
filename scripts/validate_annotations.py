#!/usr/bin/env python3

import os
import sys
import json
from jsonschema import validate, ValidationError

def validate_directory(schema_path, json_root):
    with open(schema_path) as f:
        schema = json.load(f)

    failures = []

    def validate_file(full_path):
        try:
            with open(full_path) as f:
                data = json.load(f)

            validate(instance=data, schema=schema)
            print(f"✅ {full_path}")
        except (ValidationError, ValueError) as e:
            print(f"❌ {full_path} - {e}")
            if isinstance(e, ValidationError):
                print(f"  → Message: {e.message}")
                print(f"  → Path: {'/'.join(map(str, e.absolute_path))}")
                print(f"  → Offending section: {json.dumps(e.instance, indent=2)}")
            failures.append((full_path, str(e)))
        except Exception as e:
            print(f"⚠️ {full_path} - Failed to read or parse: {e}")
            failures.append((full_path, str(e)))

    if os.path.isdir(json_root):
        for root, _, files in os.walk(json_root):
            for file in files:
                if file.endswith(".json") and not file.startswith("._"):
                    full_path = os.path.join(root, file)
                    validate_file(full_path)
    elif os.path.isfile(json_root) and json_root.endswith(".json"):
        validate_file(json_root)
    else:
        print(f"⚠️ {json_root} is not a .json file or directory.")

    print(f"\nDone. {len(failures)} file(s) failed validation.")
    return failures

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python validate_annotations.py <schema_file> <directory>")
        sys.exit(1)

    schema_file = sys.argv[1]
    target_dir = sys.argv[2]
    validate_directory(schema_file, target_dir)