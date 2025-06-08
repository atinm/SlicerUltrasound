#!/bin/bash

set -e

# Usage: ./combine_all_files_to_tree.sh /path/to/source_dir /path/to/output_dir
src_dir="$1"
out_dir="$2"

if [[ ! -d "$src_dir" || -z "$out_dir" ]]; then
  echo "Usage: $0 /path/to/source_dir /path/to/output_dir"
  exit 1
fi

mkdir -p "$out_dir"

find "$src_dir" -type f \( -name "*.dcm" -o -name "*.json" \) | while read -r file; do
  mkdir -p "$(dirname "$out_dir")"
  cp "$file" "$out_dir"
  echo "Copied $file → $out_dir"
done
