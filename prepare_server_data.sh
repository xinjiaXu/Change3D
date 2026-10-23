#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
data_root="${1:-/home/xxj/Data}"
adapter_root="$repo_root/data_adapters"

link_dir() {
  local source="$1"
  local target="$2"
  if [[ ! -d "$source" ]]; then
    printf 'Missing source directory: %s\n' "$source" >&2
    exit 1
  fi
  if [[ -e "$target" && ! -L "$target" ]]; then
    printf 'Refusing to replace existing non-symlink: %s\n' "$target" >&2
    exit 1
  fi
  ln -sfn "$source" "$target"
}

prepare_dataset() {
  local name="$1"
  local source_root="$2"
  local image_a="$3"
  local image_b="$4"
  local label_dir="$5"
  for split in train val test; do
    local split_source="$source_root/$split"
    local split_target="$adapter_root/$name/$split"
    mkdir -p "$split_target"
    link_dir "$split_source/$image_a" "$split_target/t1"
    link_dir "$split_source/$image_b" "$split_target/t2"
    link_dir "$split_source/$label_dir" "$split_target/label"
  done
}

prepare_dataset LEVIR-CD "$data_root/LEVIR_CD" A B label
prepare_dataset LEVIR-CD-PLUS "$data_root/LEVIR_CD_PLUS/cut_data" A B label
prepare_dataset CDD "$data_root/CDD" A B OUT
prepare_dataset WHU-CD "$data_root/WHUCD" A B label
prepare_dataset SYSU-CD "$data_root/SYSU_CD" time1 time2 label
printf 'Change3D data adapters prepared under %s\n' "$adapter_root"
