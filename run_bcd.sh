#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gpu_id="${GPU_ID:-0}"
batch_size="${BATCH_SIZE:-16}"
num_workers="${NUM_WORKERS:-4}"
max_steps="${MAX_STEPS:-80000}"
lr="${LR:-2e-4}"
pretrained="${PRETRAINED:-$repo_root/model/X3D_L.pyth}"
data_root="${DATA_ROOT:-/home/xxj/Data}"

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd}\n' "$0" >&2
  exit 2
fi

case "$1" in
  levircd) dataset="LEVIR-CD"; dataset_root="$data_root/LEVIR_CD"; image_a="A"; image_b="B"; label_dir="label" ;;
  levircdplus) dataset="LEVIR-CD-PLUS"; dataset_root="$data_root/LEVIR_CD_PLUS/cut_data"; image_a="A"; image_b="B"; label_dir="label" ;;
  cdd) dataset="CDD"; dataset_root="$data_root/CDD"; image_a="A"; image_b="B"; label_dir="OUT" ;;
  whucd) dataset="WHU-CD"; dataset_root="$data_root/WHUCD"; image_a="A"; image_b="B"; label_dir="label" ;;
  sysucd) dataset="SYSU-CD"; dataset_root="$data_root/SYSU_CD"; image_a="time1"; image_b="time2"; label_dir="label" ;;
  *) printf 'Unknown dataset key: %s\n' "$1" >&2; exit 2 ;;
esac

save_dir="$repo_root/Change3D_work_dirs/$1"
for split in train val test; do
  for folder in "$image_a" "$image_b" "$label_dir"; do
    if [[ ! -d "$dataset_root/$split/$folder" ]]; then
      printf 'Dataset directory missing: %s\nRun prepare_server_data.sh first or set DATA_ROOT.\n' "$dataset_root/$split/$folder" >&2
      exit 1
    fi
  done
done

if [[ ! -f "$pretrained" ]]; then
  printf 'Pretrained X3D-L checkpoint not found: %s\n' "$pretrained" >&2
  printf 'Download it from the official URL in README.md and place it at that path.\n' >&2
  exit 1
fi

cd "$repo_root"
python ./scripts/train_BCD.py \
  --dataset "$dataset" \
  --file_root "$dataset_root" \
  --image_dir_a "$image_a" \
  --image_dir_b "$image_b" \
  --label_dir "$label_dir" \
  --pretrained "$pretrained" \
  --save_dir "$save_dir" \
  --gpu_id "$gpu_id" \
  --batch_size "$batch_size" \
  --num_workers "$num_workers" \
  --max_steps "$max_steps" \
  --lr "$lr"
