#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
gpu_id="${GPU_ID:-0}"
batch_size="${BATCH_SIZE:-16}"
num_workers="${NUM_WORKERS:-4}"
max_steps="${MAX_STEPS:-80000}"
lr="${LR:-2e-4}"
pretrained="${PRETRAINED:-$repo_root/model/X3D_L.pyth}"

if [[ $# -ne 1 ]]; then
  printf 'Usage: %s {levircd|levircdplus|cdd|whucd|sysucd}\n' "$0" >&2
  exit 2
fi

case "$1" in
  levircd) dataset="LEVIR-CD" ;;
  levircdplus) dataset="LEVIR-CD-PLUS" ;;
  cdd) dataset="CDD" ;;
  whucd) dataset="WHU-CD" ;;
  sysucd) dataset="SYSU-CD" ;;
  *) printf 'Unknown dataset key: %s\n' "$1" >&2; exit 2 ;;
esac

data_root="$repo_root/data_adapters/$dataset"
save_dir="$repo_root/Change3D_work_dirs/$1"
for split in train val test; do
  for folder in t1 t2 label; do
    if [[ ! -d "$data_root/$split/$folder" ]]; then
      printf 'Data adapter missing: %s\nRun prepare_server_data.sh first.\n' "$data_root/$split/$folder" >&2
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
  --file_root "$data_root" \
  --pretrained "$pretrained" \
  --save_dir "$save_dir" \
  --gpu_id "$gpu_id" \
  --batch_size "$batch_size" \
  --num_workers "$num_workers" \
  --max_steps "$max_steps" \
  --lr "$lr"
