#!/usr/bin/env bash
# 一键跑满 28 帧：人(模板匹配) + 导线(逐帧 PCA 线性度 + 最近深度簇) 测距。
# 默认 --wire_mode per_frame（无绝对深度/高度常量）；如需静态悬链线可加
#   --wire_mode catenary [--wire_curve recompute]。
# 依赖 conda env ov3d；权重与数据已随包附上，默认 GPU 0。
set -euo pipefail
cd "$(dirname "$0")"

source /home/xjt/miniconda3/etc/profile.d/conda.sh
conda activate ov3d

GPU="${1:-0}"
exec python range_person_wire.py --gpu "$GPU" --frames all
