#!/usr/bin/env bash
# 一键跑满数据目录内全部帧：人(模板匹配) + 导线(逐帧 PCA 线性度 + 最近深度簇) 测距。
# 默认 --wire_mode per_frame（无绝对深度/高度常量）；如需静态悬链线可加
#   --wire_mode catenary [--wire_curve recompute]。
#
# 环境：直接用当前已激活的 Python 环境（需装有 torch/ultralytics/sam2）。
# 若希望脚本自动进入某个 conda 环境，设置 WPR_CONDA_ENV，例如：
#   WPR_CONDA_ENV=ov3d ./run_ranging.sh 0
# 更多参数见 README「快速开始」。
set -euo pipefail
cd "$(dirname "$0")"

GPU="${1:-0}"
shift || true

PY="python"
if [[ -n "${WPR_CONDA_ENV:-}" ]] && command -v conda >/dev/null 2>&1; then
  PY="conda run -n ${WPR_CONDA_ENV} --no-capture-output python"
fi

# 前置检查：三个必需依赖缺失时给出明确提示，而不是深埋 ImportError。
if ! $PY -c "import torch, ultralytics, sam2" >/dev/null 2>&1; then
  echo "[run_ranging.sh] 当前 Python 环境缺少 torch / ultralytics / sam2。" >&2
  echo "                请先激活项目环境（或设置 WPR_CONDA_ENV=<env> 指向已装好的 conda 环境）。" >&2
  echo "                从零搭建环境见 README「环境搭建」。依赖安装：pip install -r requirements.txt" >&2
  exit 1
fi

exec $PY range_person_wire.py --gpu "$GPU" --frames all "$@"
