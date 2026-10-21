# benchmarks — 冻结的验证基线

`baseline/` 存放重构前（git tag `pre-refactor`）在验证数据（`data/` 28 帧）上产出的**文本基线**，
供第三方部署者与 CI 在没有原作者环境参与的情况下自证管线正确性。

| 文件 | 来源 | 内容 |
|---|---|---|
| `perframe_summary.csv` | shipped `viz_range_perframe/summary.csv` | 逐帧 11 列汇总（27 ok + 1 错误帧） |
| `perframe_results.json` | shipped `viz_range_perframe/results.json` | 逐帧完整结果 |
| `gt_range.json` | shipped `viz_range_perframe/img_1700603346.892936/range.json` | 参考帧（GT）完整结果 |
| `catenary_summary.csv` | shipped `viz_range/summary.csv` | 悬链线模式逐帧汇总（旧 schema） |
| `catenary_gt_range.json` | shipped `viz_range/img_1700603346.892936/range.json` | 悬链线模式参考帧结果 |

关键数值（per_frame 模式，27 ok 帧）：

- 导线→人最近距离 mean **0.840 m**、std 0.035 m（手工真值 mean 0.848 m）
- 参考帧 `img_1700603346.892936`：算法 **0.905 m**、真值 0.896 m、偏差 +0.009 m
- 悬链线模式参考帧：算法 **0.908 m**
- 唯一错误帧：`img_1700603348.322151`（`wire_too_few_inliers`，该帧线状目标仅 4 个雷达点）

用法：

```bash
# 容差化自检（第三方跨环境兜底，vs 本目录基线）
python scripts/regression_check.py <你的 out_dir>

# 同机严格门禁（md5 / 像素级全等，仅同一环境前后两次运行可比）
python tests/compare_runs.py /tmp/run_before /tmp/run_after --strict
```

注意：严格 md5 门禁**只能**用于同一机器同一环境的前后两次运行对比；
跨机器/跨驱动版本允许微小数值漂移，此时用 `scripts/regression_check.py` 的容差口径。
