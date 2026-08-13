# H36M 3D 标签循环性诊断

## 目的

验证旧的 H36M `.npz` 数据中的 `joints_3d` 是否由输入 2D 关键点经 DLT 三角化生成（循环标签），而新的真 GT v2 数据则不应存在这种循环关系。

方法：对每一帧，用 `points_2d` + 相机参数做**无权重 DLT 三角化**，得到 `j3d_dlt`，再与存储的 `joints_3d` 计算 MPJPE。

---

## 脚本

- 路径：`scripts/diagnose_circular_h36m_labels.py`

用法：

```bash
python scripts/diagnose_circular_h36m_labels.py <npz_path> [-o <json_path>]
```

---

## 运行命令

```bash
# 旧循环数据（h36m_hf）
python scripts/diagnose_circular_h36m_labels.py data/h36m_hf/s_01_act_02_multiview.npz -o tmp/diagnosis_old.json

# 旧循环数据（webbridge/h36m）
python scripts/diagnose_circular_h36m_labels.py data/webbridge/h36m/s_01_acts_02_multiview.npz -o tmp/diagnosis_webbridge.json

# 新真 GT v2 数据（S01 短序列）
python scripts/diagnose_circular_h36m_labels.py data/h36m_true_gt_v2/s_01_act_02_multiview.npz -o tmp/diagnosis_new.json

# 新真 GT v2 数据（S09 长序列，更大样本）
python scripts/diagnose_circular_h36m_labels.py data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz -o tmp/diagnosis_new_s09.json
```

---

## 结果摘要

| 数据 | 文件 | 平均 MPJPE | 中位数 MPJPE | 最大 MPJPE |
|------|------|-----------|-------------|-----------|
| 旧数据（h36m_hf） | `data/h36m_hf/s_01_act_02_multiview.npz` | **0.000000 mm** | 0.000000 mm | 0.000000 mm |
| 旧数据（webbridge/h36m） | `data/webbridge/h36m/s_01_acts_02_multiview.npz` | **0.000000 mm** | 0.000000 mm | 0.000000 mm |
| 新真 GT v2（S01） | `data/h36m_true_gt_v2/s_01_act_02_multiview.npz` | **0.014529 mm** | 0.014407 mm | 0.035126 mm |
| 新真 GT v2（S09） | `data/h36m_true_gt_v2/s_09_acts_02_03_04_05_06_07_08_09_10_11_12_13_14_15_16_multiview_m.npz` | **0.033827 mm** | 0.029085 mm | 4.265902 mm |

### 解释

- **旧数据**：无权重 DLT 三角化结果与 `joints_3d` 完全一致，MPJPE 精确为 0。这证明旧 H36M 标签就是由输入 2D 经 DLT 三角化生成的，属于循环标签，**不能用于模型选择**。
- **新真 GT v2 数据**：DLT 三角化结果与 `joints_3d` 的平均偏差约为 **0.015 mm**（S01 短序列）和 **0.034 mm**（S09 长序列），最大约 4.27 mm（单个异常帧）。该残余相对于 H36M 尺度极小，反映 v2 标签与自身投影到各视角的 2D 点是几何自洽的（这是真值动捕数据的正常特征），但并非像旧数据那样“由 2D 三角化精确复现”。

### 结论

- 旧 `data/h36m_hf/` 与 `data/webbridge/h36m/` 中的 H36M 标签是**循环标签**，必须废弃。
- `data/h36m_true_gt_v2/` 中的标签不是简单由输入 2D DLT 三角化而来，可作为真值基准使用。
