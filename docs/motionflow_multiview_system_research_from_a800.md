# MotionFlow 多视角工作流与机器人 Profile 系统研究设计

> **来源：** 本文件是从 A800-D 主机 `/mnt/nvme0n1/zhangzy/projects/motionflow-research-multiview-easymocap-robot-profiles/research/multiview-easymocap-robot-profiles.md` 以只读方式复制到本地仓库，用于对齐论文方向。
>
> 状态：研究定位与接口骨架，不是产品实现。
>
> 分支：`research/multiview-easymocap-robot-profiles`
>
> 基线：GitLab `master` 的 `98c13e8d633ac753e25d12e7aec4730b18732392`
>
> 本研究不安装或集成 EasyMocap，不修改 GVHMR、GMR、MJLab 三个上游
> vendor，不构建镜像，也不改变 A800 生产容器、任务数据库或训练任务。

## 1. 论文定位

论文的主要贡献定位为 **MotionFlow 可复现工作流与系统**，不是声称首创
“无标定多视角人体运动恢复（HMR）”算法。

系统研究的问题是：

1. 如何在不破坏现有单目 GVHMR 稳定链路的前提下，把多路视频、同步、
   标定、人体恢复、跨视角融合和质量门控组织成可替换、可追溯的任务流。
2. 如何定义统一的 `HumanMotionIR`，让单目结果、不同多视角融合模块和
   EasyMocap 等候选后端能够进入同一机器人重定向与策略训练链路。
3. 如何把融合置信度、时间同步误差、遮挡和尺度不确定性传播到机器人
   重定向、参考动作筛选和策略训练，而不是只输出一段看起来平滑的动作。
4. 如何通过版本化 robot profile 支持不同机器人自由度、关节顺序、模型
   资产、重定向配置、MJLab 任务和 ONNX 导出约定。
5. 如何用端到端策略层指标验证系统价值，而不只比较 HMR 的视觉误差。

论文主线应写成：

```text
multi-view inputs
    → per-view frozen GVHMR
    → latent spatiotemporal alignment / fusion
    → HumanMotionIR
    → robot profiles
    → GMR / MJLab
    → policy preview / export
```

其中：

- 单视角时，现有 GVHMR 输出可直接进入 `HumanMotionIR`。
- 多视角时，每一路先独立经过 **冻结的 GVHMR**，避免把系统贡献与上游
  模型微调混在一起。
- 跨视角融合是可插拔模块，不绑定某一个算法；MUC、DMMR、ScoreHMR 或
  EasyMocap 路线只能作为候选模块/基线，实际接口和许可需分别核验。
- `HumanMotionIR` 是 MotionFlow 内部稳定契约，再按需导出
  `hmr4d_results.pt` 兼容 artifact 给现有 GMR。
- robot profile 和策略层评测是系统论文区别于纯 HMR 论文的重要部分。

## 2. 贡献边界

### 2.1 计划主张的系统贡献

候选贡献应以实验为前提，表述为：

1. **统一接口。** 用 `HumanMotionIR` 隔离输入、单目恢复、跨视角融合和
   机器人消费端，使不同模块可以替换并保持 artifact 可追溯。
2. **任务编排。** 对多视频上传、同步、标定、逐视角恢复、融合、质量门控、
   重定向、训练、预览和导出建立可恢复状态机。
3. **质量门控。** 在进入 GMR/MJLab 前显式判断同步、标定、人物关联、
   运动连续性和融合可靠性，支持失败、降级到最佳单视角或人工确认。
4. **不确定性传播。** 将帧级/关节级置信度、视角支持数、同步误差和尺度
   不确定性传递给动作裁剪、重定向权重和训练数据选择。
5. **多机器人 profile。** 把机器人模型、DoF、关节映射、限制、观测/动作
   顺序和导出元数据从业务代码中抽离并版本化。
6. **端到端可复现。** 固化输入 manifest、模块版本、配置、随机种子、
   artifact hash、profile hash 和策略评测结果。
7. **策略层验证。** 同时报告 HMR、重定向和策略跟踪指标，验证上游质量
   变化是否真正改善机器人任务。

### 2.2 不可宣称

除非后续实验和文献证据充分，否则论文不得声称：

- 首创无标定、弱标定或手持手机多视角 HMR。
- 发明了 MUC、DMMR、ScoreHMR、EasyMocap 或其核心融合算法。
- 候选方法天然可以互换；每种方法都需要独立 adapter 和复现实验。
- 不经标定的手持视频能稳定恢复绝对尺度或准确全局相机轨迹。
- 多视角一定优于最佳单视角，或 HMR 指标改善必然提升策略效果。
- 已支持 Unitree G1 29DoF；在权威模型和 action order 未核验前只能称为
  profile 接口预留。
- 只在仿真验证后就具备真实机器人泛化、稳定性或安全性。
- 已解决人体到机器人的不确定性传播；研究阶段只能称为提出接口、门控和
  候选传播策略，并通过消融验证。
- EasyMocap 代码、模型或权重可以随 MotionFlow Docker 镜像再分发。

## 3. 系统架构

```text
Capture Session
  ├─ video_00 + metadata
  ├─ video_01 + metadata
  ├─ ...
  ├─ synchronization
  └─ calibration tier
          │
          ▼
Input Normalization + Quality Precheck
          │
          ├──────────── single view ────────────┐
          │                                      │
          ▼                                      ▼
Per-view Frozen GVHMR Workers              GVHMR passthrough
          │                                      │
          ▼                                      │
Latent Spatiotemporal Alignment / Fusion Plugin  │
          │                                      │
          └──────────────────┬───────────────────┘
                             ▼
                       HumanMotionIR
                   + uncertainty / provenance
                             │
                             ▼
             hmr4d_results.pt compatibility export
                             │
                             ▼
                   Robot Profile Resolver
                             │
                             ▼
                  GMR → PKL → NPZ → smoothing
                             │
                             ▼
                 MJLab training / policy preview
                             │
                             ▼
                 deployment NPZ + ONNX export
```

### 3.1 稳定路径

当前单目路径保持为零行为变化基线：

```text
single video → frozen GVHMR → HumanMotionIR passthrough
             → existing GMR/MJLab → preview/export
```

研究代码不得修改 GVHMR 权重、推理参数或现有任务的数值结果。IR passthrough
必须通过 golden artifact 回归证明不改变下游。

### 3.2 多视角路径

多视角默认先做逐视角独立恢复：

```text
N videos → N frozen GVHMR artifacts
         → temporal alignment
         → coordinate/latent alignment
         → fusion plugin
         → HumanMotionIR
```

这样可以：

- 复用现有、已验证的单目恢复能力。
- 把失败定位到具体视角、同步、对齐或融合模块。
- 用最佳单视角作为天然降级路径。
- 在不重训 GVHMR 的情况下公平比较融合模块。

## 4. 多视角输入与三档标定

### 4.1 档位 A：宽松同步、弱/无标定

适用于手持手机、未知内外参和不稳定 FPS：

- 用容器时间戳、音频互相关、拍手或闪光事件做粗同步。
- 用人体运动相关性做细同步。
- 允许只做时间对齐和 latent-level fusion。
- 不能承诺绝对尺度和稳定全局相机轨迹。
- 质量不足时降级为最佳单视角，而不是输出虚假的高置信度融合结果。

### 4.2 档位 B：弱标定/自标定

适用于存在重叠视野和静态背景特征的多路视频：

- 使用设备内参先验、SfM/SLAM 或背景特征估计相机关系。
- 用人体关键点、地面和骨长一致性细化。
- 报告内参来源、外参漂移、退化区间和重投影误差。
- 对滚动快门、自动焦距和动态背景单独标记风险。

### 4.3 档位 C：严格标定

用于系统基准和主要论文实验：

- 固定相机、固定 FPS/焦距/曝光。
- 使用棋盘格或 AprilTag/ChArUco 标定。
- 使用硬件同步或统一音频/闪光事件。
- 记录相机序列号、标定版本、场地布局和标定 hash。
- 动作拍摄前后检查相机是否移动。

严格标定数据用于区分“融合模块能力”和“手持输入噪声”。

### 4.4 时间同步模型

每路相机时间映射到 session 全局时间：

```text
t_global = a_i * t_camera_i + b_i
```

- `b_i` 是起始偏移。
- `a_i` 表示实际时钟/帧率漂移。

同步结果必须包含：

- 原始帧索引与全局时间的映射。
- 有效重叠区间。
- 每路偏移、漂移和置信度。
- 对齐残差和被丢弃帧。
- 不能可靠同步时的失败/降级原因。

## 5. 可插拔融合模块

### 5.1 模块定位

跨视角融合是 MotionFlow 中的一个可研究组件，而不是整篇论文唯一贡献。

候选实现可能来自：

- 简单的参数/轨迹后融合。
- latent feature 的时间与视角注意力。
- MUC、DMMR、ScoreHMR 等公开方法的合规复现或 adapter。
- EasyMocap 作为几何多视角 baseline 或独立输出后端。

这些名称代表待核验的候选路线，不表示它们有相同输入、输出或许可。

### 5.2 插件契约

```python
class MultiViewFusionPlugin(Protocol):
    plugin_id: str
    plugin_version: str

    def fuse(
        self,
        views: list[PerViewHumanMotion],
        synchronization: SynchronizationResult,
        calibration: CalibrationResult | None,
        config: FusionConfig,
    ) -> HumanMotionIR:
        ...
```

每个插件必须声明：

- 是否需要相机内参/外参。
- 是否支持移动相机、无标定或动态人数。
- 输入是图像、2D 关键点、SMPL 参数还是 latent feature。
- 支持的时间长度、分辨率和 GPU 预算。
- 输出不确定性和失败状态。
- 权重、代码、数据和许可证版本。

### 5.3 最低可用融合基线

为避免只与复杂算法比较，至少实现概念性 baseline：

1. `best_single_view`：按质量分选择单路，不做融合。
2. `framewise_best_view`：逐帧选择可见性最高视角并做时间平滑。
3. `late_parameter_average`：对齐后对兼容参数做置信度加权后融合。
4. `strict_calibration_geometry`：严格标定条件下的几何一致性基线。
5. 一个公开多视角方法的复现/adapter，具体方法经 Phase 0 文献与许可审查确定。

## 6. HumanMotionIR

### 6.1 目标

`HumanMotionIR` 是上游人体恢复与下游机器人消费之间的稳定接口。它不等同
于某个模型的原生输出，也不要求不同方法内部结构完全一致。

### 6.2 提议结构

```python
HumanMotionIR = {
    "schema_version": str,
    "sequence_id": str,
    "person_id": str,
    "fps": float,
    "timestamps": FloatTensor[T],
    "frame_indices_by_view": LongTensor[V, T],
    "human_model": "smpl" | "smplx",
    "pose": {
        "body_pose": FloatTensor[T, ...],
        "global_orient": FloatTensor[T, 3],
        "transl": FloatTensor[T, 3],
        "betas": FloatTensor[...] | FloatTensor[T, ...],
    },
    "coordinate_system": {
        "handedness": str,
        "up_axis": str,
        "forward_axis": str,
        "length_unit": "meter",
        "world_from_reference": FloatTensor[4, 4],
    },
    "uncertainty": {
        "frame_confidence": FloatTensor[T],
        "joint_confidence": FloatTensor[T, J],
        "view_support_count": LongTensor[T, J],
        "temporal_alignment_error": FloatTensor[V, T],
        "scale_uncertainty": FloatTensor[T] | None,
        "fusion_disagreement": FloatTensor[T, J] | None,
    },
    "quality": {
        "frame_valid": BoolTensor[T],
        "failure_reasons": list[dict],
        "summary_metrics": dict,
    },
    "provenance": {
        "source_manifest_hash": str,
        "per_view_artifact_hashes": list[str],
        "gvhmr_version": str,
        "fusion_plugin": str | None,
        "fusion_plugin_version": str | None,
        "calibration_hash": str | None,
        "ir_builder_version": str,
    },
}
```

### 6.3 `hmr4d_results.pt` 兼容出口

实现前必须选取现有 GVHMR golden artifacts，记录实际键、shape、dtype、
坐标系和下游 GMR 读取字段。

IR exporter 负责：

- 将 `HumanMotionIR` 转成现有 GMR 可读取的 `hmr4d_results.pt` 兼容结构。
- 固定米制、坐标轴、旋转表示和 SMPL 关节顺序。
- 对 axis-angle、rotation matrix、quaternion 转换做数值测试。
- 为不确定性和 provenance 生成旁路 metadata，避免破坏旧读取器。
- 保证单目 passthrough 与现有 artifact 数值零回归或明确容差。

EasyMocap 若作为独立 baseline，应先适配到 `HumanMotionIR`，不能让其私有
结构直接进入 GMR。

## 7. 不确定性到机器人链路的传播

系统不应在融合完成后丢弃质量信息。候选传播策略：

1. **动作门控：** 低质量片段禁止进入训练，或要求人工裁剪确认。
2. **最佳视角降级：** 融合分歧过高时回退到最佳单视角。
3. **重定向权重：** 低置信度人体关节降低对应机器人目标权重。
4. **时间平滑强度：** 根据同步误差和融合分歧调整平滑/插值。
5. **关键接触保护：** 足端、地面和末端执行器不确定性超过阈值时拒绝训练。
6. **策略数据权重：** 研究是否按帧质量加权 imitation reward；这需要单独
   消融，不能默认视为有效。

传播接口应记录“原始不确定性 → 门控/权重 → 最终动作”的可追溯映射。

## 8. Robot Profile 抽象

### 8.1 Profile 内容

```yaml
schema_version: 1
profile_id: unitree_g1_23dof
profile_version: 0.1.0
display_name: Unitree G1 23DoF

assets:
  robot_model: ...
  mjcf_or_urdf: ...
  asset_hash: ...

kinematics:
  dof_count: 23
  joint_names: [...]
  qpos_indices: [...]
  neutral_pose: [...]
  joint_limits: [...]
  body_names: [...]
  end_effectors: [...]
  contact_bodies: [...]

retarget:
  config_id: ...
  human_to_robot_body_map: {...}
  uncertainty_mapping: {...}
  scale_policy: ...
  ground_alignment: ...

training:
  mjlab_task_id: ...
  observation_schema: ...
  action_schema: ...
  reward_profile_id: ...
  termination_profile_id: ...

export:
  onnx_action_order: [...]
  deployment_metadata: {...}
```

### 8.2 首批 Profile

- `bxi_elf3_current`：当前 MotionFlow 的零行为变化基准。
- `unitree_g1_23dof`：在权威模型、关节顺序、限制和控制接口核验后实现。
- `unitree_g1_29dof`：只预留 schema；没有权威定义前不填写猜测的关节表。

### 8.3 兼容性门禁

- `dof_count`、joint names、qpos、action order 长度完全一致。
- MJCF/URDF 中的关节名和 profile 显式对应。
- neutral pose、joint limits、单位和方向有效。
- GMR、NPZ、MJLab action、ONNX action 顺序使用 fixture 逐元素验证。
- profile hash 写入 PKL/NPZ、训练配置、checkpoint 和 ONNX metadata。
- profile/资产 hash 不匹配时禁止静默训练或导出。

## 9. 任务编排与可复现性

候选状态机：

```text
uploaded
  → normalizing_inputs
  → synchronizing
  → calibrating / calibration_skipped
  → per_view_hmr
  → aligning
  → fusing / best_view_fallback
  → quality_review
  → human_motion_ir_ready
  → retargeting
  → training
  → policy_preview
  → deploy_trim
  → export
```

每阶段必须：

- 独立 artifact 目录和输入 hash。
- 可恢复、可重试、可取消。
- 固定模块版本、配置、随机种子和运行设备。
- 记录失败原因、降级路径和人工决策。
- 不覆盖原始视频、逐视角结果或旧版本融合结果。
- 支持比较同一 capture session 的不同 fusion/profile 分支。

候选 manifest：

```text
session/
  manifest.json
  videos/camera_*.*
  synchronization.json
  calibration/
  per_view_gvhmr/
  fusion/<plugin_id>/<run_id>/
  human_motion_ir/<run_id>/
  retarget/<profile_id>/<run_id>/
  training/<profile_id>/<run_id>/
  reports/
```

## 10. 系统论文 Baseline

### 10.1 人体恢复与融合 Baseline

- **H0：当前单目 MotionFlow。**
  单视频 → GVHMR → GMR/MJLab。
- **H1：最佳单视角。**
  多视频分别跑冻结 GVHMR，仅选择质量最高的一路。
- **H2：逐帧最佳视角。**
  按可见性选择视角并做简单时间平滑。
- **H3：简单后融合。**
  同步后做置信度加权参数融合。
- **H4：EasyMocap baseline。**
  在许可和复现条件满足时作为几何多视角对照。
- **H5：公开融合方法。**
  从 MUC/DMMR/ScoreHMR 等候选中选取实际兼容且可复现的方法。
- **H6：严格标定上界。**
  固定相机、严格同步和标定条件下的系统表现。

### 10.2 系统能力 Baseline

- **S0：现有硬编码单机器人流水线。**
- **S1：有 HumanMotionIR、无质量门控。**
- **S2：有质量门控、无不确定性传播。**
- **S3：完整 IR + 门控 + 不确定性传播。**
- **S4：robot profile 驱动与现有硬编码 ELF3 的零回归对比。**

### 10.3 策略层 Baseline

- 同一动作、同一 profile、同一训练种子下：
  - 当前单目参考动作。
  - 最佳单视角参考动作。
  - 简单融合参考动作。
  - 候选多视角模块参考动作。
- 报告训练时间、sample throughput、episode length、tracking reward、
  termination、足滑、穿透和策略预览成功率。

## 11. 消融实验

至少包含：

1. 视角数量：1/2/3/4 路。
2. 输入档位：严格标定、自标定、弱/无标定。
3. 不做细同步 vs 只偏移校正 vs 偏移+漂移校正。
4. 不做质量门控 vs 门控 vs 门控+最佳视角降级。
5. 不传播不确定性 vs 重定向权重传播 vs 训练 reward 权重传播。
6. early/latent fusion vs late parameter fusion（仅对兼容方法）。
7. 冻结 GVHMR vs 候选方法原生前端；若不公平则明确不比较。
8. 无 HumanMotionIR 的专用 adapter vs 统一 IR。
9. 硬编码 ELF3 配置 vs `bxi_elf3_current` profile。
10. 同一 HumanMotionIR 在 ELF3 和 G1 23DoF profile 上的可移植性。
11. 只报告 HMR 指标 vs 加入重定向和策略层指标。
12. 缓存/恢复关闭 vs 开启时的端到端时间和重复计算量。

消融必须固定输入、随机种子、训练迭代和 robot profile，避免把调度或训练
预算差异误认为融合收益。

## 12. 验收指标

阈值为研究期暂定值，需在严格标定数据上校准。

### 12.1 同步与标定

- 严格标定同步误差 p95 ≤ 1 帧。
- 手持/自标定漂移校正后同步误差 p95 ≤ 2 帧。
- 严格标定重投影误差 median ≤ 2 px，p95 ≤ 5 px。
- 自标定重投影误差 median ≤ 4 px，p95 ≤ 10 px。
- 未达标 session 必须失败、降级或人工确认。

### 12.2 HMR 与融合

- 有真值时：MPJPE、PA-MPJPE、加速度误差。
- 无真值时：重投影误差、骨长变异、脚滑、地面穿透、时间连续性。
- 融合分歧、视角支持数、person ID 切换和无效帧比例。
- 相同输入/配置生成一致 IR schema 和 artifact hash。
- 多视角必须与“最佳单视角”比较，而不仅与随机单视角比较。

### 12.3 重定向与策略

- joint-limit violation、足滑、穿透、接触一致性和末端误差。
- episode length、tracking reward 分项、termination 类型和成功率。
- 训练吞吐、达到给定 reward 的时间和最终策略预览完整率。
- 不确定性门控是否降低失败率，以及是否损害动作覆盖率。

### 12.4 系统

- 阶段恢复成功率、缓存命中率、重复计算减少量。
- 每分钟视频的 wall time、GPU time、峰值显存、CPU 内存和磁盘增量。
- artifact/provenance/profile 可追溯完整率 100%。
- 单目 GVHMR 回归和 `bxi_elf3_current` profile 零回归。

## 13. 分阶段研究

### Phase 0：论文与契约审计

- 阅读并核验 MUC、DMMR、ScoreHMR、EasyMocap 等候选方法的原论文、代码、
  输入输出、模型和许可。
- 固化当前 `hmr4d_results.pt` golden schema。
- 明确 GMR 实际读取字段、坐标约定和容差。
- 获取权威 G1 23DoF 模型和 action order。
- 定义 HumanMotionIR v0 和 fusion plugin protocol。

停止条件：许可不可接受，或候选方法无法在统一输入上公平复现。

### Phase 1：单目 IR 零回归

- 实现概念性 GVHMR → HumanMotionIR → 兼容 exporter fixture。
- 证明现有单目 artifact 和下游行为不变。
- 建立 provenance、hash 和 schema validator。

未通过零回归前不得进入多视角集成。

### Phase 2：严格标定多视角基准

- 使用 2–4 台固定相机和严格同步。
- 运行逐视角冻结 GVHMR。
- 比较最佳单视角、简单后融合、EasyMocap/公开方法候选。
- 同时报告 HMR、重定向和短策略训练指标。

这是多视角路线的主要 Go/No-Go。

### Phase 3：质量门控与不确定性传播

- 建立同步、遮挡、人物关联和融合分歧评分。
- 验证失败、人工确认和最佳视角降级。
- 消融重定向权重和训练数据权重。

### Phase 4：手持手机输入

- 先双手机，再扩展到三/四手机。
- 分别测试静态/动态背景、慢速/快速相机运动和长视频漂移。
- 不以手持结果取代严格标定基准。

### Phase 5：Robot Profile

- `bxi_elf3_current` 零回归。
- G1 23DoF 离线重定向和仿真预览。
- 通过门禁后再训练和导出。
- 29DoF 仅在权威模型确认后启动。

### Phase 6：系统端到端实验

- 固化任务编排、缓存、恢复和 artifact lineage。
- 运行完整 baseline/ablation。
- 形成可复现实验包，而不是先进入生产 WebUI。

## 14. 风险与许可

技术风险：

- 手持相机滚动快门、自动焦距和同时运动导致不可辨识。
- 动态背景、遮挡和相似服装导致跨视角 person association 错误。
- 不同方法的人体模型、关节定义、尺度和坐标系不一致。
- 复杂融合改善视觉指标但损害接触或策略跟踪。
- robot profile 与 reward、termination、observation 强耦合。

工程风险：

- 候选方法与当前 CUDA/PyTorch/PyTorch3D 依赖冲突。
- 多路 GVHMR 显著增加 GPU、存储和调度压力。
- 多分支 artifact 增加任务恢复、清理和下载复杂度。
- 不同 fusion 模块难以获得公平的输入、预算和运行环境。

许可与隐私：

- 分别审计 EasyMocap、MUC/DMMR/ScoreHMR 候选实现及依赖许可证。
- 核验检测器、姿态模型、权重、SMPL/SMPL-X 和数据集条款。
- 核验 Unitree 模型、mesh、SDK 和部署接口的再分发边界。
- 多视角视频包含更多旁观者和场地信息，需要访问控制与保留策略。
- 许可不兼容时使用外部可选插件，不进入默认 Docker release。

## 15. 建议研究排序

1. HumanMotionIR/golden schema 与不可宣称内容。
2. 文献、代码、许可和复现成本审计。
3. 单目 GVHMR passthrough 零回归。
4. 严格标定多视角 baseline。
5. 质量门控与不确定性传播。
6. `bxi_elf3_current` profile 零回归。
7. G1 23DoF 离线 profile。
8. 手持手机弱/自标定。
9. 完整策略层实验与系统论文消融。

建议 Go/No-Go：

- 严格标定多视角若不能稳定优于最佳单视角，暂停复杂融合产品化。
- HMR 改善若不能传递到重定向或策略层，论文需如实报告边界。
- `bxi_elf3_current` profile 无法零回归时，暂停新增机器人 profile。
- 关键依赖不可合法复现/再分发时，不进入默认 MotionFlow 环境。

## 16. 待确认问题

- 论文目标会议/期刊及系统论文、机器人论文的评价偏好是什么？
- MUC、DMMR、ScoreHMR 中哪些有可用代码、权重和兼容许可证？
- 多视角实验是否有固定相机、标定板和可用真值？
- 手持手机能否保留原音轨、时间戳并锁定曝光/焦距？
- 目标是单人还是多人；主角选择如何进入任务状态机？
- G1 的准确型号、23DoF 模型、action order 和控制频率是什么？
- 真实机器人实验是否可用；若不可用，论文必须限定为仿真策略评测。

在这些问题和 Phase 0 排序确认前，本分支只保留研究设计，不进入代码实现。
