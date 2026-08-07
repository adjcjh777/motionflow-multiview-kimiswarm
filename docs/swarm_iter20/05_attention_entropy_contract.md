# Attention-entropy 训练契约审计

## 结论

`bayesian_tri_v2_attention_entropy` 的既有训练记录不能证明 attention-entropy 机制有效。旧 trainer 没有把该模型返回的 combined auxiliary loss 加入总损失，因此 entropy 与 epipolar 两个辅助项在该训练入口里都实际未生效。

本轮只修源码契约并做 CPU 静态验证，不运行 GPU 训练或重评。

## 三处确定性错误

1. 模型输出的 weights 形状是 `(B,T,V,J)`。旧代码 reshape 后又 permute 成 `(B*T,J,V)`，但 `_entropy_loss` 仍沿 `dim=1` 归一化，实际跨 joints 而不是 views 计算。
2. Shannon entropy `H=-sum(p log p)` 为正。旧函数返回 `-H`，最小化时会最大化 entropy，与“集中到少量视角”的设计目标相反。
3. 主 trainer 只给 `bayesian_tri`、`bayesian_tri_v2` 和 visibility 变体加入末尾辅助项，遗漏了 attention-entropy 变体；同时构造该变体时也没有传入 CLI 的 `epipolar_loss_weight`。

## 修正后的单一契约

- weights 保持 `(B*T,V,J)`，沿 view 轴归一化。
- `entropy_temperature=1` 时得到普通归一化权重；更低温度使分布更尖锐。
- 模型返回 `scaled_epipolar_loss + attention_entropy_weight * entropy`。
- trainer 直接加一次模型已缩放的 combined auxiliary loss。
- CLI `--epipolar_loss_weight` 直接传入模型，不在 trainer 再乘一次。

## 历史结果边界

旧 `bayesian_tri_v2_attention_entropy` checkpoint 可保留为旧 recipe 的历史数值，但不能支持以下说法：

- entropy regularisation 改善了精度或鲁棒性；
- triangulation weights 因 entropy 项变得更集中；
- 该变体中的 epipolar auxiliary loss 对训练有贡献。

由于子类不改变预测路径、只替换末尾 auxiliary scalar，旧 trainer 对该 scalar 的遗漏意味着这两个辅助机制都未进入反向传播。其他训练参数和数据 recipe 仍可能与 base checkpoint 不同，因此不把二者宣称为完全相同实验。

## 当前判定

`STOP bayesian_tri_v2_attention_entropy historical mechanism claim`。该范围不包含仓库中独立的 hierarchical entropy trainer。源码修正只恢复未来实验的定义，不生成任何新性能证据；本轮不启动 GPU 实验。
