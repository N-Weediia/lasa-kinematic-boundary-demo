# LaSA 运动学辅助边界候选校正

这是一个围绕 LaSA 公开后处理接口整理的独立 NumPy demo。它保留 LaSA 学到的边界峰作为唯一候选，只在三维重建 QC 可信时，尝试把已有候选吸附到邻近的运动学事件峰。质心速度变化和角速度是辅助线索；它们不会单独生成新边界，也不替代模型的动作语义预测。

> 这份代码用于说明一个可测试的研究想法，不是 LaSA 复现结果或性能改进证据。所有演示输入均为合成信号，包括有利构造与错误线索反例。

## 安装与运行

需要 Python 3.9+ 和 NumPy。无需 PyTorch、LaSA 权重或训练数据。

```bash
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -v
```

`demo.py` 会打印并写出 `demo_output.json`。该小例子展示候选峰的边界吸附、低 QC 回退，以及当 QC 被高估且运动线索错误时校正也可能变差。它不是实验结果，不能据此推断泛化性、精度或真实数据收益。

## 设计要点

- **候选边界来自模型。** 选峰继续采用 LaSA `libs/metric.py::argrelmax` 的严格局部极大值规则：首帧始终保留，内部峰须严格高于两侧，并达到当前 `PostProcessor.boundary_th`（默认 0.7）。
- **运动学只校正已有候选。** 等间隔帧的 CoM 速度差分幅值与角速度幅值经稳健归一化后形成 eventness。输入线索必须与模型输出使用相同帧时钟，`snap_radius` 以帧计。只有邻域内存在高 eventness 的严格局部峰时，才考虑移动候选；不会因为运动学信号强而添加模型没有提出的边界。
- **低质量先回退。** 缺少任一运动信号、缺少 QC、QC 低于 0.5，或序列不足三个有效帧时，直接返回 LaSA 风格的 baseline 峰，不计算可能失败的运动学分支。QC 是调用方提供的外部可信度，不由本 demo 推断。
- **重建质量决定是否可相信运动线索。** 若 QC 被高估，或运动特征与真实转段不一致，吸附可能把边界移错；两个模型候选也可能吸附到同一事件峰并合并。这些情况在测试中明确覆盖。
- **其余 LaSA 流程保持原样。** 可选补丁只连接后处理边界选择；模型 logits、语言模块、训练目标和多数投票逻辑不变。未使用或未修改 LaSA 权重。

## 与 LaSA 源码的关系

集成补丁针对上游 commit [`f823dc8`](https://github.com/HaoyuJi/LaSA/commit/f823dc8354e90251a0bd2f2b89fbd14eb4a9d047)，其干净 `libs/postprocess.py` 已通过 GitHub API 按 commit 核对。补丁在 `libs/postprocess.py` 添加可选 `(N,T)` 运动信号和 QC 入参，同时沿用 `(N,1,T)` mask 并对每条边界序列展平后选峰；不传 cue/QC 时，继续调用 LaSA 原有 `argrelmax`。另外把 `libs/metric.py` 中已弃用的 `np.bool` 改成等价的 `np.bool_`，以兼容 NumPy 1.24+，不改变比较规则。

完整说明和补丁在 [`integration/README.md`](integration/README.md) 与 [`integration/integration.patch`](integration/integration.patch)。上游为 MIT 许可，来源和许可说明见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 研究问题与验证边界

这个原型来自一个可检验的观察：起步、减速和转向等阶段交界，可能同时表现为质心速度变化或角速度变化。一个后续实验可以在受试者级留出划分中对比原始 LaSA 后处理、QC 门控后的候选校正和 oracle 边界，并报告总体及各转段的偏移、误检和漏检。阈值应只用开发集选择，最终测试集只评估一次；还需要单独检查不同重建质量下的错误。

本仓库尚未验证上述方案对 LaSA 测试集或真实队列有任何改善，也没有证明运动学事件在不同对象、场景和动作上的可迁移性。特别是，本 demo 的 fallback 只验证软件路径能回到 baseline；它不能证明外部 QC 标签准确，也不等同于三维重建质量本身。

## 目录

- `kinematic_boundary.py`：NumPy 核心、边界匹配指标。
- `demo.py`：仅含合成输入的可运行机制示例。
- `tests/`：baseline 一致性、质量回退、候选限制、误吸附风险和匹配指标测试。
- `integration/`：对指定上游版本的可审阅补丁与接入说明。
- `requirements.txt`、`LICENSE`、`THIRD_PARTY_NOTICES.md`：依赖和许可信息。

## 简短英文说明

This NumPy-only prototype keeps LaSA's learned boundary peaks as the sole candidates. When externally supplied reconstruction QC is adequate, nearby kinematic eventness may shift an existing candidate; missing or low-quality cues fall back to LaSA-style local maxima. It introduces no kinematic-only boundaries and makes no benchmark-improvement claim. The included favorable synthetic example illustrates code behavior only.

