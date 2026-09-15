# CoIS 方法代码

对应当前论文 **CoIS: Context-Guided Interval and Set Supervision for Multi-Source Remote Sensing Segmentation**。

本文件夹从真实实验源码中提取方法核心，保留上下文构造、CIS 区间监督、IGSS 集合监督以及必要的编码损失。新增统一调用接口和可执行示例，避免导入本机其他实验目录。

## 先运行什么

安装适合机器的 PyTorch，然后在仓库目录执行：

```bash
python -m pip install -e .
python -m examples.synthetic_step
python -m unittest discover -s tests -v
```

示例在 CPU 上使用随机合成图，完成教师预测、上下文构造、完整损失、学生反向传播和一次参数更新。它用于验证接入流程，不是实际数据训练，也不产生论文成绩。

## 文件怎么对应论文

| 文件 | 作用 |
| --- | --- |
| `cois/context_views.py` | 同一目标区域换两个源背景，使用同一个教师 BN 快照 |
| `cois/losses/context_interval.py` | CIS：逐码位区间及投影 KL |
| `cois/losses/interval_code_set.py` | IGSS：区间确定数量，原图教师排名确定类别 |
| `cois/losses/context_additive.py` | 原混合监督保留，仅附加 CIS |
| `cois/losses/ecoc.py` | 有明确先行来源的编码、自训练损失和推理解码 |
| `cois/objective.py` | 把原损失、CIS、IGSS 按实验权重组合 |
| `configs/` | 去除个人路径后的四任务参数记录，不是完整训练配置 |

论文最终 IGSS 对应源码 `topk_set`，不是 `interval_set`。后者保留为开发消融。区间内为零的是附加 CIS 梯度，原目标伪监督继续生效。

## 本次发布范围

这是可独立导入和验证的**方法组件版本**，尚不具备完整数据集从头训练的一键复现入口。原工程依赖历史实验记录、其他工程的网络定义、数据处理和初始化权重；没有把这些依赖伪装为已经解决。完整基准复现还需要按 [复现说明](docs/reproduction.md) 整理。

英文 README 中成绩来自当前稿件的组件消融表。它们是单种子、目标验证集选模结果；不是本次整理重新训练的结果。未宣称论文已经录用，也未把继承组件都称为原创。具体来源见 [引用说明](docs/attribution.md)。

仓库先保持私有；暂未指定开源许可证。此目录只包含研究方法代码和配套文档，不包含申请材料、个人账号信息、数据集或大权重。
