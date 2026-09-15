"""在相同位置保留目标像素，更换源背景；原型辅助函数，不是训练入口。"""
import torch


def mix_context(target, source_image, source_mask):
    """source_mask=True处取源图，其他位置逐像素保留目标图。

    同一目标的所有源背景必须使用同一mask。这里不读取目标标签；背景
    同时改变外观与场景内容，不能解释为纯风格变化或因果识别。
    """
    if target.ndim != 4 or source_image.shape != target.shape[1:]:
        raise ValueError('Expected target [N,C,H,W] and one source [C,H,W]')
    if source_mask.dtype != torch.bool or source_mask.shape != target.shape[:1] + target.shape[2:]:
        raise ValueError('Expected boolean source mask [N,H,W]')
    return torch.where(source_mask[:, None], source_image[None], target)


@torch.no_grad()
def source_context_probabilities(teacher, target, sources, source_mask):
    """返回[S,N,K,H,W]的EMA码位概率；每个源背景分别前向，节省显存。

    调用前，训练器应已用原目标图执行原有的一次teacher.train前向，使BN
    只按原目标流更新。本函数把全部背景预测放在同一个BN快照下，暂时eval，
    最后恢复各层模式。原图前向得到的全图质量权重由调用者保留。
    """
    if sources.ndim != 4 or len(sources) < 2:
        raise ValueError('At least two source contexts are required')
    modes = [(layer, layer.training) for layer in teacher.modules()]
    probabilities = []
    try:
        teacher.eval()
        for source_image in sources:
            view = mix_context(target, source_image, source_mask)
            main = teacher(view, return_feat=True)[1]
            probabilities.append(main.sigmoid())
    finally:
        for layer, training in modes:
            layer.training = training
    return torch.stack(probabilities).detach()
