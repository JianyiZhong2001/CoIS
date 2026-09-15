"""多源背景下的逐码位区间监督，与同输入的平均软标签对照。

集合上的乐观KL损失继承credal learning；这里是Bernoulli区间的闭式特例。
参见Lienen et al., Conformal Credal Self-Supervised Learning (2023), Eq.11。
区间没有目标真值覆盖保证，不是conformal预测集，也不保证合法类别码字。
"""
import torch
import torch.nn.functional as F

from .ecoc import pixel_loss


def projected_target(logits, teacher_probabilities, mode):
    """教师和投影不反传；interval内零梯度，外部向最近端点移动。"""
    if teacher_probabilities.ndim != 5 or teacher_probabilities.shape[1:] != logits.shape:
        raise ValueError('Expected teacher probabilities [S,N,K,H,W] matching logits')
    if mode not in ('mean', 'interval'):
        raise ValueError('mode must be mean or interval')
    with torch.no_grad():
        if (not torch.isfinite(teacher_probabilities).all()
                or (teacher_probabilities < 0).any() or (teacher_probabilities > 1).any()):
            raise ValueError('Teacher probabilities must be finite and within [0,1]')
        lower = teacher_probabilities.amin(0)
        upper = teacher_probabilities.amax(0)
        if mode == 'mean':
            target = teacher_probabilities.mean(0)
        else:
            target = logits.detach().sigmoid().maximum(lower).minimum(upper)
    return target, lower, upper


def target_loss(logits, teacher_probabilities, weights, mode):
    """以全图像素/码位数归一化，与原ECOC掩码权重口径一致。

    weights必须仅覆盖ClassMix保留的目标像素。用KL而非单独BCE记录区间
    损失，区间内部应是零；投影目标detach后梯度为sigmoid(logits)-target。
    """
    if weights.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError('Expected pixel weights [N,H,W]')
    if mode not in ('mean', 'mean_matched', 'interval'):
        raise ValueError('mode must be mean, mean_matched or interval')
    target, lower, upper = projected_target(logits, teacher_probabilities,
                                            'mean' if mode == 'mean_matched' else mode)
    scale = logits.new_ones(())
    if mode == 'mean_matched':
        # Simple competitor: weaken every target bit uniformly. Match the L1
        # magnitude of this head's interval logit gradient before backprop.
        # This does not match the full network-parameter gradient or its direction.
        with torch.no_grad():
            p = logits.sigmoid()
            projected = p.maximum(lower).minimum(upper)
            numerator = ((p-projected).abs()*weights[:, None]).sum()
            denominator = ((p-target).abs()*weights[:, None]).sum()
            scale = numerator/denominator if denominator > 0 else logits.new_zeros(())
    entropy = -(torch.xlogy(target, target) + torch.xlogy(1-target, 1-target))
    kl = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') - entropy).clamp_min(0)
    loss = scale*(kl.mean(1) * weights).mean()
    with torch.no_grad():
        valid = (weights > 0)[:, None].expand_as(logits)
        p = logits.sigmoid()
        active = ((p < lower) | (p > upper)) & valid
        denominator = valid.sum().clamp_min(1)
        diagnostics = dict(interval_width=((upper-lower)*valid).sum()/denominator,
                           outside_interval=active.sum()/denominator,
                           valid_target_bits=valid.sum(), global_scale=scale.detach())
    return loss, diagnostics


def mixed_loss(outputs, source_bits, source_weights, target_weights,
               teacher_probabilities, codes, mode, aux_weight=.4, **ecoc_kwargs):
    """混合源像素保留完整ECOC，混合目标像素使用mean/interval KL。

    target上不同时保留原硬伪标签的距离/对比损失，否则这些项仍会强迫
    不确定码位取某个值。mean与interval两组必须采用完全相同的损失结构。
    独立源图分支仍由原segmentation_loss处理；由train_context.py接入。
    """
    if ((source_weights > 0) & (target_weights > 0)).any():
        raise ValueError('Source and target masks must not overlap')
    total = outputs[1].sum()*0
    terms = {}
    for name, logits, scale in (('main', outputs[1], 1.), ('aux', outputs[0], aux_weight)):
        source, components = pixel_loss(logits, source_bits, source_weights, codes, **ecoc_kwargs)
        target, diagnostics = target_loss(logits, teacher_probabilities, target_weights, mode)
        total = total + scale*(source+target)
        terms[name] = dict(source_loss=source.detach(), target_loss=target.detach(),
                           source_components=components, **diagnostics)
    return total, terms
