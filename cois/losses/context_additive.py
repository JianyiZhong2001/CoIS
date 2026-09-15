"""保留完整混合图ECOC监督，只附加目标像素的背景一致性。

原损失来源为ECOCSeg；区间KL及统一衰减来自本地已登记context_interval。
区间内只有附加项梯度为零，原有目标监督仍然存在。
"""
from .ecoc import segmentation_loss
from .context_interval import target_loss


def mixed_loss(outputs, mixed_bits, mixed_weights, target_weights,
               teacher_probabilities, codes, mode, aux_weight=.4, **ecoc_kwargs):
    base, components = segmentation_loss(outputs, mixed_bits, mixed_weights,
                                         codes, aux_weight=aux_weight, **ecoc_kwargs)
    extra = outputs[1].sum()*0
    heads = {}
    for name, logits, scale in (('main', outputs[1], 1.), ('aux', outputs[0], aux_weight)):
        loss, diagnostics = target_loss(logits, teacher_probabilities, target_weights, mode)
        extra = extra + scale*loss
        heads[name] = dict(target_loss=loss.detach(), **diagnostics)
    return base+extra, dict(base_loss=base.detach(), base_components=components,
                            context_loss=extra.detach(), **heads)
