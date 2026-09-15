"""在完整区间方法上附加码本候选集合监督；不替换任何原损失。

背景区间 -> 成对解码分数上界 -> 未被支配类别集合 -> 集合概率质量损失。
集合损失和区间界计算均有先例；本文件实现待验证的具体组合，不宣称首创。
来源与实验边界见 docs/attribution.md 和 docs/method.md。
最终CoIS使用topk_set：区间决定数量，原图教师排序决定类别。
"""
import math

import torch
import torch.nn.functional as F

from .ecoc import similarities


@torch.no_grad()
def interval_candidates(teacher_probabilities, codes, original_class):
    """返回[N,C,H,W]候选；保证包含区间内所有可能解码赢家及原教师类。

这个保证只相对于观测预测的矩形区间，不保证覆盖真实标签。
成对上界检验是必要条件，可能保留实际不可能同时击败所有对手的类别。
使用与原推理解码完全相同的1-平均L1相似度，避免更换决策几何。
"""
    if teacher_probabilities.ndim != 5 or teacher_probabilities.shape[2] != codes.shape[1]:
        raise ValueError('Expected contexts [S,N,K,H,W] and codebook [C,K]')
    if original_class.shape != teacher_probabilities.shape[1:2] + teacher_probabilities.shape[3:]:
        raise ValueError('Original teacher classes must have shape [N,H,W]')
    lower, upper = teacher_probabilities.amin(0), teacher_probabilities.amax(0)
    candidates = torch.ones((lower.shape[0], len(codes), *lower.shape[2:]),
                            device=lower.device, dtype=torch.bool)
    # s_c(p)-s_d(p) = [2(C_c-C_d).p + sum(C_d-C_c)]/K.
    # It is affine: its box maximum is attained at endpoints chosen by sign.
    for competitor in codes:
        difference = codes - competitor
        maximum = (2 * torch.einsum('bkhw,ck->bchw', upper, difference.clamp_min(0))
                   + 2 * torch.einsum('bkhw,ck->bchw', lower, difference.clamp_max(0))
                   - difference.sum(1)[None, :, None, None]) / codes.shape[1]
        candidates &= maximum >= -1e-6  # retain floating-point ties conservatively
    original_missing = ~candidates.gather(1, original_class[:, None]).squeeze(1)
    candidates.scatter_(1, original_class[:, None], True)
    return candidates, original_missing


@torch.no_grad()
def matched_topk(candidates, original_probabilities, codes):
    """直接对照：每像素同样的集合大小，仅按原图教师相似度取前k类。"""
    order = similarities(original_probabilities, codes).argsort(dim=1, descending=True, stable=True)
    ranks = torch.arange(len(codes), device=order.device)[None, :, None, None]
    in_rank_order = ranks < candidates.sum(1, keepdim=True)
    return torch.zeros_like(candidates).scatter_(1, order, in_rank_order)


def set_mass_loss(logits, candidates, weights, codes, temperature):
    """现有ECOC余弦/温度下的 -log sum_{c in S} p(c)，全像素归一化。

候选集合内不强制分配均匀概率；若S含全部类，损失/梯度严格为0。
S为单类时退化为普通码字分类监督，这种退化不是新贡献。
"""
    if candidates.dtype != torch.bool or not candidates.any(1).all():
        raise ValueError('Each pixel must have a nonempty boolean candidate set')
    if weights.shape != logits.shape[:1] + logits.shape[2:] or temperature <= 0:
        raise ValueError('Invalid weights or temperature')
    unit = F.normalize(logits, p=2, dim=1, eps=1e-8)
    scores = torch.einsum('bkhw,ck->bchw', unit, 2*codes-1) / math.sqrt(codes.shape[1]) / temperature
    all_mass = torch.logsumexp(scores, 1)
    selected_mass = torch.logsumexp(scores.masked_fill(~candidates, -torch.inf), 1)
    per_pixel = torch.where(candidates.all(1), torch.zeros_like(all_mass),
                            (all_mass-selected_mass).clamp_min(0))
    return (per_pixel*weights).mean()


def code_set_loss(outputs, contexts, original_probabilities, original_class, weights,
                  codes, mode, temperature=.5, aux_weight=.4):
    """复用已有两个背景前向，主教师集合同时监督主/辅头，无新网络参数。"""
    candidates, original_missing = interval_candidates(contexts, codes, original_class)
    point_candidates = matched_topk(candidates, original_probabilities, codes)
    if mode == 'interval_set':
        selected = candidates
    elif mode == 'topk_set':
        selected = point_candidates
    else:
        raise ValueError('Unknown set selection mode')
    main = set_mass_loss(outputs[1], selected, weights, codes, temperature)
    auxiliary = set_mass_loss(outputs[0], selected, weights, codes, temperature)
    with torch.no_grad():
        valid = weights > 0
        denominator = valid.sum().clamp_min(1)
        size = selected.sum(1)
        stats = dict(set_size=(size*valid).sum()/denominator,
            set_singleton=((size == 1)*valid).sum()/denominator,
            set_ambiguous=(((size > 1) & (size < len(codes)))*valid).sum()/denominator,
            set_uninformative=((size == len(codes))*valid).sum()/denominator,
            set_differs_topk=((candidates != point_candidates).any(1)*valid).sum()/denominator,
            set_original_added=(original_missing*valid).sum()/denominator,
            main_set_loss=main.detach(), aux_set_loss=auxiliary.detach())
    return main+aux_weight*auxiliary, stats
