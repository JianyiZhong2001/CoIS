"""ECOCSeg equations 5--12 and Algorithm 1, reimplemented in plain PyTorch.

    NeurIPS 2025: Towards Robust Pseudo-Label Learning in Semantic Segmentation:
    An Encoding Perspective. https://github.com/Woof6/ECOCSeg

Uses the paper's positive-plus-negatives contrast denominator, including a
hybrid positive when it is not a valid codebook row. This differs from the
cached author loss, whose denominator sums only the valid codebook rows.
"""
import math
import torch
import torch.nn.functional as F


def encode(labels, codes):
    valid = (labels >= 0) & (labels < len(codes))
    bits = codes[labels.clamp(0, len(codes) - 1)].permute(0, 3, 1, 2).contiguous()
    return bits, valid


def similarities(probabilities, codes):
    # 1 - mean(abs(p-b)); avoid a [batch,classes,bits,H,W] allocation.
    signs = 2 * codes - 1
    return (torch.einsum('bkhw,ck->bchw', probabilities, signs)
            + (1 - codes).sum(1)[None, :, None, None]) / codes.shape[1]


def decode(logits, codes):
    return similarities(logits.sigmoid(), codes).argmax(1)


@torch.no_grad()
def pseudo_labels(logits, codes, mining_threshold=.95, quality_threshold=.968):
    probabilities = logits.sigmoid()
    order = similarities(probabilities, codes).argsort(dim=1, descending=True, stable=True)
    nearest, _ = encode(order[:, 0], codes)
    shared = torch.ones_like(nearest, dtype=torch.bool)
    selected = torch.zeros_like(shared)
    unresolved = torch.ones_like(order[:, 0], dtype=torch.bool)
    confidence = probabilities.maximum(1 - probabilities)
    for rank in range(len(codes)):
        candidate, _ = encode(order[:, rank], codes)
        shared &= candidate == nearest
        support = shared.sum(1)
        average = (confidence * shared).sum(1) / support.clamp_min(1)
        stop = unresolved & ((average > mining_threshold) | (support == 0))
        selected = torch.where(stop[:, None], shared, selected)
        unresolved &= ~stop
    # General fallback for a codebook with a globally shared column.
    selected = torch.where(unresolved[:, None], shared, selected)
    hybrid = torch.where(selected, nearest, (logits >= 0).to(logits.dtype))
    # Equation 47: per-image global quality; do not reject individual pixels.
    quality = (confidence.mean(1) >= quality_threshold).float().mean((1, 2), keepdim=True)
    quality = quality.expand_as(order[:, 0])
    return hybrid, quality, selected, order[:, 0]


def pixel_loss(logits, bits, weights, codes, lambda_distance=5., lambda_contrast=2., temperature=.5):
    if logits.shape != bits.shape or weights.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError('ECOC tensor shapes do not match')
    binary = F.binary_cross_entropy_with_logits(logits, bits, reduction='none').mean(1)
    unit = F.normalize(logits, p=2, dim=1, eps=1e-8)
    target_signs = 2 * bits - 1
    positive_cos = (unit * target_signs).sum(1) / math.sqrt(codes.shape[1])
    code_signs = 2 * codes - 1
    negative_cos = torch.einsum('bkhw,ck->bchw', unit, code_signs) / math.sqrt(codes.shape[1])
    # Equality excludes the positive only if the hybrid is a valid class row.
    matches = torch.einsum('bkhw,ck->bchw', target_signs, code_signs) == codes.shape[1]
    negative_scores = (negative_cos / temperature).masked_fill(matches, -torch.inf)
    positive_score = positive_cos / temperature
    denominator = torch.logsumexp(torch.cat((positive_score[:, None], negative_scores), 1), 1)
    contrast = denominator - positive_score
    distance = 1 - positive_cos
    # Mean over pixels, not sum of confidence: global quality remains effective.
    values = [(term * weights).mean() for term in (binary, distance, contrast)]
    total = values[0] + lambda_distance * values[1] + lambda_contrast * values[2]
    return total, torch.stack([x.detach() for x in values])


def segmentation_loss(outputs, bits, weights, codes, aux_weight=.4, **kwargs):
    auxiliary, main = outputs[:2]
    main_loss, components = pixel_loss(main, bits, weights, codes, **kwargs)
    aux_loss, aux_components = pixel_loss(auxiliary, bits, weights, codes, **kwargs)
    return main_loss + aux_weight * aux_loss, components + aux_weight * aux_components
