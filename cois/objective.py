"""Combine the retained ECOC objective with CIS and teacher-ranked IGSS.

This packaging adapter calls the experiment loss functions without changing
their formulas. The caller constructs batches, updates its EMA teacher and
supplies the original-view teacher probabilities and confidence weights.
"""
from .losses.ecoc import segmentation_loss, similarities
from .losses.context_additive import mixed_loss
from .losses.interval_code_set import code_set_loss


def cois_loss(source_outputs, mixed_outputs, source_bits, source_weights,
              mixed_bits, mixed_weights, target_weights, context_probabilities,
              original_teacher_probabilities, codes, *, cis=True, igss=True,
              set_mode="topk_set", source_scale=2.0, mixed_scale=2.0,
              set_weight=1.0, aux_weight=0.4, lambda_distance=5.0,
              lambda_contrast=2.0, temperature=0.5):
    """Return total loss and detached diagnostics for two-head model outputs.

    Outputs are (auxiliary_logits, main_logits, ...), each [N,K,H,W].
    Context probabilities are [2,N,K,H,W]; pixel weights are [N,H,W].
    target_weights must be zero on source pixels, ignored pixels and padding.
    Teacher inputs and weights are detached here, as in the original training
    loop. `topk_set` is the manuscript method; `interval_set` is an ablation.
    """
    if igss and not cis:
        raise ValueError("The released IGSS configuration retains CIS")
    if set_mode not in ("topk_set", "interval_set"):
        raise ValueError("Unknown set selection mode")
    contexts = context_probabilities.detach()
    original = original_teacher_probabilities.detach()
    target_weights = target_weights.detach()
    kwargs = dict(aux_weight=aux_weight, lambda_distance=lambda_distance,
                  lambda_contrast=lambda_contrast, temperature=temperature)
    source, _ = segmentation_loss(source_outputs, source_bits.detach(),
                                   source_weights.detach(), codes, **kwargs)
    if cis:
        mixed, terms = mixed_loss(mixed_outputs, mixed_bits.detach(),
            mixed_weights.detach(), target_weights, contexts, codes, "interval", **kwargs)
        context = terms["context_loss"]
        base_mixed = terms["base_loss"]
    else:
        mixed, _ = segmentation_loss(mixed_outputs, mixed_bits.detach(),
                                     mixed_weights.detach(), codes, **kwargs)
        base_mixed = mixed.detach()
        context = mixed.detach().new_zeros(())
    extra = mixed.new_zeros(())
    stats = {}
    if igss:
        winner = similarities(original, codes).argmax(1)
        extra, stats = code_set_loss(mixed_outputs, contexts, original, winner,
            target_weights, codes, set_mode, temperature=temperature, aux_weight=aux_weight)
    total = source_scale * source + mixed_scale * (mixed + set_weight * extra)
    diagnostics = dict(source=source_scale * source.detach(),
        mixed_encoding=mixed_scale * base_mixed,
        cis=mixed_scale * context,
        igss=mixed_scale * set_weight * extra.detach(), **stats)
    return total, diagnostics
