# Method and tensor contracts

The network has auxiliary and main heads with logits `[N,K,H,W]`, where K=16. The binary codebook is `[C,K]`. Pixel weights are `[N,H,W]`. Background teacher responses are `[2,N,K,H,W]`.

## CIS

Let `lower = contexts.amin(0)` and `upper = contexts.amax(0)`. The detached target is `q = clip(sigmoid(z), lower, upper)`. CIS minimizes `KL(Bern(q) || Bern(sigmoid(z)))`, weighted by retained-target mask times original-teacher image confidence, averaged over the full image and bits. Its direct gradient points toward the nearest interval endpoint outside the interval and is zero inside. The original encoding losses remain active inside as well.

## IGSS

Inference similarity is `s_c(p) = 1 - mean(abs(p - C_c))`. For every class pair, the maximum score difference over the interval is obtained by selecting upper endpoints for positive codeword differences and lower endpoints for negative differences. Classes passing all pairwise bounds form a conservative candidate set; the original-view teacher winner is added if absent.

The **size** of that set determines k. IGSS then takes the **top-k original-view teacher classes**, using L1 similarity and stable ranking. It minimizes the negative log total mass of this selected set under normalized student-logit/codeword cosine scores at temperature 0.5. All-class sets have exactly zero additional loss; singleton sets reduce to class supervision. No true-label coverage or conformal guarantee is claimed.

## Combined objective

`L = 2 * L_source_encoding + 2 * (L_mixed_encoding + L_CIS + L_IGSS)`.

Each term combines main and auxiliary heads with weights 1 and 0.4. Encoding binary/distance/contrast coefficients are 1, 5 and 2. The set multiplier is 1. Teacher main-head targets supervise both student heads. Student inference uses the original L1 codebook decoder.

The teacher performs its ordinary target forward before the context forwards. Context forwards temporarily use `eval()` and restore all prior module modes. Source geometry and target mask positions must remain matched; the caller supplies its normal strong augmentation only to the student mixed image. Target weights must exclude source pixels and invalid padding. No target labels are required by the objective.

## Retained historical controls

`context_interval.target_loss(..., mode="mean_matched")` is a mean-target control with matched absolute direct logit-gradient magnitude. This does not match parameter gradients or directions.

`code_set_loss(..., mode="interval_set")` uses interval candidates directly instead of teacher-ranked membership. `context_interval.mixed_loss` is an older replacement-objective experiment and is **not** used by the released CoIS adapter; `context_additive.mixed_loss` is the correct retained-base path.
