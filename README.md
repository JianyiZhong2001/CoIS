# CoIS

**Context-Guided Interval and Set Supervision for Multi-Source Remote Sensing Segmentation**

PyTorch method components for the CoIS research manuscript. CoIS uses source-context teacher predictions to regulate auxiliary supervision of unlabeled target pixels, while retaining an encoding-based segmentation objective.

**Release status:** this is a method-component release, extracted from the local experimental implementation. It includes the original loss functions, source-context construction, codebook utilities, an integration API, numerical checks and a synthetic backward-pass example. It does **not** yet include a portable end-to-end benchmark trainer, dataset preprocessing pipelines or trained checkpoints. The example verifies execution; it does not reproduce the manuscript results. No conference acceptance is claimed.

[中文说明](README_zh.md) · [Method](docs/method.md) · [Experimental protocol](docs/reproduction.md) · [Attribution](docs/attribution.md)

## Method

1. **Source-context views:** retain the same target pixels and positions under two source backgrounds. The EMA teacher evaluates both views using fixed batch-normalization statistics with dropout disabled.
2. **Contextual Interval Supervision (CIS):** construct the minimum/maximum teacher probability for each output bit. An additional Bernoulli KL loss corrects student probabilities outside this range.
3. **Interval-Guided Set Supervision (IGSS):** derive a candidate count from interval/codebook bounds, select that many classes using the original-view teacher ranking, and supervise their total probability mass.

The full source and mixed-image encoding losses remain active. Both auxiliary objectives use the original teacher's image-level confidence, only on retained target pixels. No inference branch or trainable parameter is added; CIS requires two additional teacher context forwards, reused by IGSS.

**Implementation naming:** `topk_set` is the final manuscript's IGSS. `interval_set` selects class identities directly from interval bounds and is retained as a development ablation. Do not interchange them.

## Install and run

Use Python 3.10+ and install a [PyTorch build](https://pytorch.org/get-started/locally/) appropriate for your machine. Then, from this repository:

```bash
python -m pip install -e .
python -m examples.synthetic_step
python -m unittest discover -s tests -v
```

The CPU example needs no dataset, checkpoint, network request or GPU. See [validation.md](docs/validation.md) for the environment and checks actually run on this release.

## Integrate into a trainer

```python
from cois import cois_loss, decode, source_context_probabilities

# Model contract: teacher(image, return_feat=True) returns (aux, main, ...).
# First run the existing original-target teacher pass, deriving its quality.
# `source_mask` is True for pasted source pixels.
contexts = source_context_probabilities(teacher, target, sources, source_mask)

loss, terms = cois_loss(
    source_outputs, mixed_outputs,
    source_bits, source_weights, mixed_bits, mixed_weights,
    target_weights, contexts, original_teacher_probabilities, codes,
)
loss.backward()

# Inference uses only the student main-head logits and the fixed codebook.
prediction = decode(student_main_logits, codes)
```

`examples/synthetic_step.py` provides a complete, executable version of this data flow. A production trainer must additionally provide its architecture, source initialization, real data and splits, EMA schedule, ClassMix/strong augmentation, validation and complete checkpoint recovery. Tensor contracts and the original experiment settings are documented under `docs/`.

## Layout

```text
cois/
  context_views.py          Identical target content under source backgrounds
  codebook.py               Fixed codewords and source-based assignment utility
  objective.py              Baseline + CIS + IGSS integration API
  losses/
    ecoc.py                Attributed local ECOCSeg reimplementation
    context_interval.py    Interval KL and matched-mean control
    context_additive.py    Retain original losses and add CIS
    interval_code_set.py    Interval bounds, teacher top-k and set-mass loss
configs/                   Portable records of the experiment hyperparameters
docs/                      Method, protocol, attribution and verification
examples/                  Synthetic optimizer-step example
tests/                     Mathematical and integration regressions
```

## Results and interpretation

The following mIoU (%) values are the selected-validation component results in the current manuscript, not measurements produced by this release's synthetic example.

| Method | T1: ISPRS | T2: ISPRS | T3: Paris | T4: Berlin |
| --- | ---: | ---: | ---: | ---: |
| Enhanced encoding base | 59.34 | 71.49 | 55.49 | 57.10 |
| Base + CIS | 59.99 | 71.39 | 55.42 | 57.23 |
| CoIS: Base + CIS + IGSS | 60.37 | 71.41 | 55.55 | 57.42 |

All component variants use one seed (2333), 2,000 source initialization updates and 8,000 adaptation updates. Models are selected by target-validation mIoU among eight checkpoints. These labels informed development and model selection, although they were excluded from adaptation losses. These are not untouched test results or evidence of statistical significance. CoIS does not improve every task. T1's two sources are paired spectral views. See [protocol and outstanding reproduction assets](docs/reproduction.md).

## Attribution and licensing

The encoding/self-training backbone, codeword assignment and set-valued supervision have prior work. See [attribution.md](docs/attribution.md) for sources and a disclosed difference between the local ECOC contrastive denominator and the upstream implementation.

No repository-wide open-source license has been selected for this private preparation copy. Public redistribution and any later license must respect the applicable rights. Dataset files, pretrained weights, application materials and private experiment logs are not included.
