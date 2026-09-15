# CoIS: Context-Guided Interval and Set Supervision for Multi-Source Remote Sensing Segmentation

PyTorch implementation of the CoIS method components. The full benchmark training pipeline and pretrained models are not included in this release.

## Abstract

Teacher predictions for a target pixel can change across source contexts, making precise pseudo-label supervision unreliable. CoIS uses these responses to regulate auxiliary supervision at the code-bit and class-set levels. Contextual Interval Supervision (CIS) corrects student responses outside teacher-derived probability ranges. Interval-Guided Set Supervision (IGSS) uses interval/codebook bounds to determine a candidate count and original-view teacher rankings to select the corresponding classes. Both objectives retain the encoding-based learning framework without adding trainable parameters or an inference branch.

## Highlights

- **CIS:** supervise the same target content using probability intervals obtained under two source backgrounds.
- **IGSS:** supervise the total probability mass of teacher-ranked candidate classes, with candidate counts derived from the intervals.
- **Unchanged inference:** use the student network and fixed codebook; additional context predictions are needed only during training.

## Environment

Python 3.10+, PyTorch 2.0+ and NumPy 1.24+. Verified with PyTorch 2.7.1 and NumPy 2.0.2.

```bash
git clone https://github.com/JianyiZhong2001/CoIS.git
cd CoIS
python -m pip install -e .
```

## Code and Usage

| File | Description |
| --- | --- |
| `cois/context_views.py` | Source-context teacher predictions |
| `cois/losses/context_interval.py` | CIS interval projection loss |
| `cois/losses/interval_code_set.py` | IGSS candidate construction and set-mass loss |
| `cois/losses/context_additive.py` | Original mixed-image loss with added CIS |
| `cois/losses/ecoc.py` | Encoding losses, pseudo-labels and decoding |
| `cois/codebook.py` | Codebook and source-based assignment |
| `cois/objective.py` | Combined source, mixed-image, CIS and IGSS objective |

In an existing encoding-based trainer, use `cois_loss` to combine the objectives. Model outputs are `(auxiliary_logits, main_logits, ...)` with shape `[N,K,H,W]`; context probabilities have shape `[2,N,K,H,W]`, pixel weights `[N,H,W]`, and the codebook `[C,K]`.

```python
from cois import cois_loss, decode, source_context_probabilities

# Run the ordinary target-teacher pass first to obtain probabilities and quality.
contexts = source_context_probabilities(teacher, target, sources, source_mask)
loss, terms = cois_loss(
    source_outputs, mixed_outputs, source_bits, source_weights,
    mixed_bits, mixed_weights, target_weights, contexts,
    original_teacher_probabilities, codes,
)
loss.backward()
prediction = decode(student_main_logits, codes)
```

`source_mask=True` selects source pixels. Target weights are the retained-target mask times original-teacher image confidence, excluding invalid pixels and padding. The teacher must support `teacher(image, return_feat=True)[1]` for main-head logits. The final IGSS uses `topk_set`; `interval_set` is a development control.

## Results

Selected-validation mIoU (%) from the manuscript's component comparison:

| Method | T1 | T2 | T3 | T4 |
| --- | ---: | ---: | ---: | ---: |
| Enhanced encoding base | 59.34 | 71.49 | 55.49 | 57.10 |
| Base + CIS | 59.99 | 71.39 | 55.42 | 57.23 |
| **CoIS** | **60.37** | **71.41** | **55.55** | **57.42** |

T1: Potsdam RGB + Potsdam IRRG → Vaihingen IRRG. T2: Potsdam RGB + Vaihingen IRRG → Potsdam IRRG. T3/T4: Zurich + Chicago → Paris/Berlin (CITY-OSM).

These single-seed results use 2,000 source updates and 8,000 adaptation updates. Models are selected by target-validation mIoU among eight checkpoints; target labels informed selection and development, but not adaptation losses. These are validation results, not held-out test results.

## Acknowledgments

The encoding and self-training components build on [ECOCSeg](https://github.com/Woof6/ECOCSeg) and [DACS](https://arxiv.org/abs/2007.08702). Codeword assignment, contextual consistency and optimistic set-valued supervision have precedents in [Evron et al.](https://proceedings.mlr.press/v206/evron23a.html), [PiPa](https://doi.org/10.1145/3581783.3611708) and [Conformal Credal Self-Supervised Learning](https://arxiv.org/abs/2205.15239).

The local ECOC loss uses a positive-plus-negatives contrastive denominator, including a hybrid positive when needed; the upstream implementation sums valid codebook rows. This difference is retained in the matched component comparisons.

## Contact

Questions and feedback are welcome through [GitHub Issues](https://github.com/JianyiZhong2001/CoIS/issues).
