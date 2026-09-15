# Attribution and rights

This repository distinguishes the proposed context-guided supervision from its inherited components. Source hashes and local origin identifiers are recorded in `source_manifest.json`.

| Component | Prior work / source |
| --- | --- |
| Encoding and hybrid pseudo-label learning | [ECOCSeg, NeurIPS 2025](https://github.com/Woof6/ECOCSeg), *Towards Robust Pseudo-Label Learning in Semantic Segmentation: An Encoding Perspective* |
| Teacher/self-training and cross-domain mixing framework | [DACS, WACV 2021](https://arxiv.org/abs/2007.08702) |
| Codeword-to-class assignment precedent | [Evron et al., AISTATS 2023](https://proceedings.mlr.press/v206/evron23a.html) |
| Contextual consistency precedent | [PiPa, ACM MM 2023](https://doi.org/10.1145/3581783.3611708) |
| Optimistic loss over sets of labels | [Conformal Credal Self-Supervised Learning, COPA 2023](https://arxiv.org/abs/2205.15239) |

`losses/ecoc.py` is the project's plain-PyTorch equation-based reimplementation. Its contrastive denominator includes a positive plus negatives, including a hybrid positive not necessarily equal to a valid codeword. This differs from the cached author implementation, whose denominator sums valid codebook rows. It is a local adaptation, not an assertion of exact upstream reproduction.

Fixed codewords and source-confusion assignment are inherited experimental components, not newly established standalone contributions. The Bernoulli projection and set-mass loss also have prior formulations. CoIS concerns their specific use with source-context responses and the interval-count/teacher-ranking construction.

This private preparation copy does not select a repository-wide open-source license or grant rights to datasets, third-party implementations or model weights. Before a public licensed release, reconcile relevant notices and rights with the code actually distributed. No copied third-party backbone, dataset or pretrained checkpoint is bundled in this component release.
