# Release verification

Verified on 2026-09-15 with Python in the existing research environment, PyTorch 2.7.1+cu118 and NumPy 2.0.2. All numerical checks and the example ran on CPU; this is not a GPU benchmark validation.

Commands run from the repository directory:

```bash
python -m unittest discover -s tests -v
python -m examples.synthetic_step
```

Results: **6 tests passed**. The example completed teacher prediction, matched-context generation, the combined loss, finite student gradients, an optimizer update and student-only decoding. Teacher parameters received no gradients. Its confidence override is explicitly synthetic-only.

The tests cover:

- Pairwise interval score bounds against exhaustive corners of 20 random four-bit boxes, plus 2,000 sampled interior winners.
- CIS gradient direction, zero interval-interior correction, masked pixels and detached teacher targets.
- Set-mass loss behavior for full sets and ignored pixels, and a small descent step.
- Equality of interval candidate counts and teacher-ranked top-k counts, with correct ranking.
- Exact value and student-gradient equality of the new adapter and the original experiment's loss composition; disabling CIS/IGSS recovers the encoding base.
- Preservation of teacher batch-normalization buffers and mixed train/eval modes during context forwards.

All six extracted source modules were also parsed and compared with the original experiment files. Their executable ASTs match exactly after removing docstrings; documentation updates are recorded by source and release hashes in `source_manifest.json`.

No training results, random seeds, original experiment code, logs or weights were modified. Fresh-environment installation, full benchmark reproduction and released-checkpoint validation remain separate work.
