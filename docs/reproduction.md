# Experimental protocol and reproduction status

## What this release verifies

The six extracted core source files have SHA256 provenance in `source_manifest.json`. The integration API is new packaging code. Numerical checks test interval gradients, set bounds, target masking, retained original gradients and teacher detachment. The synthetic example tests the actual source/context/loss/backward interface on a small demonstration network.

No benchmark training, real-image inference or checkpoint conversion was run during packaging. The included configuration JSON files record hyperparameters only; they are not executable trainer configurations.

## Reported protocol

| Task | Sources | Target | Classes | Validation crops |
| --- | --- | --- | ---: | ---: |
| T1 | Potsdam RGB, Potsdam IRRG | Vaihingen IRRG | 6 | 440 |
| T2 | Potsdam RGB, Vaihingen IRRG | Potsdam IRRG | 6 | 2,016 |
| T3 | Zurich, Chicago | Paris | 3 | 2,485 |
| T4 | Zurich, Chicago | Berlin | 3 | 600 |

The experimental model is ResNet-50/DeepLabV2 with 16-bit main and auxiliary heads. Inputs are 512×512. Each source slot contributes one source and one target image. T1/T2 raw labels 1–6 map to 0–5; raw 0 and 255 are ignored. CITY-OSM classes are background, road and building, with padding ignored. T1 sources are paired spectral views rather than independent source cities.

Seed: 2333. Shared source initialization: 2,000 updates. Adaptation: 8,000 updates per arm. AdamW: encoder learning rate 0.00006, head rate 0.0006, weight decay 0.01, 1,500-update learning-rate warmup and linear adaptation decay. EMA cap: 0.99. Bit-mining/image-quality thresholds: 0.95/0.968. Eight validation checkpoints occur at logical steps 3,000 through 10,000; maximum mIoU selects the model, with the earlier checkpoint breaking ties. OA and class scores come from that same checkpoint.

T1/T2 target evaluation data were repeatedly inspected during development. CITY-OSM target original images were split approximately 70/10/20% for adaptation/validation/test before cropping. The manuscript component table reports validation selection, with held-out test evaluation pending. The current evidence is single-seed and should not be presented as statistically significant improvement or an untouched test benchmark.

## Assets needed for end-to-end reproduction

- Portable ResNet-50/DeepLabV2 training and evaluation entry points matching the original model and preprocessing, with relevant upstream notices.
- Download instructions and exact image-level split/crop manifests for licensed ISPRS and CITY-OSM datasets; no dataset redistribution is included here.
- Portable initialization generation and source-confusion assignment, including the source-statistic protocol and checksums.
- Complete training state, evaluation/selection code, actual command lines and verified dependency versions for each benchmark.
- If checkpoint distribution is chosen: permitted weights with checksums and matching codebook/class order.

The original experiment launchers were not copied as pretend portable commands: they enforce hashes of historical run records and refer to machine-specific paths. The current release deliberately exposes the reusable method without claiming those dependencies are already packaged.

## Result provenance

The README values are transcribed from the current manuscript's `tables/joint_ablation.tex` (source hash recorded in `source_manifest.json`). They are rounded values from that table. They must not be combined with fixed-endpoint comparison scores or used to imply equal budgets across unrelated historical baselines.
