"""One CPU optimizer step through CoIS; no dataset or checkpoint required."""
import copy
import torch
from torch import nn

from cois import cois_loss, decode, source_context_probabilities
from cois.codebook import initial_codebook
from cois.losses.ecoc import encode, pseudo_labels


class DemoNetwork(nn.Module):
    """Small interface demonstration, NOT the experimental ResNet-50 model."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1),
                                      nn.BatchNorm2d(8), nn.ReLU())
        self.auxiliary = nn.Conv2d(8, 16, 1)
        self.main = nn.Conv2d(8, 16, 1)

    def forward(self, image, return_feat=False):
        features = self.features(image)
        return self.auxiliary(features), self.main(features)


def main():
    torch.manual_seed(2333)
    torch.set_num_threads(2)
    student = DemoNetwork().train()
    teacher = copy.deepcopy(student).requires_grad_(False).train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=6e-5)
    codes = torch.tensor(initial_codebook(), dtype=torch.float32)
    sources, target = torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)
    labels = torch.randint(0, 6, (2, 16, 16))
    source_bits, valid = encode(labels, codes)
    # Stand-in mask: real experiments use the original source ClassMix sampler.
    source_mask = labels < 3
    with torch.no_grad():
        original_logits = teacher(target, return_feat=True)[1]
        hard_bits, quality, _, _ = pseudo_labels(original_logits, codes)
        original_probabilities = original_logits.sigmoid()
        contexts = source_context_probabilities(teacher, target, sources, source_mask)
        # Random untrained logits give zero original quality. This explicit
        # synthetic-only override exercises auxiliary gradients, not accuracy.
        quality = torch.ones_like(quality)
        target_weights = (~source_mask) * quality
        mixed_bits = torch.where(source_mask[:, None], source_bits, hard_bits)
        mixed_weights = torch.where(source_mask, valid.float(), quality)
        mixed = torch.where(source_mask[:, None], sources, target)
    before = student.main.weight.detach().clone()
    loss, terms = cois_loss(student(sources), student(mixed), source_bits,
        valid.float(), mixed_bits, mixed_weights, target_weights,
        contexts, original_probabilities, codes)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    optimizer.step()
    assert not torch.equal(before, student.main.weight)
    student.eval()
    with torch.no_grad():
        prediction = decode(student(target)[1], codes)
    assert prediction.shape == labels.shape
    print("PASS: synthetic teacher/context/CoIS/backward/optimizer/inference path")
    print("Demonstration network only; synthetic confidence override = 1; no benchmark result.")
    print({key: round(float(terms[key]), 6) for key in ("source", "mixed_encoding", "cis", "igss")})


if __name__ == "__main__":
    main()
