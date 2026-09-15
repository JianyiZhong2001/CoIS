"""Inherited fixed codewords and source-confusion assignment; see docs/attribution.md."""
import itertools
import numpy as np


INITIAL_CODES = [
    '0011010101011111', '1100011100100110', '1110000011101001',
    '0100101100001000', '0101010000110001', '0011101000101101',
]


def initial_codebook():
    return np.array([[int(bit) for bit in row] for row in INITIAL_CODES], dtype=np.int8)


def assignments(counts):
    """Enumerate all 720 assignments. Rows must have measured source support.

    Counts have shape (source/augmentation profiles, true class, predicted class).
    The objective is the balanced-class whole-codeword bit-error proxy, NOT a
    target generalization bound or the exact hybrid-pseudo-label training risk.
    """
    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 3 or counts.shape[1:] != (6, 6) or (counts < 0).any():
        raise ValueError('Expected nonnegative measured counts [profiles,6,6]')
    support = counts.sum(-1)
    if (support == 0).any():
        raise ValueError('Missing source class support; do not invent confusion counts')
    q = counts / support[..., None]
    codes = initial_codebook()
    distance = (codes[:, None] != codes[None, :]).sum(-1)
    permutations = np.array(list(itertools.permutations(range(6))), dtype=np.int64)
    pair_distances = distance[permutations[:, :, None], permutations[:, None, :]]
    risk = np.einsum('scd,pcd->ps', q, pair_distances) / (6 * codes.shape[1])
    worst, mean = risk.max(1), risk.mean(1)
    # Lexicographic enumeration puts identity first, implementing both tie rules.
    wi = int(np.flatnonzero(worst <= worst.min() + 1e-12)[0])
    mi = int(np.flatnonzero(mean <= mean.min() + 1e-12)[0])
    return dict(counts=counts.tolist(), support=support.tolist(), q=q.tolist(),
                initial_codes=codes.tolist(), reference=list(range(6)),
                robust=permutations[wi].tolist(), mean=permutations[mi].tolist(),
                reference_risk=risk[0].tolist(), robust_risk=risk[wi].tolist(),
                mean_risk=risk[mi].tolist(), worst_before=float(worst[0]),
                worst_after=float(worst[wi]), minimum_distance=int(distance[distance > 0].min()),
                distinct_assignment=bool(wi != 0), permutations=permutations.tolist(),
                all_risks=risk.tolist())
