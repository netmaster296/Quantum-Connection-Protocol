"""
CASCADE Error Correction for BB84 QKD.

Implements the CASCADE protocol (Brassard & Salvail, 1993) which corrects
bit errors between Alice's and Bob's sifted keys using only parity
comparisons over the public classical channel.

The protocol works in multiple passes with increasing block sizes.  When an
error is found and corrected via binary search, the correction *cascades*
back through all previous-pass blocks that contain the corrected bit,
potentially uncovering additional errors.

Every parity bit exchanged leaks one bit of information to a potential
eavesdropper, so the total leak count is tracked and fed into privacy
amplification.

This version also tracks:
  - EC efficiency factor f = leaked_bits / (n * h(QBER)), measuring how
    close CASCADE comes to the Shannon limit (f=1.0 is perfect).
  - Per-pass statistics for diagnostics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class CascadePassStats:
    """Statistics for a single CASCADE pass."""
    pass_number: int
    block_size: int
    blocks_checked: int
    errors_found: int
    parity_bits_exchanged: int


@dataclass
class CascadeResult:
    """Outcome of a CASCADE error-correction run."""
    corrected_key: list[int]            # Bob's corrected key
    errors_corrected: int               # Total bit flips applied
    leaked_bits: int                    # Parity bits exchanged (info leaked to Eve)
    passes_run: int                     # Number of passes completed
    residual_error_rate: float          # Estimated remaining error rate
    ec_efficiency: float = 1.0          # f = leaked / (n * h(QBER)), lower is better
    pass_stats: list[CascadePassStats] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cascade_correct(
    alice_key: list[int],
    bob_key: list[int],
    estimated_qber: float = 0.05,
    num_passes: int = 4,
) -> CascadeResult:
    """
    Run the CASCADE error-correction protocol.

    Args:
        alice_key:      Alice's sifted key bits.
        bob_key:        Bob's sifted key bits (same length, may contain errors).
        estimated_qber: Estimated quantum bit error rate (used to size blocks).
        num_passes:     Number of CASCADE passes (typically 4–10).

    Returns:
        CascadeResult with corrected key, leakage, and efficiency stats.
    """
    n = len(alice_key)
    assert len(bob_key) == n, "Keys must be the same length."

    corrected = list(bob_key)
    leaked = 0
    total_corrections = 0
    pass_stats: list[CascadePassStats] = []

    # Initial block size from QBER:  k1 ~ 0.73 / QBER  (Brassard & Salvail)
    if estimated_qber > 0:
        k1 = max(4, int(0.73 / estimated_qber))
    else:
        k1 = n

    # Store per-pass shuffled index orders so we can cascade back
    pass_info: list[tuple[list[int], int]] = []

    for pass_num in range(num_passes):
        block_size = k1 * (2 ** pass_num)
        block_size = min(block_size, n)

        pass_leaked = 0
        pass_errors = 0
        pass_blocks = 0

        # First pass uses natural order; later passes shuffle
        if pass_num == 0:
            indices = list(range(n))
        else:
            indices = list(range(n))
            random.shuffle(indices)

        pass_info.append((indices, block_size))

        # Process every block in this pass
        for blk_start in range(0, n, block_size):
            blk_end = min(blk_start + block_size, n)
            blk_indices = indices[blk_start:blk_end]
            pass_blocks += 1

            a_par = _parity(alice_key, blk_indices)
            b_par = _parity(corrected, blk_indices)
            leaked += 1
            pass_leaked += 1

            if a_par != b_par:
                err_idx, leak = _binary_search(alice_key, corrected, blk_indices)
                leaked += leak
                pass_leaked += leak

                if err_idx is not None:
                    corrected[err_idx] ^= 1
                    total_corrections += 1
                    pass_errors += 1

                    # Cascade back through all previous passes
                    for prev_p in range(pass_num):
                        prev_idx, prev_bs = pass_info[prev_p]
                        pos = prev_idx.index(err_idx)
                        pb_start = (pos // prev_bs) * prev_bs
                        pb_end = min(pb_start + prev_bs, n)
                        pb_indices = prev_idx[pb_start:pb_end]

                        pa = _parity(alice_key, pb_indices)
                        pb = _parity(corrected, pb_indices)
                        leaked += 1
                        pass_leaked += 1

                        if pa != pb:
                            e2, lk2 = _binary_search(
                                alice_key, corrected, pb_indices
                            )
                            leaked += lk2
                            pass_leaked += lk2
                            if e2 is not None:
                                corrected[e2] ^= 1
                                total_corrections += 1
                                pass_errors += 1

        pass_stats.append(CascadePassStats(
            pass_number=pass_num + 1,
            block_size=block_size,
            blocks_checked=pass_blocks,
            errors_found=pass_errors,
            parity_bits_exchanged=pass_leaked,
        ))

    # Estimate residual errors
    residual = _estimate_residual(alice_key, corrected)

    # Compute EC efficiency factor:  f = leaked / (n * h(QBER))
    # Shannon limit is f = 1.0; practical CASCADE achieves f ~ 1.05–1.20
    h_qber = _h(estimated_qber)
    if h_qber > 0 and n > 0:
        ec_efficiency = leaked / (n * h_qber)
    else:
        ec_efficiency = 1.0  # No errors → efficiency is moot

    return CascadeResult(
        corrected_key=corrected,
        errors_corrected=total_corrections,
        leaked_bits=leaked,
        passes_run=num_passes,
        residual_error_rate=residual,
        ec_efficiency=ec_efficiency,
        pass_stats=pass_stats,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parity(key: list[int], indices: list[int]) -> int:
    """Compute parity (XOR) of key bits at the given positions."""
    p = 0
    for i in indices:
        p ^= key[i]
    return p


def _binary_search(
    alice_key: list[int],
    bob_key: list[int],
    indices: list[int],
) -> tuple[int | None, int]:
    """
    Binary search for a single error within *indices*.

    Returns:
        (index_of_error, parity_bits_leaked)
    """
    if len(indices) == 1:
        return indices[0], 0

    mid = len(indices) // 2
    left = indices[:mid]

    a_par = _parity(alice_key, left)
    b_par = _parity(bob_key, left)
    leaked = 1

    if a_par != b_par:
        idx, more = _binary_search(alice_key, bob_key, left)
    else:
        idx, more = _binary_search(alice_key, bob_key, indices[mid:])

    return idx, leaked + more


def _estimate_residual(
    alice_key: list[int],
    bob_key: list[int],
    sample_size: int = 0,
) -> float:
    """Quick residual-error estimate (for diagnostics only)."""
    n = len(alice_key)
    if sample_size <= 0 or sample_size > n:
        sample_size = min(n, 256)
    idxs = random.sample(range(n), sample_size)
    errors = sum(alice_key[i] != bob_key[i] for i in idxs)
    return errors / sample_size


def _h(x: float) -> float:
    """Binary Shannon entropy."""
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)
