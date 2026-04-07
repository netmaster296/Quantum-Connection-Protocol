"""
Finite-Key Security Analysis for BB84 QKD.

Asymptotic key-rate formulas assume infinitely long keys.  In practice,
keys have finite length N, introducing statistical fluctuations in
parameter estimation that must be accounted for with a security
parameter epsilon (eps).

This module implements the finite-key analysis of Tomamichel et al.
(Nature Communications 3, 634, 2012) and Lim et al. (PRA 89, 022307,
2014), providing:

  - Tight finite-key bounds on QBER estimation
  - Finite-size-corrected privacy amplification lengths
  - Composable epsilon-security framework (eps_sec = eps_PA + eps_EC + eps_PE)
  - Secret key rate as a function of block size N and security parameter eps
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class FiniteKeyResult:
    """Output of finite-key analysis."""
    secure_key_length: int          # Maximum extractable secure bits
    key_rate_per_pulse: float       # Secure bits per raw pulse sent
    key_rate_per_sifted: float      # Secure bits per sifted bit
    epsilon_security: float         # Composable security parameter
    epsilon_EC: float               # Error-correction failure probability
    epsilon_PA: float               # Privacy-amplification smoothing
    epsilon_PE: float               # Parameter-estimation confidence
    qber_upper_bound: float         # Worst-case QBER (after finite stats)
    min_block_size: int             # Minimum N for positive key rate


def finite_key_analysis(
    n_sifted: int,
    n_raw: int,
    observed_qber: float,
    leaked_ec_bits: int,
    epsilon_sec: float = 1e-10,
    sample_fraction: float = 0.15,
    f_ec: float = 1.16,
) -> FiniteKeyResult:
    """
    Compute the finite-key secure key length.

    Args:
        n_sifted:        Number of sifted bits (after basis matching).
        n_raw:           Total raw pulses sent.
        observed_qber:   Measured QBER from the sample subset.
        leaked_ec_bits:  Parity bits leaked during error correction.
        epsilon_sec:     Total composable security parameter (default 1e-10).
        sample_fraction: Fraction of sifted bits used for QBER estimation.
        f_ec:            Error-correction efficiency factor (CASCADE ~ 1.16).

    Returns:
        FiniteKeyResult with all security parameters and the key length.

    The total security parameter decomposes as:
        eps_sec = eps_EC + eps_PA + eps_PE
    We split it equally by default.
    """
    # Split epsilon three ways
    eps_EC = epsilon_sec / 3
    eps_PA = epsilon_sec / 3
    eps_PE = epsilon_sec / 3

    # Bits used for parameter estimation vs key generation
    n_sample = max(8, int(n_sifted * sample_fraction))
    n_key = n_sifted - n_sample

    if n_key <= 0:
        return _empty_result(epsilon_sec, eps_EC, eps_PA, eps_PE)

    # --- Statistical bound on true QBER (Hoeffding inequality) ---
    # With confidence 1 - eps_PE, true QBER <= observed_qber + delta
    # where delta = sqrt( ln(1/eps_PE) / (2 * n_sample) )
    if eps_PE > 0 and n_sample > 0:
        delta = math.sqrt(math.log(1 / eps_PE) / (2 * n_sample))
    else:
        delta = 0.5
    qber_upper = min(observed_qber + delta, 0.5)

    # --- Secure key length (Tomamichel et al. 2012, Eq. 1) ---
    # l <= n_key * [1 - h(qber_upper)] - leaked_EC - correction_terms
    #
    # Privacy amplification penalty:
    #   2 * log2(1 / (2 * eps_PA))
    # Error verification:
    #   log2(2 / eps_EC)

    h_qber = _h(qber_upper)
    pa_penalty = 2 * math.log2(1 / (2 * eps_PA)) if eps_PA > 0 else 0

    # The information leaked by EC is bounded by f_ec * n_key * h(QBER)
    # plus the explicit parity bits.  Use the tighter of:
    # (a) the actual leaked_ec_bits, or (b) the theoretical bound.
    ec_leakage = max(leaked_ec_bits, f_ec * n_key * _h(observed_qber))

    secure_len = n_key * (1 - h_qber) - ec_leakage - pa_penalty
    # Subtract the log2(2/eps_EC) for error verification
    secure_len -= math.log2(2 / eps_EC) if eps_EC > 0 else 0

    secure_len = max(0, int(secure_len))

    # Key rates
    rate_per_pulse = secure_len / n_raw if n_raw > 0 else 0.0
    rate_per_sifted = secure_len / n_sifted if n_sifted > 0 else 0.0

    # Minimum block size for positive key rate (binary search)
    min_N = _find_min_block_size(observed_qber, epsilon_sec, f_ec, sample_fraction)

    return FiniteKeyResult(
        secure_key_length=secure_len,
        key_rate_per_pulse=rate_per_pulse,
        key_rate_per_sifted=rate_per_sifted,
        epsilon_security=epsilon_sec,
        epsilon_EC=eps_EC,
        epsilon_PA=eps_PA,
        epsilon_PE=eps_PE,
        qber_upper_bound=qber_upper,
        min_block_size=min_N,
    )


def asymptotic_key_rate(
    qber: float,
    sifting_efficiency: float = 0.5,
    f_ec: float = 1.16,
) -> float:
    """
    Asymptotic (infinite-key) secure key rate per sifted bit.

    R = sifting_eff * [1 - h(QBER) - f_ec * h(QBER)]

    For comparison with finite-key results.
    """
    if qber >= 0.5:
        return 0.0
    h_q = _h(qber)
    return max(0.0, sifting_efficiency * (1 - h_q - f_ec * h_q))


def key_rate_vs_epsilon(
    n_sifted: int,
    n_raw: int,
    observed_qber: float,
    leaked_ec_bits: int,
    epsilons: list[float] | None = None,
    f_ec: float = 1.16,
) -> list[tuple[float, float]]:
    """
    Sweep epsilon to show how security tightness affects key rate.

    Returns list of (epsilon, key_rate_per_pulse) pairs.
    """
    if epsilons is None:
        epsilons = [10**(-i) for i in range(3, 16)]

    results: list[tuple[float, float]] = []
    for eps in epsilons:
        fk = finite_key_analysis(
            n_sifted, n_raw, observed_qber, leaked_ec_bits,
            epsilon_sec=eps, f_ec=f_ec,
        )
        results.append((eps, fk.key_rate_per_pulse))
    return results


# ===================================================================
# Internal helpers
# ===================================================================

def _h(x: float) -> float:
    """Binary Shannon entropy."""
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def _find_min_block_size(
    qber: float,
    eps: float,
    f_ec: float,
    sample_frac: float,
    max_n: int = 10_000_000,
) -> int:
    """Binary search for the smallest N giving a positive key rate."""
    eps_PE = eps / 3
    eps_PA = eps / 3
    eps_EC = eps / 3

    lo, hi = 100, max_n
    if _key_len_at_n(hi, qber, eps_PE, eps_PA, eps_EC, f_ec, sample_frac) <= 0:
        return max_n  # Even max_n is insufficient

    while lo < hi:
        mid = (lo + hi) // 2
        if _key_len_at_n(mid, qber, eps_PE, eps_PA, eps_EC, f_ec, sample_frac) > 0:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _key_len_at_n(
    n: int, qber: float,
    eps_PE: float, eps_PA: float, eps_EC: float,
    f_ec: float, sample_frac: float,
) -> float:
    """Compute secure key length for a given sifted block size n."""
    n_sample = max(8, int(n * sample_frac))
    n_key = n - n_sample
    if n_key <= 0:
        return 0

    delta = math.sqrt(math.log(1 / eps_PE) / (2 * n_sample)) if eps_PE > 0 else 0.5
    qber_upper = min(qber + delta, 0.5)
    h_q = _h(qber_upper)

    pa_penalty = 2 * math.log2(1 / (2 * eps_PA)) if eps_PA > 0 else 0
    ec_leakage = f_ec * n_key * _h(qber)
    ec_verify = math.log2(2 / eps_EC) if eps_EC > 0 else 0

    return n_key * (1 - h_q) - ec_leakage - pa_penalty - ec_verify


def _empty_result(eps: float, eps_EC: float, eps_PA: float, eps_PE: float) -> FiniteKeyResult:
    return FiniteKeyResult(
        secure_key_length=0,
        key_rate_per_pulse=0.0,
        key_rate_per_sifted=0.0,
        epsilon_security=eps,
        epsilon_EC=eps_EC,
        epsilon_PA=eps_PA,
        epsilon_PE=eps_PE,
        qber_upper_bound=0.5,
        min_block_size=0,
    )
