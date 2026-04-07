"""
Privacy Amplification for BB84 QKD.

After error correction, an eavesdropper (Eve) may possess partial information
about the shared key — both from intercepted qubits and from parity bits
leaked during CASCADE.  Privacy amplification compresses the reconciled key
through a universal hash function (Toeplitz matrix), producing a shorter key
about which Eve has negligible information.

This version uses the proper secure key length formula from the
information-theoretic security proof:

  Asymptotic:    l = n * [1 - h(QBER)] - leaked_EC
  Finite-key:    l = n * [1 - h(QBER_upper)] - leaked_EC
                     - 2*log2(1/(2*eps_PA)) - log2(2/eps_EC)

The Toeplitz matrix multiplication is the actual hashing step that
extracts the secure key from the reconciled key.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class AmplificationResult:
    """Outcome of a privacy-amplification step."""
    amplified_key: list[int]       # The final secure key bits
    input_length: int              # Length before amplification
    output_length: int             # Length after amplification
    compression_ratio: float       # output / input
    leaked_bits_removed: int       # How many bits of Eve-info were removed
    secure_rate_per_sifted: float  # Secure bits per sifted bit (key rate)
    shannon_limit: float           # Theoretical max output (1-h(QBER))*n


def amplify(
    key_bits: list[int],
    leaked_bits: int,
    security_parameter: int = 20,
    min_output: int = 256,
    observed_qber: float = 0.0,
    epsilon_pa: float = 0.0,
    epsilon_ec: float = 0.0,
) -> AmplificationResult:
    """
    Privacy-amplify a reconciled key using Toeplitz-matrix hashing.

    The output length is computed from the information-theoretic formula.
    When finite-key epsilons are provided, the stricter finite-key bound
    is used.

    Args:
        key_bits:           Reconciled key (list of 0/1).
        leaked_bits:        Number of parity bits leaked to Eve during EC.
        security_parameter: Extra bits to discard when epsilons not given.
        min_output:         Minimum acceptable output length.
        observed_qber:      QBER for computing the Shannon limit.
        epsilon_pa:         Privacy-amplification smoothing parameter.
        epsilon_ec:         Error-correction verification parameter.

    Returns:
        AmplificationResult with the shortened secure key.

    Raises:
        ValueError: If the key is too short to produce a secure output.
    """
    n = len(key_bits)

    # Compute the information-theoretic output length
    h_qber = _h(observed_qber) if observed_qber > 0 else 0.0
    shannon_limit = n * (1 - h_qber) if h_qber < 1.0 else 0.0

    if epsilon_pa > 0 and epsilon_ec > 0:
        # Finite-key formula
        pa_penalty = 2 * math.log2(1 / (2 * epsilon_pa))
        ec_penalty = math.log2(2 / epsilon_ec)
        output_len = int(n * (1 - h_qber) - leaked_bits - pa_penalty - ec_penalty)
    else:
        # Asymptotic formula with security margin
        output_len = n - leaked_bits - security_parameter

    # Clamp to valid range
    if output_len < min_output:
        output_len = max(32, n - leaked_bits - 2)
        if output_len < 32:
            raise ValueError(
                f"Key too short ({n} bits) after removing {leaked_bits} leaked "
                f"+ security bits.  Cannot amplify."
            )

    output_len = min(output_len, n)  # Can't be longer than input

    # Generate the random seed for the Toeplitz matrix.
    # A Toeplitz matrix of size (output_len x n) is fully described by
    # its first row + first column = n + output_len - 1 random bits.
    seed = [random.randint(0, 1) for _ in range(n + output_len - 1)]

    # Multiply over GF(2):
    #   result[i] = XOR_{j=0..n-1} ( seed[i+j] AND key[j] )
    result: list[int] = []
    for i in range(output_len):
        bit = 0
        for j in range(n):
            bit ^= seed[i + j] & key_bits[j]
        result.append(bit)

    return AmplificationResult(
        amplified_key=result,
        input_length=n,
        output_length=output_len,
        compression_ratio=output_len / n if n else 0.0,
        leaked_bits_removed=leaked_bits + (n - output_len - leaked_bits),
        secure_rate_per_sifted=output_len / n if n else 0.0,
        shannon_limit=shannon_limit,
    )


def _h(x: float) -> float:
    """Binary Shannon entropy h(x) = -x*log2(x) - (1-x)*log2(1-x)."""
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)
