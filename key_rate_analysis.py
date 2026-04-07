"""
Key-Rate Analysis Engine for BB84 QKD.

Provides sweep functions that compute the secret key rate as a function
of distance, QBER, epsilon, and other parameters — both for asymptotic
and finite-key regimes, with and without decoy states.

Used by the visualization dashboard to generate the key-rate plots, and
by the demo to compare standard vs decoy vs biased protocols.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from noise_model import ChannelModel
from decoy_state import DecoyProtocol, BiasedBasisSelector, _h
from finite_key import finite_key_analysis, asymptotic_key_rate


# ===================================================================
# Sweep result container
# ===================================================================

@dataclass
class KeyRateSweepResult:
    """Result of a parameter sweep."""
    x_values: list[float]        # Independent variable (distance, QBER, …)
    x_label: str                 # Axis label
    # Each series is (label, [y-values])
    series: list[tuple[str, list[float]]] = field(default_factory=list)


# ===================================================================
# Distance sweep
# ===================================================================

def sweep_distance(
    distances_km: list[float] | None = None,
    channel: ChannelModel | None = None,
    decoy: DecoyProtocol | None = None,
    biased_p_z: float = 0.9,
    block_size: int = 100_000,
    epsilon: float = 1e-10,
    f_ec: float = 1.16,
) -> KeyRateSweepResult:
    """
    Compute key rate vs fibre distance for multiple protocol variants.

    Produces four series:
      1. Asymptotic standard BB84 (unbiased, no decoy)
      2. Asymptotic decoy-state BB84 (biased basis)
      3. Finite-key standard BB84
      4. Finite-key decoy-state BB84

    Args:
        distances_km: List of distances to evaluate (default: 0–200 km).
        channel:      Base ChannelModel (distance will be varied).
        decoy:        DecoyProtocol config (default parameters if None).
        biased_p_z:   Z-basis probability for biased selection.
        block_size:   Total pulses sent per round (for finite-key).
        epsilon:      Finite-key security parameter.
        f_ec:         Error-correction efficiency.
    """
    if distances_km is None:
        distances_km = [d for d in range(0, 205, 5)]
    if channel is None:
        channel = ChannelModel()
    if decoy is None:
        decoy = DecoyProtocol()

    biased = BiasedBasisSelector(biased_p_z)

    asymp_standard: list[float] = []
    asymp_decoy: list[float] = []
    fk_standard: list[float] = []
    fk_decoy: list[float] = []

    for d in distances_km:
        ch = channel.at_distance(d)
        qber = ch.expected_qber
        eta = ch.overall_efficiency
        p_det = ch.detection_probability

        # --- Asymptotic standard BB84 ---
        r_asymp_std = asymptotic_key_rate(qber, sifting_efficiency=0.5, f_ec=f_ec)
        # Scale by detection probability (only detected pulses contribute)
        asymp_standard.append(r_asymp_std * p_det)

        # --- Asymptotic decoy-state BB84 with biased basis ---
        dec_est = decoy.estimate_from_channel(
            transmittance=ch.transmittance,
            e_misalign=ch.misalignment_error,
            dark_count_prob=ch.dark_count_prob,
            detector_efficiency=ch.detector_efficiency,
        )
        # Apply biased sifting efficiency
        r_asymp_dec = dec_est.key_rate_per_pulse * biased.sifting_efficiency / 0.5
        asymp_decoy.append(max(r_asymp_dec, 0.0))

        # --- Finite-key standard BB84 ---
        n_sifted_std = int(block_size * p_det * 0.5)  # 50% sifting
        fk_std = finite_key_analysis(
            n_sifted=n_sifted_std,
            n_raw=block_size,
            observed_qber=qber,
            leaked_ec_bits=int(n_sifted_std * f_ec * _h(qber)),
            epsilon_sec=epsilon,
            f_ec=f_ec,
        )
        fk_standard.append(fk_std.key_rate_per_pulse)

        # --- Finite-key decoy-state BB84 ---
        n_sifted_dec = int(block_size * p_det * biased.sifting_efficiency)
        fk_dec = finite_key_analysis(
            n_sifted=n_sifted_dec,
            n_raw=block_size,
            observed_qber=min(dec_est.e_1_upper, qber),
            leaked_ec_bits=int(n_sifted_dec * f_ec * _h(qber)),
            epsilon_sec=epsilon,
            f_ec=f_ec,
        )
        fk_decoy.append(fk_dec.key_rate_per_pulse)

    return KeyRateSweepResult(
        x_values=distances_km,
        x_label="Distance (km)",
        series=[
            ("Asymptotic BB84", asymp_standard),
            ("Asymptotic Decoy+Biased", asymp_decoy),
            (f"Finite-key BB84 (N={block_size:.0e}, eps={epsilon:.0e})", fk_standard),
            (f"Finite-key Decoy (N={block_size:.0e}, eps={epsilon:.0e})", fk_decoy),
        ],
    )


# ===================================================================
# QBER sweep
# ===================================================================

def sweep_qber(
    qber_values: list[float] | None = None,
    block_size: int = 100_000,
    epsilon: float = 1e-10,
    f_ec: float = 1.16,
    sifting_efficiency: float = 0.5,
) -> KeyRateSweepResult:
    """
    Compute key rate vs QBER (at fixed distance / block size).

    Useful for understanding the QBER tolerance of different protocols.
    """
    if qber_values is None:
        qber_values = [i * 0.005 for i in range(0, 23)]  # 0% to 11%

    asymp: list[float] = []
    fk: list[float] = []

    for q in qber_values:
        r = asymptotic_key_rate(q, sifting_efficiency=sifting_efficiency, f_ec=f_ec)
        asymp.append(r)

        n_sifted = int(block_size * sifting_efficiency)
        fk_res = finite_key_analysis(
            n_sifted=n_sifted,
            n_raw=block_size,
            observed_qber=q,
            leaked_ec_bits=int(n_sifted * f_ec * _h(q)),
            epsilon_sec=epsilon,
            f_ec=f_ec,
        )
        fk.append(fk_res.key_rate_per_pulse)

    return KeyRateSweepResult(
        x_values=[q * 100 for q in qber_values],
        x_label="QBER (%)",
        series=[
            ("Asymptotic", asymp),
            (f"Finite-key (N={block_size:.0e}, eps={epsilon:.0e})", fk),
        ],
    )


# ===================================================================
# Eve attack analysis
# ===================================================================

@dataclass
class EveAttackResult:
    """Comparison of key rates with and without an eavesdropper."""
    qber_no_eve: float
    qber_with_eve: float
    key_rate_no_eve: float
    key_rate_with_eve: float
    eve_info_fraction: float  # Fraction of key known to Eve before PA
    protocol_aborted: bool
    threshold: float


def analyze_eve_attack(
    channel: ChannelModel | None = None,
    eve_interception_rate: float = 1.0,
    error_threshold: float = 0.11,
    block_size: int = 100_000,
    epsilon: float = 1e-10,
    f_ec: float = 1.16,
) -> EveAttackResult:
    """
    Analyze the impact of an intercept-resend eavesdropper.

    Args:
        channel:               Physical channel model.
        eve_interception_rate: Fraction of pulses Eve intercepts (0–1).
        error_threshold:       QBER above which protocol aborts.
        block_size:            Pulses per round.
        epsilon:               Finite-key security parameter.
    """
    if channel is None:
        channel = ChannelModel(distance_km=20)

    qber_clean = channel.expected_qber

    # Eve's intercept-resend attack adds ~25% QBER to intercepted fraction
    # (she guesses the wrong basis 50% of the time, causing 50% errors
    # on those qubits → 0.5 * 0.5 = 0.25 added QBER)
    qber_eve = qber_clean + 0.25 * eve_interception_rate
    aborted = qber_eve > error_threshold

    # Key rate without Eve
    n_sifted = int(block_size * channel.detection_probability * 0.5)
    fk_clean = finite_key_analysis(
        n_sifted=n_sifted, n_raw=block_size,
        observed_qber=qber_clean,
        leaked_ec_bits=int(n_sifted * f_ec * _h(qber_clean)),
        epsilon_sec=epsilon, f_ec=f_ec,
    )

    # Key rate with Eve (if not aborted)
    fk_eve = finite_key_analysis(
        n_sifted=n_sifted, n_raw=block_size,
        observed_qber=qber_eve,
        leaked_ec_bits=int(n_sifted * f_ec * _h(qber_eve)),
        epsilon_sec=epsilon, f_ec=f_ec,
    )

    # Eve's information fraction: she learns about 2*QBER_added fraction
    # of the key bits before privacy amplification
    eve_info = min(eve_interception_rate, 1.0)

    return EveAttackResult(
        qber_no_eve=qber_clean,
        qber_with_eve=qber_eve,
        key_rate_no_eve=fk_clean.key_rate_per_pulse,
        key_rate_with_eve=0.0 if aborted else fk_eve.key_rate_per_pulse,
        eve_info_fraction=eve_info,
        protocol_aborted=aborted,
        threshold=error_threshold,
    )


def sweep_eve_interception(
    channel: ChannelModel | None = None,
    interception_rates: list[float] | None = None,
    error_threshold: float = 0.11,
    block_size: int = 100_000,
    epsilon: float = 1e-10,
) -> KeyRateSweepResult:
    """
    Sweep Eve's interception rate and show key rate degradation.
    """
    if interception_rates is None:
        interception_rates = [i * 0.02 for i in range(51)]  # 0% to 100%

    rates: list[float] = []
    qbers: list[float] = []

    for ir in interception_rates:
        res = analyze_eve_attack(
            channel=channel,
            eve_interception_rate=ir,
            error_threshold=error_threshold,
            block_size=block_size,
            epsilon=epsilon,
        )
        rates.append(res.key_rate_with_eve)
        qbers.append(res.qber_with_eve * 100)

    return KeyRateSweepResult(
        x_values=[r * 100 for r in interception_rates],
        x_label="Eve Interception Rate (%)",
        series=[
            ("Key Rate (bits/pulse)", rates),
            ("QBER (%)", qbers),
        ],
    )
