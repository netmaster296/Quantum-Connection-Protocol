"""
Decoy-State BB84 with Biased Basis Selection.

Standard BB84 uses a 50/50 basis choice, meaning ~50% of pulses are
discarded during sifting.  Two independent improvements raise throughput:

  1. **Biased basis selection** (Lo, Chau, Ardehali, 2004):
     Alice and Bob both favour the Z-basis with probability p_z >> 0.5.
     Because the Z-basis sifting efficiency approaches 1 when p_z -> 1,
     the overall sifting yield jumps from ~0.5 to ~p_z^2 (typically ~0.9).
     The X-basis data is still needed to bound Eve's information but
     most key material comes from Z.

  2. **Decoy states** (Hwang, 2003; Lo, Ma, Chen, 2005):
     Real sources are weak coherent pulses (WCPs), not single photons.
     Multi-photon pulses let Eve perform a photon-number-splitting (PNS)
     attack.  By randomly varying the pulse intensity among signal (mu),
     decoy (nu), and vacuum (0), Alice and Bob can tightly bound the
     single-photon yield Y_1 and single-photon QBER e_1, giving security
     comparable to a perfect single-photon source.

This module provides:
  - BiasedBasisSelector: generates biased basis choices
  - DecoyProtocol: three-intensity decoy analysis
  - Combined helper that feeds into the main QKD pipeline
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


# ===================================================================
# Biased basis selection
# ===================================================================

class BiasedBasisSelector:
    """
    Generate biased basis choices for Alice and Bob.

    Args:
        p_z: Probability of choosing the Z (computational) basis.
             0.5 = standard BB84, 0.9 = typical efficient choice.
    """

    def __init__(self, p_z: float = 0.9):
        if not 0.0 < p_z < 1.0:
            raise ValueError("p_z must be in (0, 1)")
        self.p_z = p_z

    def choose_bases(self, n: int) -> list[int]:
        """
        Return n basis choices.  0 = Z-basis, 1 = X-basis.
        """
        return [0 if random.random() < self.p_z else 1 for _ in range(n)]

    @property
    def sifting_efficiency(self) -> float:
        """
        Expected sifting yield.

        P(both Z) + P(both X) = p_z^2 + (1-p_z)^2
        Standard BB84: 0.5^2 + 0.5^2 = 0.5
        Biased (0.9) : 0.9^2 + 0.1^2 = 0.82
        """
        return self.p_z ** 2 + (1 - self.p_z) ** 2


# ===================================================================
# Decoy-state analysis
# ===================================================================

@dataclass
class DecoyEstimate:
    """Result of decoy-state parameter estimation."""
    Y_1_lower: float        # Lower bound on single-photon yield
    e_1_upper: float        # Upper bound on single-photon QBER
    Q_mu: float             # Overall gain for signal intensity
    E_mu: float             # Overall QBER for signal intensity
    key_rate_per_pulse: float  # Asymptotic key rate (bits per pulse)
    mu: float               # Signal intensity used
    nu: float               # Decoy intensity used


@dataclass
class DecoyProtocol:
    """
    Three-intensity decoy-state BB84 analysis.

    Intensities:
        mu  — signal (typically 0.3–0.8 photons/pulse)
        nu  — weak decoy (typically 0.05–0.2)
        vacuum (0) — for background estimation

    The fractions p_mu, p_nu, p_vac control how often each intensity is
    sent.  They must sum to 1.

    Attrs:
        mu, nu: mean photon numbers.
        p_mu, p_nu: sending probabilities (p_vac = 1 - p_mu - p_nu).
    """
    mu: float = 0.48
    nu: float = 0.10
    p_mu: float = 0.60
    p_nu: float = 0.25

    @property
    def p_vac(self) -> float:
        return 1.0 - self.p_mu - self.p_nu

    # -- Intensity selection -----------------------------------------------

    def choose_intensities(self, n: int) -> list[float]:
        """
        Randomly assign an intensity to each of n pulses.

        Returns list of mu, nu, or 0.0 per pulse.
        """
        intensities: list[float] = []
        for _ in range(n):
            r = random.random()
            if r < self.p_mu:
                intensities.append(self.mu)
            elif r < self.p_mu + self.p_nu:
                intensities.append(self.nu)
            else:
                intensities.append(0.0)
        return intensities

    # -- Decoy analysis (Lo–Ma–Chen bounds) --------------------------------

    def estimate(
        self,
        Q_mu: float,
        Q_nu: float,
        Q_vac: float,
        E_mu: float,
        E_nu: float,
    ) -> DecoyEstimate:
        """
        Bound Y_1 and e_1 using the three observed gains and QBERs.

        Args:
            Q_mu:  Gain (detection probability) for signal pulses.
            Q_nu:  Gain for decoy pulses.
            Q_vac: Gain for vacuum pulses (≈ dark-count rate).
            E_mu:  QBER for signal pulses.
            E_nu:  QBER for decoy pulses.

        The key formulas (Lo, Ma, Chen, PRL 94, 2005):

          Y_1 >= (mu / (mu*nu - nu^2)) * (Q_nu*e^nu - Q_vac*e^0*(nu^2/mu^2)
                  - (mu^2 - nu^2)/(mu^2) * Q_vac)

        Simplified lower bound used here:
          Y_1 >= (mu/(mu-nu)) * (Q_nu*e^nu - (nu^2/mu^2)*Q_mu*e^mu)

          e_1 <= (E_nu*Q_nu*e^nu - e_0*Y_0) / (Y_1 * nu)
        """
        mu, nu = self.mu, self.nu

        # Poisson weights
        e_mu = math.exp(mu)
        e_nu = math.exp(nu)

        # Background / vacuum yield
        Y_0 = max(Q_vac, 1e-12)
        e_0 = 0.5  # Dark counts are random

        # Lower bound on single-photon yield Y_1
        Y_1_lower = max(
            (mu / (mu * nu - nu ** 2)) *
            (Q_nu * e_nu - Q_vac * (nu ** 2 / mu ** 2) * e_mu
             - ((mu ** 2 - nu ** 2) / mu ** 2) * Y_0),
            0.0,
        )

        # Upper bound on single-photon error rate e_1
        if Y_1_lower > 0 and nu > 0:
            e_1_upper = min(
                (E_nu * Q_nu * e_nu - e_0 * Y_0) / (Y_1_lower * nu),
                0.5,
            )
            e_1_upper = max(e_1_upper, 0.0)
        else:
            e_1_upper = 0.5

        # Asymptotic secure key rate (GLLP formula)
        key_rate = self._gllp_rate(Y_1_lower, e_1_upper, Q_mu, E_mu, mu)

        return DecoyEstimate(
            Y_1_lower=Y_1_lower,
            e_1_upper=e_1_upper,
            Q_mu=Q_mu,
            E_mu=E_mu,
            key_rate_per_pulse=key_rate,
            mu=mu,
            nu=nu,
        )

    def estimate_from_channel(
        self,
        transmittance: float,
        e_misalign: float = 0.015,
        dark_count_prob: float = 1e-6,
        detector_efficiency: float = 0.10,
    ) -> DecoyEstimate:
        """
        Run decoy analysis using theoretical channel parameters
        (no simulation needed — pure analytical calculation).
        """
        eta = transmittance * detector_efficiency
        p_dark = dark_count_prob

        def _gain(intensity: float) -> float:
            """Q_k = 1 - (1 - 2*p_dark) * exp(-eta * k)"""
            return 1 - (1 - 2 * p_dark) * math.exp(-eta * intensity)

        def _qber(intensity: float, gain: float) -> float:
            """E_k = (e_misalign * (1 - exp(-eta*k)) + p_dark) / Q_k"""
            if gain <= 0:
                return 0.5
            return (e_misalign * (1 - math.exp(-eta * intensity)) + p_dark) / gain

        Q_mu = _gain(self.mu)
        Q_nu = _gain(self.nu)
        Q_vac = _gain(0.0)  # = 2 * p_dark approximately
        E_mu = _qber(self.mu, Q_mu)
        E_nu = _qber(self.nu, Q_nu)

        return self.estimate(Q_mu, Q_nu, Q_vac, E_mu, E_nu)

    # -- GLLP key-rate formula ---------------------------------------------

    @staticmethod
    def _gllp_rate(
        Y_1: float, e_1: float, Q_mu: float, E_mu: float, mu: float,
    ) -> float:
        """
        Gottesman–Lo–Lütkenhaus–Preskill key rate formula.

        R >= q * { -Q_mu * f * h(E_mu) + Q_1 * [1 - h(e_1)] }

        where:
          q   = sifting efficiency (1/2 for standard BB84)
          f   = error-correction efficiency (CASCADE ≈ 1.16)
          h() = binary entropy
          Q_1 = Y_1 * mu * exp(-mu)  (single-photon gain)
        """
        if Q_mu <= 0 or Y_1 <= 0:
            return 0.0

        f_ec = 1.16  # Typical CASCADE efficiency
        q = 0.5      # Will be overridden by biased sifting in full protocol

        Q_1 = Y_1 * mu * math.exp(-mu)

        rate = q * (
            -Q_mu * f_ec * _h(E_mu) + Q_1 * (1 - _h(e_1))
        )
        return max(rate, 0.0)


# ===================================================================
# Helpers
# ===================================================================

def _h(x: float) -> float:
    """Binary Shannon entropy h(x) = -x*log2(x) - (1-x)*log2(1-x)."""
    if x <= 0 or x >= 1:
        return 0.0
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)
