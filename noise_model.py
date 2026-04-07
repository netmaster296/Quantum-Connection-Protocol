"""
Quantum Channel Noise Models for BB84 QKD Simulation.

Two layers of modelling:

  1. **Qiskit noise** — depolarizing / bit-flip / phase-flip errors injected
     into the Aer simulator to corrupt qubit operations and measurements.

  2. **ChannelModel** — a physics-based fibre-optic channel that accounts for
     photon loss (Beer–Lambert), detector dark counts, detector efficiency,
     and optical misalignment.  The channel model drives both the Qiskit noise
     rate and the photon-survival mask used during simulation.

The two layers work together: ChannelModel computes the expected QBER from
physical parameters, which is mapped to an equivalent depolarizing rate for
Qiskit, *and* it generates a per-pulse loss mask so that the simulation
faithfully reproduces the detection statistics of a real fibre link.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error


# ===================================================================
# Qiskit-level noise presets (kept for backward compatibility)
# ===================================================================

NOISE_PRESETS = {
    "ideal":        {"type": "depolarizing", "rate": 0.0},
    "low":          {"type": "depolarizing", "rate": 0.01},
    "moderate":     {"type": "depolarizing", "rate": 0.03},
    "high":         {"type": "depolarizing", "rate": 0.08},
    "fiber_10km":   {"type": "combined",     "rate": 0.02},
    "fiber_50km":   {"type": "combined",     "rate": 0.05},
    "free_space":   {"type": "combined",     "rate": 0.03},
}


def create_noise_model(error_type: str = "depolarizing",
                        error_rate: float = 0.02) -> NoiseModel | None:
    """
    Build a Qiskit Aer NoiseModel for the quantum channel.

    Args:
        error_type: One of 'depolarizing', 'bit_flip', 'phase_flip', 'combined'.
        error_rate: Per-gate error probability (0.0 – 1.0).

    Returns:
        Configured NoiseModel, or None if error_rate is 0.
    """
    if error_rate <= 0.0:
        return None

    noise_model = NoiseModel()

    if error_type == "depolarizing":
        gate_error = depolarizing_error(error_rate, 1)
        noise_model.add_all_qubit_quantum_error(gate_error, ["x", "h", "id"])

    elif error_type == "bit_flip":
        gate_error = pauli_error([("X", error_rate), ("I", 1 - error_rate)])
        noise_model.add_all_qubit_quantum_error(gate_error, ["x", "h", "id"])

    elif error_type == "phase_flip":
        gate_error = pauli_error([("Z", error_rate), ("I", 1 - error_rate)])
        noise_model.add_all_qubit_quantum_error(gate_error, ["x", "h", "id"])

    elif error_type == "combined":
        gate_error = depolarizing_error(error_rate, 1)
        meas_error = pauli_error([
            ("X", error_rate / 2),
            ("I", 1 - error_rate / 2),
        ])
        noise_model.add_all_qubit_quantum_error(gate_error, ["x", "h", "id"])
        noise_model.add_all_qubit_quantum_error(meas_error, ["measure"])

    else:
        raise ValueError(
            f"Unknown noise type '{error_type}'. "
            "Choose from: depolarizing, bit_flip, phase_flip, combined."
        )

    return noise_model


def from_preset(name: str) -> NoiseModel | None:
    """Create a noise model from a named preset (see NOISE_PRESETS)."""
    if name not in NOISE_PRESETS:
        raise ValueError(
            f"Unknown preset '{name}'. Available: {list(NOISE_PRESETS.keys())}"
        )
    cfg = NOISE_PRESETS[name]
    return create_noise_model(cfg["type"], cfg["rate"])


# ===================================================================
# Physics-based channel model
# ===================================================================

# Common detector configurations
DETECTOR_PRESETS = {
    "ingaas_spad": {  # Telecom InGaAs SPAD (most common commercial)
        "detector_efficiency": 0.10,
        "dark_count_prob": 1e-6,
    },
    "snspd": {  # Superconducting nanowire SPD (state of the art)
        "detector_efficiency": 0.93,
        "dark_count_prob": 1e-8,
    },
    "si_spad": {  # Silicon SPAD (visible wavelength, short range)
        "detector_efficiency": 0.50,
        "dark_count_prob": 1e-5,
    },
}


@dataclass
class ChannelModel:
    """
    Physics-based fibre-optic QKD channel.

    Models the three dominant impairments in a real BB84 link:

      - **Photon loss**: exponential attenuation in fibre (Beer–Lambert law).
        Standard telecom fibre (SMF-28) has alpha ~ 0.2 dB/km at 1550 nm.

      - **Dark counts**: spurious detector clicks from thermal noise, after-
        pulsing, or stray light.  Expressed as probability per detector gate.

      - **Misalignment error**: optical imperfections causing bit-flip errors
        even for correctly-basis-matched detections.

    Attributes:
        fiber_attenuation_db_km: Fibre loss in dB/km (default 0.2 for SMF-28).
        distance_km:             Fibre length.
        detector_efficiency:     Probability that a photon reaching the detector
                                 produces a click (eta_det).
        dark_count_prob:         Probability of a dark count per detector per gate
                                 window (p_dark).
        misalignment_error:      Intrinsic bit-flip probability from optical
                                 misalignment (e_align, typically 0.5–2%).
    """
    fiber_attenuation_db_km: float = 0.2
    distance_km: float = 10.0
    detector_efficiency: float = 0.10
    dark_count_prob: float = 1e-6
    misalignment_error: float = 0.015

    # -- Derived physical quantities ----------------------------------------

    @property
    def transmittance(self) -> float:
        """Channel transmittance: eta_ch = 10^(-alpha*L/10)."""
        return 10 ** (-self.fiber_attenuation_db_km * self.distance_km / 10)

    @property
    def overall_efficiency(self) -> float:
        """End-to-end efficiency: eta = eta_ch * eta_det."""
        return self.transmittance * self.detector_efficiency

    @property
    def detection_probability(self) -> float:
        """
        Probability of a click for a single-photon pulse.

        P(click) = 1 - (1 - 2*p_dark) * (1 - eta)
        The factor of 2 accounts for two detectors (0/1) in a BB84 receiver.
        """
        eta = self.overall_efficiency
        return 1 - (1 - 2 * self.dark_count_prob) * (1 - eta)

    @property
    def expected_qber(self) -> float:
        """
        Expected quantum bit error rate from channel physics.

        QBER = (e_align * eta + p_dark) / (eta + 2*p_dark)

        At short distances (high eta), QBER ~ e_align.
        At long distances (low eta), QBER -> 0.5 (noise floor).
        """
        eta = self.overall_efficiency
        p_dark = self.dark_count_prob
        denom = eta + 2 * p_dark
        if denom <= 0:
            return 0.5
        return (self.misalignment_error * eta + p_dark) / denom

    # -- Integration with Qiskit simulation ---------------------------------

    def to_depolarizing_rate(self) -> float:
        """
        Map expected QBER to an equivalent depolarizing error rate.

        A single-qubit depolarizing channel with parameter p gives
        QBER = 2p/3 (in the single-qubit formalism used by Qiskit Aer).
        """
        return min(self.expected_qber * 3 / 2, 0.75)

    def build_noise_model(self) -> NoiseModel | None:
        """Build a Qiskit NoiseModel matching this channel's QBER."""
        dep_rate = self.to_depolarizing_rate()
        if dep_rate <= 0:
            return None
        return create_noise_model("combined", dep_rate)

    # -- Photon loss mask ---------------------------------------------------

    def loss_mask(self, n: int) -> list[bool]:
        """
        Generate a per-pulse detection mask.

        Returns a list of n booleans; True = photon detected, False = lost.
        Lost pulses should be discarded before basis sifting (they carry
        no information, or at best a dark-count random bit).
        """
        p_det = self.detection_probability
        return [random.random() < p_det for _ in range(n)]

    def dark_count_bit(self) -> int:
        """Random bit produced by a dark-count event (uniformly random)."""
        return random.randint(0, 1)

    # -- Convenience --------------------------------------------------------

    @classmethod
    def from_detector_preset(cls, preset: str, distance_km: float = 10.0,
                              **kwargs) -> "ChannelModel":
        """
        Create a channel model with a named detector configuration.

        >>> ch = ChannelModel.from_detector_preset("snspd", distance_km=50)
        """
        if preset not in DETECTOR_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available: {list(DETECTOR_PRESETS.keys())}"
            )
        params = {**DETECTOR_PRESETS[preset], "distance_km": distance_km}
        params.update(kwargs)
        return cls(**params)

    def at_distance(self, km: float) -> "ChannelModel":
        """Return a copy of this channel at a different distance."""
        return ChannelModel(
            fiber_attenuation_db_km=self.fiber_attenuation_db_km,
            distance_km=km,
            detector_efficiency=self.detector_efficiency,
            dark_count_prob=self.dark_count_prob,
            misalignment_error=self.misalignment_error,
        )

    def __repr__(self) -> str:
        return (
            f"ChannelModel(L={self.distance_km}km, "
            f"eta_ch={self.transmittance:.2e}, "
            f"QBER={self.expected_qber:.4f}, "
            f"P_det={self.detection_probability:.4e})"
        )
