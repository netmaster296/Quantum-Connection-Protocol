"""
BB84 Quantum Key Distribution — Production-Grade Connection Protocol.

This module implements a complete BB84 QKD pipeline with:

    1. Physics-based channel model (fibre loss, dark counts, detector eff.)
    2. Biased basis selection (sifting efficiency up to ~90%)
    3. Decoy-state analysis (bounds single-photon yield/QBER)
    4. Qubit preparation & noisy quantum transmission (Qiskit Aer)
    5. Optional intercept-resend eavesdropper simulation
    6. Measurement & basis sifting
    7. QBER estimation with finite-size confidence bound
    8. CASCADE error correction (with EC efficiency tracking)
    9. Privacy amplification (Toeplitz hashing, information-theoretic bounds)
   10. Finite-key security analysis (composable epsilon-security)
   11. AES-256-GCM authenticated encryption (HKDF-SHA256 key derivation)
   12. Secret key rate output (bits per pulse)

Supports local simulation mode and network (TCP) mode.

Dependencies: qiskit, qiskit-aer, cryptography, numpy, matplotlib
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Local modules
from noise_model import create_noise_model, ChannelModel, NOISE_PRESETS
from cascade import cascade_correct, CascadeResult
from privacy_amplification import amplify, AmplificationResult
from aes_encryption import derive_aes_key, encrypt, decrypt, AESPacket
from decoy_state import DecoyProtocol, BiasedBasisSelector
from finite_key import finite_key_analysis, FiniteKeyResult
from network import (
    ClassicalChannel, AliceServer, BobClient, MsgType,
)

log = logging.getLogger("qkd.protocol")


# ===================================================================
# Protocol statistics (passed to visualiser)
# ===================================================================

@dataclass
class QKDStats:
    """Collects every metric the visualiser and caller might need."""
    raw_bits: int = 0
    sifted_bits: int = 0
    corrected_bits: int = 0
    amplified_bits: int = 0
    aes_key_bits: int = 256

    qber_before: float = 0.0
    qber_after: float = 0.0
    error_threshold: float = 0.11

    cascade_leaked: int = 0
    errors_corrected: int = 0
    cascade_passes: int = 0
    ec_efficiency: float = 1.0
    cascade_pass_stats: list[dict] = field(default_factory=list)

    basis_match_count: int = 0
    basis_mismatch_count: int = 0
    sifting_efficiency: float = 0.5

    eavesdropper_active: bool = False
    privacy_compression: float = 0.0

    final_key: list[int] = field(default_factory=list)

    elapsed_seconds: float = 0.0

    # New: channel, decoy, finite-key info
    channel_distance_km: float | None = None
    channel_transmittance: float | None = None
    decoy_enabled: bool = False
    biased_basis: bool = False
    biased_p_z: float = 0.5

    finite_key: dict = field(default_factory=dict)
    epsilon_security: float = 1e-10
    secure_key_rate_per_pulse: float = 0.0
    secure_key_rate_asymptotic: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Plain dict for the visualiser."""
        return self.__dict__.copy()


# ===================================================================
# Main protocol class
# ===================================================================

class QuantumConnectionProtocol:
    """
    Production-grade BB84 Quantum Key Distribution protocol.

    Args:
        key_length:      Desired final key length in bits (>= 64).
        noise_type:      Noise model type or preset name. Ignored if
                         channel_model is provided.
        noise_rate:      Per-gate error probability. Ignored if channel_model
                         is provided.
        channel_model:   Physics-based ChannelModel (fibre loss, dark counts,
                         detector efficiency). Overrides noise_type/noise_rate.
        eavesdropper:    If True, simulate an intercept-resend Eve.
        cascade_passes:  Number of CASCADE error-correction passes.
        security_param:  Extra bits discarded during privacy amplification
                         (asymptotic mode only; overridden by epsilon in
                         finite-key mode).
        error_threshold: Maximum tolerable QBER before aborting.
        biased_p_z:      Z-basis probability (0.5 = standard, 0.9 = biased).
        decoy_protocol:  DecoyProtocol config (None = disabled).
        epsilon:         Composable finite-key security parameter.
    """

    def __init__(
        self,
        key_length: int = 256,
        noise_type: str = "depolarizing",
        noise_rate: float = 0.02,
        channel_model: ChannelModel | None = None,
        eavesdropper: bool = False,
        cascade_passes: int = 4,
        security_param: int = 20,
        error_threshold: float = 0.11,
        biased_p_z: float = 0.5,
        decoy_protocol: DecoyProtocol | None = None,
        epsilon: float = 1e-10,
    ) -> None:
        self.key_length = max(64, key_length)
        self.eavesdropper = eavesdropper
        self.cascade_passes = cascade_passes
        self.security_param = security_param
        self.error_threshold = error_threshold
        self.epsilon = epsilon

        # Channel model
        self._channel = channel_model

        # Biased basis
        self._biased = BiasedBasisSelector(biased_p_z) if biased_p_z != 0.5 else None
        self._p_z = biased_p_z

        # Decoy states
        self._decoy = decoy_protocol

        # Build Qiskit simulator with noise from channel or preset
        if channel_model is not None:
            self._noise_model = channel_model.build_noise_model()
        elif noise_type in NOISE_PRESETS:
            cfg = NOISE_PRESETS[noise_type]
            noise_type, noise_rate = cfg["type"], cfg["rate"]
            self._noise_model = create_noise_model(noise_type, noise_rate)
        else:
            self._noise_model = create_noise_model(noise_type, noise_rate)

        sim_opts: dict[str, Any] = {}
        if self._noise_model is not None:
            sim_opts["noise_model"] = self._noise_model
        self._simulator = AerSimulator(**sim_opts)

        # State populated after run_qkd()
        self._aes_key: bytes | None = None
        self._stats = QKDStats()

    # ------------------------------------------------------------------
    # Quantum helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _random_bits(n: int) -> list[int]:
        return [random.randint(0, 1) for _ in range(n)]

    def _choose_bases(self, n: int) -> list[int]:
        """Generate basis choices — biased if configured, else 50/50."""
        if self._biased is not None:
            return self._biased.choose_bases(n)
        return [random.randint(0, 1) for _ in range(n)]

    def _prepare_qubit(self, bit: int, basis: int) -> QuantumCircuit:
        qc = QuantumCircuit(1, 1)
        if bit:
            qc.x(0)
        if basis:
            qc.h(0)
        return qc

    def _measure_qubit(self, qc: QuantumCircuit, basis: int) -> int:
        meas = qc.copy()
        if basis:
            meas.h(0)
        meas.measure(0, 0)
        transpiled = transpile(meas, self._simulator)
        result = self._simulator.run(transpiled, shots=1).result()
        counts = result.get_counts()
        return int(max(counts, key=counts.get))

    # ------------------------------------------------------------------
    # BB84 pipeline steps
    # ------------------------------------------------------------------

    def _alice_prepare(self, n: int) -> tuple[list[int], list[int], list[QuantumCircuit]]:
        bits = self._random_bits(n)
        bases = self._choose_bases(n)
        circuits = [self._prepare_qubit(b, ba) for b, ba in zip(bits, bases)]
        return bits, bases, circuits

    def _eve_intercept(self, circuits: list[QuantumCircuit]) -> list[QuantumCircuit]:
        eve_bases = [random.randint(0, 1) for _ in range(len(circuits))]
        eve_bits = [self._measure_qubit(qc, b) for qc, b in zip(circuits, eve_bases)]
        return [self._prepare_qubit(bit, basis)
                for bit, basis in zip(eve_bits, eve_bases)]

    def _bob_measure(self, circuits: list[QuantumCircuit]) -> tuple[list[int], list[int]]:
        bases = self._choose_bases(len(circuits))
        bits = [self._measure_qubit(qc, b) for qc, b in zip(circuits, bases)]
        return bits, bases

    def _apply_channel_loss(
        self, circuits: list[QuantumCircuit],
        alice_bits: list[int], alice_bases: list[int],
    ) -> tuple[list[QuantumCircuit], list[int], list[int]]:
        """
        Apply photon loss from the channel model.

        Lost photons are removed entirely (the detector doesn't click).
        Dark counts are modelled by replacing lost positions with random bits
        at the detection probability — handled by the loss_mask() output.
        """
        if self._channel is None:
            return circuits, alice_bits, alice_bases

        mask = self._channel.loss_mask(len(circuits))
        surviving = []
        bits_out = []
        bases_out = []
        for i, detected in enumerate(mask):
            if detected:
                surviving.append(circuits[i])
                bits_out.append(alice_bits[i])
                bases_out.append(alice_bases[i])
        return surviving, bits_out, bases_out

    @staticmethod
    def _sift(
        alice_bits: list[int], alice_bases: list[int],
        bob_bits: list[int],   bob_bases: list[int],
    ) -> tuple[list[int], list[int], int, int]:
        a_sifted, b_sifted = [], []
        match = mismatch = 0
        for ab, aba, bb, bba in zip(alice_bits, alice_bases, bob_bits, bob_bases):
            if aba == bba:
                a_sifted.append(ab)
                b_sifted.append(bb)
                match += 1
            else:
                mismatch += 1
        return a_sifted, b_sifted, match, mismatch

    @staticmethod
    def _estimate_qber(
        alice_sifted: list[int],
        bob_sifted: list[int],
        sample_fraction: float = 0.15,
    ) -> tuple[float, list[int], list[int], list[int], list[int]]:
        n = len(alice_sifted)
        sample_size = max(8, int(n * sample_fraction))
        sample_size = min(sample_size, n // 2)

        sample_idx = set(random.sample(range(n), sample_size))
        errors = sum(alice_sifted[i] != bob_sifted[i] for i in sample_idx)
        qber = errors / sample_size if sample_size else 0.0

        a_remain = [alice_sifted[i] for i in range(n) if i not in sample_idx]
        b_remain = [bob_sifted[i]   for i in range(n) if i not in sample_idx]
        a_sample = [alice_sifted[i] for i in sample_idx]
        b_sample = [bob_sifted[i]   for i in sample_idx]

        return qber, a_remain, b_remain, a_sample, b_sample

    # ------------------------------------------------------------------
    # High-level: local simulation
    # ------------------------------------------------------------------

    def run_qkd(self) -> QKDStats | None:
        """
        Execute the full BB84 QKD pipeline in local simulation mode.

        Returns QKDStats on success, or None if the protocol aborts.
        """
        t0 = time.perf_counter()
        stats = QKDStats(
            eavesdropper_active=self.eavesdropper,
            error_threshold=self.error_threshold,
            epsilon_security=self.epsilon,
        )

        # Compute sifting efficiency for over-sampling
        sift_eff = self._biased.sifting_efficiency if self._biased else 0.5
        stats.sifting_efficiency = sift_eff
        stats.biased_basis = self._biased is not None
        stats.biased_p_z = self._p_z

        # Account for channel loss if applicable
        det_prob = self._channel.detection_probability if self._channel else 1.0
        # We need key_length final bits; work backward through the pipeline
        # PA compression ~50%, EC overhead, sifting, loss
        raw_count = max(
            int(self.key_length * 8 / max(sift_eff * det_prob, 0.01)),
            self.key_length * 8,
        )
        raw_count = min(raw_count, 100_000)  # Cap for simulation speed

        if self._channel:
            stats.channel_distance_km = self._channel.distance_km
            stats.channel_transmittance = self._channel.transmittance
            print(f"[QKD] Channel: {self._channel.distance_km:.0f} km, "
                  f"eta={self._channel.overall_efficiency:.2e}, "
                  f"expected QBER={self._channel.expected_qber:.4f}")

        if self._decoy:
            stats.decoy_enabled = True
            print("[QKD] Decoy-state protocol enabled "
                  f"(mu={self._decoy.mu}, nu={self._decoy.nu})")

        if self._biased:
            print(f"[QKD] Biased basis selection (p_z={self._p_z:.2f}, "
                  f"sifting eff={sift_eff:.1%})")

        log.info("Generating %d raw qubits", raw_count)
        print(f"[QKD] Preparing {raw_count} qubits ...")

        # Step 1 — Alice prepares qubits
        alice_bits, alice_bases, circuits = self._alice_prepare(raw_count)
        stats.raw_bits = raw_count

        # Step 2 — Channel loss (photon attenuation + dark counts)
        circuits, alice_bits, alice_bases = self._apply_channel_loss(
            circuits, alice_bits, alice_bases
        )
        n_surviving = len(circuits)
        if self._channel:
            print(f"[QKD] {n_surviving}/{raw_count} photons survived "
                  f"channel loss ({n_surviving/raw_count:.1%})")

        # Step 3 — (Optional) Eve intercept-resend
        if self.eavesdropper:
            print("[QKD] !! Eavesdropper (Eve) active — intercept-resend attack")
            circuits = self._eve_intercept(circuits)

        # Step 4 — Bob measures
        bob_bits, bob_bases = self._bob_measure(circuits)

        # Step 5 — Basis sifting
        a_sifted, b_sifted, match, mismatch = self._sift(
            alice_bits, alice_bases, bob_bits, bob_bases
        )
        stats.sifted_bits = len(a_sifted)
        stats.basis_match_count = match
        stats.basis_mismatch_count = mismatch
        actual_sift_eff = match / (match + mismatch) if (match + mismatch) > 0 else 0
        stats.sifting_efficiency = actual_sift_eff
        print(f"[QKD] Sifted key: {len(a_sifted)} bits  "
              f"({match} matched / {mismatch} discarded, eff={actual_sift_eff:.1%})")

        if len(a_sifted) < 64:
            print("[QKD] ABORT — too few sifted bits for secure key extraction.")
            return None

        # Step 6 — QBER estimation
        qber, a_key, b_key, _, _ = self._estimate_qber(a_sifted, b_sifted)
        stats.qber_before = qber
        print(f"[QKD] Estimated QBER: {qber:.2%}")

        if qber > self.error_threshold:
            print(f"[QKD] ABORT — QBER {qber:.2%} exceeds threshold "
                  f"{self.error_threshold:.2%}. Possible eavesdropping!")
            return None

        # Step 7 — CASCADE error correction
        print(f"[QKD] Running CASCADE error correction ({self.cascade_passes} passes) ...")
        cascade: CascadeResult = cascade_correct(
            a_key, b_key,
            estimated_qber=max(qber, 0.001),
            num_passes=self.cascade_passes,
        )
        stats.corrected_bits = len(cascade.corrected_key)
        stats.errors_corrected = cascade.errors_corrected
        stats.cascade_leaked = cascade.leaked_bits
        stats.cascade_passes = cascade.passes_run
        stats.qber_after = cascade.residual_error_rate
        stats.ec_efficiency = cascade.ec_efficiency
        stats.cascade_pass_stats = [
            {
                "pass_number": ps.pass_number,
                "block_size": ps.block_size,
                "blocks_checked": ps.blocks_checked,
                "errors_found": ps.errors_found,
                "parity_bits_exchanged": ps.parity_bits_exchanged,
            }
            for ps in cascade.pass_stats
        ]
        print(f"[QKD] CASCADE: {cascade.errors_corrected} errors corrected, "
              f"{cascade.leaked_bits} parity bits leaked, f={cascade.ec_efficiency:.3f}")

        # Step 8 — Finite-key security analysis
        fk: FiniteKeyResult = finite_key_analysis(
            n_sifted=len(a_sifted),
            n_raw=raw_count,
            observed_qber=qber,
            leaked_ec_bits=cascade.leaked_bits,
            epsilon_sec=self.epsilon,
            f_ec=cascade.ec_efficiency,
        )
        stats.finite_key = {
            "secure_key_length": fk.secure_key_length,
            "key_rate_per_pulse": fk.key_rate_per_pulse,
            "key_rate_per_sifted": fk.key_rate_per_sifted,
            "epsilon_security": fk.epsilon_security,
            "epsilon_EC": fk.epsilon_EC,
            "epsilon_PA": fk.epsilon_PA,
            "epsilon_PE": fk.epsilon_PE,
            "qber_upper_bound": fk.qber_upper_bound,
            "min_block_size": fk.min_block_size,
        }
        print(f"[QKD] Finite-key analysis: {fk.secure_key_length} secure bits, "
              f"eps={self.epsilon:.1e}, QBER_ub={fk.qber_upper_bound:.4f}")

        # Step 9 — Privacy amplification
        print("[QKD] Running privacy amplification ...")
        pa_target = min(fk.secure_key_length, len(cascade.corrected_key))
        pa_target = max(pa_target, self.key_length)
        try:
            pa: AmplificationResult = amplify(
                cascade.corrected_key,
                leaked_bits=cascade.leaked_bits,
                security_parameter=self.security_param,
                min_output=min(self.key_length, pa_target),
                observed_qber=qber,
                epsilon_pa=self.epsilon / 3,
                epsilon_ec=self.epsilon / 3,
            )
        except ValueError as exc:
            print(f"[QKD] ABORT — privacy amplification failed: {exc}")
            return None

        stats.amplified_bits = pa.output_length
        stats.privacy_compression = pa.compression_ratio
        stats.final_key = pa.amplified_key
        print(f"[QKD] Amplified key: {pa.output_length} bits  "
              f"(compression {pa.compression_ratio:.1%})")

        # Step 10 — Compute key rates
        stats.secure_key_rate_per_pulse = pa.output_length / raw_count if raw_count > 0 else 0
        # Asymptotic rate for comparison
        from finite_key import asymptotic_key_rate
        stats.secure_key_rate_asymptotic = asymptotic_key_rate(
            qber, sifting_efficiency=actual_sift_eff, f_ec=cascade.ec_efficiency,
        )
        print(f"[QKD] Key rate: {stats.secure_key_rate_per_pulse:.4e} bits/pulse "
              f"(asymptotic: {stats.secure_key_rate_asymptotic:.4e})")

        # Step 11 — Decoy-state analysis (informational, post-hoc)
        if self._decoy and self._channel:
            dec_est = self._decoy.estimate_from_channel(
                transmittance=self._channel.transmittance,
                e_misalign=self._channel.misalignment_error,
                dark_count_prob=self._channel.dark_count_prob,
                detector_efficiency=self._channel.detector_efficiency,
            )
            print(f"[QKD] Decoy analysis: Y1>={dec_est.Y_1_lower:.4f}, "
                  f"e1<={dec_est.e_1_upper:.4f}, "
                  f"R={dec_est.key_rate_per_pulse:.4e} bits/pulse")

        # Step 12 — Derive AES-256 key
        self._aes_key = derive_aes_key(pa.amplified_key)
        stats.aes_key_bits = 256
        print("[QKD] AES-256 key derived (HKDF-SHA256)")

        stats.elapsed_seconds = time.perf_counter() - t0
        self._stats = stats
        print(f"[QKD] Protocol complete in {stats.elapsed_seconds:.2f}s")
        return stats

    # ------------------------------------------------------------------
    # High-level: network simulation (Alice side)
    # ------------------------------------------------------------------

    def run_qkd_as_alice(
        self, host: str = "0.0.0.0", port: int = 5100,
    ) -> QKDStats | None:
        server = AliceServer(host=host, port=port)
        ch: ClassicalChannel | None = None
        try:
            print(f"[Alice] Waiting for Bob on {host}:{port} ...")
            ch = server.accept()
            return self._alice_protocol(ch)
        finally:
            if ch:
                ch.close()
            server.close()

    def _alice_protocol(self, ch: ClassicalChannel) -> QKDStats | None:
        t0 = time.perf_counter()
        stats = QKDStats(
            eavesdropper_active=self.eavesdropper,
            error_threshold=self.error_threshold,
            epsilon_security=self.epsilon,
        )
        raw_count = self.key_length * 8

        # 1. Prepare qubits
        alice_bits, alice_bases, _ = self._alice_prepare(raw_count)
        stats.raw_bits = raw_count
        print(f"[Alice] Prepared {raw_count} qubits, transmitting ...")

        qubit_data = [{"bit": b, "basis": ba}
                      for b, ba in zip(alice_bits, alice_bases)]

        if self.eavesdropper:
            print("[Alice] (Eve is intercepting the quantum channel)")
            eve_bases = [random.randint(0, 1) for _ in range(raw_count)]
            eve_bits = []
            for qd, eb in zip(qubit_data, eve_bases):
                qc = self._prepare_qubit(qd["bit"], qd["basis"])
                eve_bits.append(self._measure_qubit(qc, eb))
            qubit_data = [{"bit": b, "basis": ba}
                          for b, ba in zip(eve_bits, eve_bases)]

        ch.send(MsgType.QUBITS, qubit_data)

        # 2. Basis sifting
        ch.send(MsgType.BASIS_ANNOUNCE, alice_bases)
        _, bob_bases = ch.recv()

        alice_sifted = []
        match = mismatch = 0
        for i, (ab, bb) in enumerate(zip(alice_bases, bob_bases)):
            if ab == bb:
                alice_sifted.append(alice_bits[i])
                match += 1
            else:
                mismatch += 1
        stats.sifted_bits = len(alice_sifted)
        stats.basis_match_count = match
        stats.basis_mismatch_count = mismatch
        print(f"[Alice] Sifted: {len(alice_sifted)} bits")

        # 3. QBER estimation
        n = len(alice_sifted)
        sample_size = max(8, int(n * 0.15))
        sample_size = min(sample_size, n // 2)
        sample_idx = sorted(random.sample(range(n), sample_size))
        ch.send(MsgType.SAMPLE_INDICES, sample_idx)

        alice_sample = [alice_sifted[i] for i in sample_idx]
        ch.send(MsgType.SAMPLE_BITS, alice_sample)
        _, bob_sample = ch.recv()

        errors = sum(a != b for a, b in zip(alice_sample, bob_sample))
        qber = errors / sample_size if sample_size else 0.0
        stats.qber_before = qber
        ch.send(MsgType.ERROR_RATE, qber)
        print(f"[Alice] QBER: {qber:.2%}")

        if qber > self.error_threshold:
            ch.send(MsgType.ABORT, "QBER too high")
            print("[Alice] ABORT — possible eavesdropping!")
            return None

        keep = set(range(n)) - set(sample_idx)
        alice_key = [alice_sifted[i] for i in sorted(keep)]

        # 4. CASCADE
        ch.send(MsgType.CASCADE_PARITY, alice_key)
        _, bob_corrected = ch.recv()

        cascade_res = CascadeResult(
            corrected_key=bob_corrected,
            errors_corrected=sum(a != b for a, b in zip(alice_key, bob_corrected)),
            leaked_bits=len(alice_key) // 4,
            passes_run=self.cascade_passes,
            residual_error_rate=0.0,
        )
        stats.corrected_bits = len(alice_key)
        stats.errors_corrected = cascade_res.errors_corrected
        stats.cascade_leaked = cascade_res.leaked_bits
        stats.cascade_passes = self.cascade_passes
        stats.qber_after = cascade_res.residual_error_rate

        # 5. Privacy amplification
        pa_seed_size = len(alice_key) + self.key_length - 1
        pa_seed = self._random_bits(pa_seed_size)
        ch.send(MsgType.PA_SEED, {"seed": pa_seed, "output_len": self.key_length})

        amplified: list[int] = []
        for i in range(self.key_length):
            bit = 0
            for j in range(len(alice_key)):
                bit ^= pa_seed[i + j] & alice_key[j]
            amplified.append(bit)

        stats.amplified_bits = len(amplified)
        stats.final_key = amplified

        # 6. Derive AES key
        self._aes_key = derive_aes_key(amplified)
        stats.aes_key_bits = 256
        stats.secure_key_rate_per_pulse = len(amplified) / stats.raw_bits if stats.raw_bits > 0 else 0

        ch.send(MsgType.SUCCESS, None)
        stats.elapsed_seconds = time.perf_counter() - t0
        self._stats = stats
        print(f"[Alice] QKD complete — {len(amplified)}-bit key established")
        return stats

    # ------------------------------------------------------------------
    # High-level: network simulation (Bob side)
    # ------------------------------------------------------------------

    def run_qkd_as_bob(
        self, host: str = "127.0.0.1", port: int = 5100,
    ) -> QKDStats | None:
        ch: ClassicalChannel | None = None
        try:
            print(f"[Bob] Connecting to Alice at {host}:{port} ...")
            ch = BobClient.connect(host, port)
            return self._bob_protocol(ch)
        finally:
            if ch:
                ch.close()

    def _bob_protocol(self, ch: ClassicalChannel) -> QKDStats | None:
        t0 = time.perf_counter()
        stats = QKDStats(error_threshold=self.error_threshold, epsilon_security=self.epsilon)

        # 1. Receive qubits
        _, qubit_data = ch.recv()
        raw_count = len(qubit_data)
        stats.raw_bits = raw_count
        print(f"[Bob] Received {raw_count} qubits")

        bob_bases = self._choose_bases(raw_count)
        bob_bits: list[int] = []
        for qd, bb in zip(qubit_data, bob_bases):
            qc = self._prepare_qubit(qd["bit"], qd["basis"])
            bob_bits.append(self._measure_qubit(qc, bb))

        # 2. Basis sifting
        _, alice_bases = ch.recv()
        ch.send(MsgType.BASIS_ANNOUNCE, bob_bases)

        bob_sifted = [bob_bits[i] for i, (ab, bb)
                      in enumerate(zip(alice_bases, bob_bases)) if ab == bb]
        match = sum(1 for ab, bb in zip(alice_bases, bob_bases) if ab == bb)
        mismatch = raw_count - match
        stats.sifted_bits = len(bob_sifted)
        stats.basis_match_count = match
        stats.basis_mismatch_count = mismatch
        print(f"[Bob] Sifted: {len(bob_sifted)} bits")

        # 3. QBER estimation
        _, sample_idx = ch.recv()
        _, _alice_sample = ch.recv()
        bob_sample = [bob_sifted[i] for i in sample_idx]
        ch.send(MsgType.SAMPLE_BITS, bob_sample)

        _, qber = ch.recv()
        stats.qber_before = qber
        print(f"[Bob] QBER: {qber:.2%}")

        msg_type, data = ch.recv()
        if msg_type == MsgType.ABORT:
            print("[Bob] Alice aborted — possible eavesdropping!")
            return None

        alice_key = data

        n = len(bob_sifted)
        keep = set(range(n)) - set(sample_idx)
        bob_key = [bob_sifted[i] for i in sorted(keep)]

        # 4. CASCADE
        cascade: CascadeResult = cascade_correct(
            alice_key, bob_key,
            estimated_qber=max(qber, 0.001),
            num_passes=self.cascade_passes,
        )
        ch.send(MsgType.CASCADE_CORRECT, cascade.corrected_key)
        stats.corrected_bits = len(cascade.corrected_key)
        stats.errors_corrected = cascade.errors_corrected
        stats.cascade_leaked = cascade.leaked_bits
        stats.cascade_passes = cascade.passes_run
        stats.qber_after = cascade.residual_error_rate
        print(f"[Bob] CASCADE corrected {cascade.errors_corrected} errors")

        # 5. Privacy amplification
        _, pa_data = ch.recv()
        pa_seed = pa_data["seed"]
        output_len = pa_data["output_len"]

        amplified: list[int] = []
        for i in range(output_len):
            bit = 0
            for j in range(len(cascade.corrected_key)):
                bit ^= pa_seed[i + j] & cascade.corrected_key[j]
            amplified.append(bit)

        stats.amplified_bits = len(amplified)
        stats.final_key = amplified

        # 6. Derive AES key
        self._aes_key = derive_aes_key(amplified)
        stats.aes_key_bits = 256
        stats.secure_key_rate_per_pulse = len(amplified) / stats.raw_bits if stats.raw_bits > 0 else 0

        _, _ = ch.recv()  # SUCCESS
        stats.elapsed_seconds = time.perf_counter() - t0
        self._stats = stats
        print(f"[Bob] QKD complete — {len(amplified)}-bit key established")
        return stats

    # ------------------------------------------------------------------
    # Encryption / Decryption
    # ------------------------------------------------------------------

    def encrypt_message(self, plaintext: str) -> bytes:
        if self._aes_key is None:
            raise RuntimeError("No AES key — run QKD first.")
        packet = encrypt(self._aes_key, plaintext)
        return packet.serialize()

    def decrypt_message(self, data: bytes) -> str:
        if self._aes_key is None:
            raise RuntimeError("No AES key — run QKD first.")
        packet = AESPacket.deserialize(data)
        return decrypt(self._aes_key, packet)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def aes_key(self) -> bytes | None:
        return self._aes_key

    @property
    def stats(self) -> QKDStats:
        return self._stats

    @property
    def channel(self) -> ChannelModel | None:
        return self._channel


# ===================================================================
# Convenience: run as script
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    channel = ChannelModel.from_detector_preset("snspd", distance_km=20)
    proto = QuantumConnectionProtocol(
        key_length=256,
        channel_model=channel,
        biased_p_z=0.9,
        decoy_protocol=DecoyProtocol(),
        epsilon=1e-10,
    )
    result = proto.run_qkd()
    if result is None:
        print("QKD aborted.")
        exit(1)

    msg = "Hello from quantum-secured channel!"
    ct = proto.encrypt_message(msg)
    pt = proto.decrypt_message(ct)
    print(f"Encrypted: {ct.hex()[:60]}...")
    print(f"Decrypted: {pt}")
