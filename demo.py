#!/usr/bin/env python3
"""
BB84 QKD — Full Demo with Protocol Dashboard & Key-Rate Analysis.

Two modes of operation:

  1. **Single run** (default):
     Execute the full QKD pipeline once and display the 9-panel protocol
     dashboard.  Supports channel model, biased basis, decoy states, and
     finite-key security analysis.

  2. **Analysis sweep** (--analysis):
     Compute key rate vs distance, QBER, Eve interception, and epsilon,
     then render the 4-panel analysis dashboard.  No Qiskit simulation
     needed — uses pure analytical formulas.

Usage examples:
    python demo.py                                # standard local run
    python demo.py --distance 50 --detector snspd # 50 km with SNSPD detectors
    python demo.py --biased 0.9 --decoy           # biased basis + decoy
    python demo.py --eve                          # eavesdropper detection
    python demo.py --analysis                     # key-rate analysis sweep
    python demo.py --analysis --save              # save plots to PNG
"""

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BB84 QKD full demo with analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Noise presets (for --noise):
  ideal, low, moderate, high, fiber_10km, fiber_50km, free_space

Detector presets (for --detector):
  ingaas_spad   InGaAs SPAD (eta=10%%, dark=1e-6)  — most common
  snspd         Superconducting nanowire (eta=93%%, dark=1e-8) — state of art
  si_spad       Silicon SPAD (eta=50%%, dark=1e-5) — visible wavelength
        """,
    )

    # Mode
    parser.add_argument("--analysis", action="store_true",
                        help="Run analytical key-rate sweeps (no Qiskit sim)")

    # Channel
    parser.add_argument("--distance", type=float, default=None,
                        help="Fibre distance in km (enables channel model)")
    parser.add_argument("--detector", default="ingaas_spad",
                        choices=["ingaas_spad", "snspd", "si_spad"],
                        help="Detector preset (default: ingaas_spad)")
    parser.add_argument("--noise", default=None,
                        help="Noise preset (alternative to --distance)")

    # Protocol
    parser.add_argument("--key-length", type=int, default=256,
                        help="Desired key length in bits (default: 256)")
    parser.add_argument("--biased", type=float, default=0.5, metavar="P_Z",
                        help="Biased Z-basis probability (0.5=standard, 0.9=biased)")
    parser.add_argument("--decoy", action="store_true",
                        help="Enable decoy-state protocol")
    parser.add_argument("--eve", action="store_true",
                        help="Simulate intercept-resend eavesdropper")
    parser.add_argument("--cascade-passes", type=int, default=4,
                        help="CASCADE passes (default: 4)")
    parser.add_argument("--epsilon", type=float, default=1e-10,
                        help="Finite-key security parameter (default: 1e-10)")

    # Output
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip all plots")
    parser.add_argument("--save", action="store_true",
                        help="Save dashboard PNGs to disk")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    if args.analysis:
        _run_analysis(args)
    else:
        _run_single(args)


# ===================================================================
# Single-run mode
# ===================================================================

def _run_single(args) -> None:
    from quantumconnectionprotocol import QuantumConnectionProtocol
    from noise_model import ChannelModel
    from decoy_state import DecoyProtocol
    from visualization import show_dashboard

    # Build channel model if distance is specified
    channel = None
    if args.distance is not None:
        channel = ChannelModel.from_detector_preset(
            args.detector, distance_km=args.distance
        )

    # Decoy protocol
    decoy = DecoyProtocol() if args.decoy else None

    # Noise preset (only used when no channel model)
    noise_type = args.noise or "moderate"

    # Banner
    print("=" * 66)
    print("  BB84 Quantum Key Distribution — Production Demo")
    print("=" * 66)
    print(f"  Key length       : {args.key_length} bits")
    if channel:
        print(f"  Channel          : {args.distance} km fibre, {args.detector}")
        print(f"  Transmittance    : {channel.transmittance:.2e}")
        print(f"  Expected QBER    : {channel.expected_qber:.4f}")
        print(f"  Detection prob.  : {channel.detection_probability:.4e}")
    else:
        print(f"  Noise model      : {noise_type}")
    print(f"  Basis selection  : {'biased (p_z=' + str(args.biased) + ')' if args.biased != 0.5 else 'standard (50/50)'}")
    print(f"  Decoy states     : {'YES' if args.decoy else 'no'}")
    print(f"  Eavesdropper     : {'YES' if args.eve else 'no'}")
    print(f"  Security (eps)   : {args.epsilon:.1e}")
    print(f"  CASCADE passes   : {args.cascade_passes}")
    print("=" * 66)
    print()

    # Build & run
    protocol = QuantumConnectionProtocol(
        key_length=args.key_length,
        noise_type=noise_type,
        channel_model=channel,
        eavesdropper=args.eve,
        cascade_passes=args.cascade_passes,
        biased_p_z=args.biased,
        decoy_protocol=decoy,
        epsilon=args.epsilon,
    )

    stats = protocol.run_qkd()
    if stats is None:
        print("\n[DEMO] Protocol aborted — no secure key established.")
        sys.exit(1)

    # Summary
    print()
    print("-" * 66)
    print("  QKD Pipeline Summary")
    print("-" * 66)
    print(f"  Raw qubits transmitted  : {stats.raw_bits}")
    if stats.channel_distance_km is not None:
        n_surviving = int(stats.raw_bits * (stats.channel_transmittance or 1))
        print(f"  Photons surviving loss  : ~{n_surviving}")
    print(f"  After basis sifting     : {stats.sifted_bits}  (eff={stats.sifting_efficiency:.1%})")
    print(f"  QBER (estimated)        : {stats.qber_before:.2%}")
    print(f"  CASCADE corrections     : {stats.errors_corrected}")
    print(f"  Parity bits leaked      : {stats.cascade_leaked}")
    print(f"  EC efficiency (f)       : {stats.ec_efficiency:.3f}")
    print(f"  Residual error rate     : {stats.qber_after:.4%}")
    print(f"  After privacy amp.      : {stats.amplified_bits} bits")
    print(f"  AES-256 key derived     : yes")
    print(f"  Key rate (finite-key)   : {stats.secure_key_rate_per_pulse:.4e} bits/pulse")
    print(f"  Key rate (asymptotic)   : {stats.secure_key_rate_asymptotic:.4e} bits/pulse")
    fk = stats.finite_key
    if fk:
        print(f"  Finite-key secure len   : {fk.get('secure_key_length', '?')} bits")
        print(f"  QBER upper bound        : {fk.get('qber_upper_bound', 0):.4f}")
        print(f"  Min block for pos. rate : {fk.get('min_block_size', '?'):,}")
    print(f"  Security parameter      : eps = {args.epsilon:.1e}")
    print(f"  Total elapsed time      : {stats.elapsed_seconds:.2f}s")
    print("-" * 66)

    # Encryption demo
    print()
    messages = [
        "Telemetry: Temp=23.4C, Humidity=45%, Battery=92%",
        "Command: ROTATE_ANTENNA az=145.2 el=32.7",
        "Alert: intrusion detected at perimeter sector 7",
    ]
    print("  AES-256-GCM Encryption Demo")
    print("-" * 66)
    for msg in messages:
        ct = protocol.encrypt_message(msg)
        pt = protocol.decrypt_message(ct)
        print(f"  Plain : {msg}")
        print(f"  Cipher: {ct.hex()[:64]}...")
        print(f"  Decrpt: {pt}")
        assert pt == msg, "Decryption mismatch!"
        print()

    print("[DEMO] All messages encrypted and decrypted successfully.")

    # Dashboard
    if not args.no_viz:
        print("\n[DEMO] Rendering protocol dashboard ...")
        save_path = "qkd_dashboard.png" if args.save else None
        show_dashboard(stats.as_dict(), save_path=save_path)

    print("\n[DEMO] Done.")


# ===================================================================
# Analysis sweep mode
# ===================================================================

def _run_analysis(args) -> None:
    from noise_model import ChannelModel
    from decoy_state import DecoyProtocol
    from key_rate_analysis import (
        sweep_distance, sweep_qber, sweep_eve_interception,
    )
    from finite_key import key_rate_vs_epsilon
    from visualization import show_analysis_dashboard

    print("=" * 66)
    print("  BB84 QKD — Key-Rate & Security Analysis")
    print("=" * 66)

    channel = ChannelModel.from_detector_preset(
        args.detector, distance_km=args.distance or 10
    )
    decoy = DecoyProtocol() if args.decoy else DecoyProtocol()

    # 1. Distance sweep (0–200 km)
    print("[Analysis] Computing key rate vs distance (0–200 km) ...")
    dist_result = sweep_distance(
        channel=channel,
        decoy=decoy,
        biased_p_z=args.biased if args.biased != 0.5 else 0.9,
        block_size=100_000,
        epsilon=args.epsilon,
    )

    # 2. QBER sweep (0–11%)
    print("[Analysis] Computing key rate vs QBER (0–11%) ...")
    qber_result = sweep_qber(
        block_size=100_000,
        epsilon=args.epsilon,
    )

    # 3. Eve interception sweep
    print("[Analysis] Computing Eve impact analysis ...")
    eve_result = sweep_eve_interception(
        channel=channel.at_distance(args.distance or 20),
        block_size=100_000,
        epsilon=args.epsilon,
    )

    # 4. Epsilon sensitivity
    print("[Analysis] Computing key rate vs epsilon ...")
    eps_data = key_rate_vs_epsilon(
        n_sifted=50_000,
        n_raw=100_000,
        observed_qber=channel.at_distance(args.distance or 20).expected_qber,
        leaked_ec_bits=5_000,
    )

    print("[Analysis] Rendering analysis dashboard ...")
    save_path = "qkd_analysis.png" if args.save else None
    show_analysis_dashboard(
        distance_sweep=dist_result,
        qber_sweep=qber_result,
        eve_sweep=eve_result,
        epsilon_data=eps_data,
        save_path=save_path,
    )

    print("\n[Analysis] Done.")


if __name__ == "__main__":
    main()
