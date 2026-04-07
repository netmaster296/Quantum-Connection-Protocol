#!/usr/bin/env python3
"""
Run the BB84 QKD protocol as Alice (server).

Usage:
    python run_alice.py [--host 0.0.0.0] [--port 5100] [--key-length 256]
                        [--noise moderate] [--eve]

Alice listens for a Bob connection, executes the full QKD pipeline over
TCP, then demonstrates AES-256-GCM encryption with the shared key.
"""

import argparse
import logging
import sys

from quantumconnectionprotocol import QuantumConnectionProtocol
from visualization import show_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="BB84 QKD — Alice (server)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5100, help="TCP port (default: 5100)")
    parser.add_argument("--key-length", type=int, default=256, help="Desired key length in bits")
    parser.add_argument("--noise", default="moderate", help="Noise preset or type (default: moderate)")
    parser.add_argument("--eve", action="store_true", help="Simulate eavesdropper")
    parser.add_argument("--biased", type=float, default=0.5, metavar="P_Z",
                        help="Biased Z-basis probability (0.5=standard)")
    parser.add_argument("--decoy", action="store_true", help="Enable decoy-state protocol")
    parser.add_argument("--epsilon", type=float, default=1e-10, help="Security parameter")
    parser.add_argument("--visualize", action="store_true", help="Show dashboard after QKD")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    from decoy_state import DecoyProtocol

    protocol = QuantumConnectionProtocol(
        key_length=args.key_length,
        noise_type=args.noise,
        eavesdropper=args.eve,
        biased_p_z=args.biased,
        decoy_protocol=DecoyProtocol() if args.decoy else None,
        epsilon=args.epsilon,
    )

    stats = protocol.run_qkd_as_alice(host=args.host, port=args.port)
    if stats is None:
        print("QKD aborted.")
        sys.exit(1)

    # Demo: encrypt a message
    message = "Quantum-secured telemetry: sensor_id=7, temp=22.1C, status=OK"
    ct = protocol.encrypt_message(message)
    print(f"\nEncrypted ({len(ct)} bytes): {ct.hex()[:80]}...")
    print(f"Decrypted: {protocol.decrypt_message(ct)}")

    if args.visualize:
        show_dashboard(stats.as_dict(), save_path="qkd_alice_dashboard.png")


if __name__ == "__main__":
    main()
