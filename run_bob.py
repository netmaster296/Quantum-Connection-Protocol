#!/usr/bin/env python3
"""
Run the BB84 QKD protocol as Bob (client).

Usage:
    python run_bob.py [--host 127.0.0.1] [--port 5100] [--key-length 256]
                      [--noise moderate]

Bob connects to Alice, receives qubits (simulated), and executes the full
QKD pipeline over TCP.  Once the shared AES key is established, Bob can
decrypt messages from Alice.
"""

import argparse
import logging
import sys

from quantumconnectionprotocol import QuantumConnectionProtocol
from visualization import show_dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="BB84 QKD — Bob (client)")
    parser.add_argument("--host", default="127.0.0.1", help="Alice's address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5100, help="TCP port (default: 5100)")
    parser.add_argument("--key-length", type=int, default=256, help="Desired key length in bits")
    parser.add_argument("--noise", default="moderate", help="Noise preset or type (default: moderate)")
    parser.add_argument("--biased", type=float, default=0.5, metavar="P_Z",
                        help="Biased Z-basis probability (0.5=standard)")
    parser.add_argument("--epsilon", type=float, default=1e-10, help="Security parameter")
    parser.add_argument("--visualize", action="store_true", help="Show dashboard after QKD")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
    )

    protocol = QuantumConnectionProtocol(
        key_length=args.key_length,
        noise_type=args.noise,
        biased_p_z=args.biased,
        epsilon=args.epsilon,
    )

    stats = protocol.run_qkd_as_bob(host=args.host, port=args.port)
    if stats is None:
        print("QKD aborted.")
        sys.exit(1)

    print(f"\nShared AES-256 key established. Ready to decrypt messages.")
    print(f"Key fingerprint: {protocol.aes_key.hex()[:16]}...")

    if args.visualize:
        show_dashboard(stats.as_dict(), save_path="qkd_bob_dashboard.png")


if __name__ == "__main__":
    main()
