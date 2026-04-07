"""
Socket-Based Classical Channel for BB84 QKD.

In real QKD the classical channel is an authenticated (but not necessarily
encrypted) TCP/IP link used for basis comparison, error correction, and
privacy amplification.  This module provides:

  - ClassicalChannel : length-prefixed JSON messaging over a TCP socket.
  - AliceServer      : listens for a Bob connection, then runs the protocol.
  - BobClient        : connects to Alice and runs the protocol.

The *quantum* channel is inherently simulated (Qiskit Aer), so qubit states
are serialised as lightweight dicts rather than physical photons.
"""

from __future__ import annotations

import json
import socket
import struct
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("qkd.network")


# ---------------------------------------------------------------------------
# Message types exchanged during the BB84 protocol
# ---------------------------------------------------------------------------
class MsgType:
    # Quantum phase (simulated)
    QUBITS          = "QUBITS"           # Alice -> Bob: encoded qubit params
    # Sifting
    BASIS_ANNOUNCE  = "BASIS_ANNOUNCE"    # Both: announce measurement bases
    SIFTED_ACK      = "SIFTED_ACK"       # Ack after sifting
    # Error estimation
    SAMPLE_INDICES  = "SAMPLE_INDICES"    # Agree on sample positions
    SAMPLE_BITS     = "SAMPLE_BITS"       # Reveal sample bit values
    ERROR_RATE      = "ERROR_RATE"        # Computed QBER
    # CASCADE error correction
    CASCADE_PARITY  = "CASCADE_PARITY"    # Parity exchange
    CASCADE_CORRECT = "CASCADE_CORRECT"   # Correction notification
    # Privacy amplification
    PA_SEED         = "PA_SEED"           # Toeplitz seed
    # Protocol control
    ABORT           = "ABORT"
    SUCCESS         = "SUCCESS"


# ---------------------------------------------------------------------------
# Low-level framed channel
# ---------------------------------------------------------------------------

class ClassicalChannel:
    """
    Length-prefixed JSON messaging over a TCP socket.

    Wire format:  [4-byte big-endian length][UTF-8 JSON payload]
    """

    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.sock.settimeout(60)  # 60 s per message

    # -- Send / Receive -----------------------------------------------------

    def send(self, msg_type: str, data: Any = None) -> None:
        """Send a typed message."""
        payload = json.dumps({"type": msg_type, "data": data}).encode("utf-8")
        header = struct.pack("!I", len(payload))
        self.sock.sendall(header + payload)
        log.debug("TX  %s (%d bytes)", msg_type, len(payload))

    def recv(self) -> tuple[str, Any]:
        """Receive a typed message.  Returns (msg_type, data)."""
        header = self._recv_exact(4)
        length = struct.unpack("!I", header)[0]
        payload = self._recv_exact(length)
        msg = json.loads(payload.decode("utf-8"))
        log.debug("RX  %s (%d bytes)", msg["type"], length)
        return msg["type"], msg["data"]

    # -- Helpers ------------------------------------------------------------

    def _recv_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Peer closed the connection.")
            buf.extend(chunk)
        return bytes(buf)

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()


# ---------------------------------------------------------------------------
# Server (Alice) and Client (Bob)
# ---------------------------------------------------------------------------

@dataclass
class AliceServer:
    """
    TCP server that waits for a single Bob connection.

    Usage:
        server = AliceServer(host="0.0.0.0", port=5100)
        channel = server.accept()   # blocks until Bob connects
        # ... run protocol over channel ...
        channel.close()
        server.close()
    """
    host: str = "0.0.0.0"
    port: int = 5100
    _server_sock: socket.socket = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_sock.bind((self.host, self.port))
        self._server_sock.listen(1)
        log.info("Alice listening on %s:%d", self.host, self.port)

    def accept(self) -> ClassicalChannel:
        """Block until Bob connects; return the channel."""
        conn, addr = self._server_sock.accept()
        log.info("Bob connected from %s:%d", *addr)
        return ClassicalChannel(conn)

    def close(self) -> None:
        self._server_sock.close()


@dataclass
class BobClient:
    """
    TCP client that connects to Alice.

    Usage:
        channel = BobClient.connect("192.168.1.10", 5100)
        # ... run protocol over channel ...
        channel.close()
    """
    @staticmethod
    def connect(host: str = "127.0.0.1", port: int = 5100) -> ClassicalChannel:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        log.info("Connected to Alice at %s:%d", host, port)
        return ClassicalChannel(sock)
