"""
AES-256-GCM Authenticated Encryption for QKD-derived Keys.

Replaces the one-time pad from the original protocol with industry-standard
AES-256 in Galois/Counter Mode (GCM).  GCM provides both confidentiality
and integrity (AEAD — Authenticated Encryption with Associated Data).

Key derivation uses HKDF-SHA256 to stretch/condition the raw QKD bit-string
into a cryptographically uniform 256-bit AES key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bits_to_bytes(bits: list[int]) -> bytes:
    """Pack a list of 0/1 ints into a bytes object (MSB first)."""
    # Pad to a multiple of 8
    padded = bits + [0] * ((8 - len(bits) % 8) % 8)
    out = bytearray()
    for i in range(0, len(padded), 8):
        byte = 0
        for b in padded[i:i + 8]:
            byte = (byte << 1) | b
        out.append(byte)
    return bytes(out)


def bytes_to_bits(data: bytes) -> list[int]:
    """Unpack bytes into a list of 0/1 ints (MSB first)."""
    bits: list[int] = []
    for byte in data:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def derive_aes_key(
    qkd_key_bits: list[int],
    info: bytes = b"bb84-qkd-aes256-gcm",
) -> bytes:
    """
    Derive a 256-bit AES key from QKD key bits using HKDF-SHA256.

    Args:
        qkd_key_bits: Raw key bits from the QKD protocol (>= 64 bits).
        info:         Context string bound into the derivation.

    Returns:
        32-byte (256-bit) AES key.
    """
    raw_bytes = bits_to_bytes(qkd_key_bits)
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,       # 256 bits
        salt=None,       # Random salt is ideal but not required
        info=info,
    )
    return hkdf.derive(raw_bytes)


# ---------------------------------------------------------------------------
# Encrypt / Decrypt
# ---------------------------------------------------------------------------

@dataclass
class AESPacket:
    """An encrypted message packet with its nonce."""
    nonce: bytes       # 12-byte GCM nonce
    ciphertext: bytes  # Ciphertext + 16-byte GCM auth tag

    def serialize(self) -> bytes:
        """Nonce ‖ ciphertext for transmission."""
        return self.nonce + self.ciphertext

    @classmethod
    def deserialize(cls, data: bytes) -> "AESPacket":
        return cls(nonce=data[:12], ciphertext=data[12:])


def encrypt(key: bytes, plaintext: str, aad: bytes | None = None) -> AESPacket:
    """
    Encrypt a plaintext string with AES-256-GCM.

    Args:
        key:       32-byte AES key (from derive_aes_key).
        plaintext: UTF-8 message to encrypt.
        aad:       Optional associated data (authenticated but not encrypted).

    Returns:
        AESPacket containing nonce and ciphertext.
    """
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)  # 96-bit nonce — unique per message
    ct = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), aad)
    return AESPacket(nonce=nonce, ciphertext=ct)


def decrypt(key: bytes, packet: AESPacket, aad: bytes | None = None) -> str:
    """
    Decrypt an AES-256-GCM packet.

    Raises cryptography.exceptions.InvalidTag if the ciphertext was
    tampered with or the wrong key is used.
    """
    aesgcm = AESGCM(key)
    plaintext = aesgcm.decrypt(packet.nonce, packet.ciphertext, aad)
    return plaintext.decode("utf-8")
