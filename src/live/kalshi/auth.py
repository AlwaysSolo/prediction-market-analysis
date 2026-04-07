from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

from src.live.kalshi.config import KalshiCredentials


@lru_cache(maxsize=8)
def _load_private_key(private_key_path: str) -> rsa.RSAPrivateKey:
    with Path(private_key_path).open("rb") as key_file:
        return serialization.load_pem_private_key(
            key_file.read(),
            password=None,
            backend=default_backend(),
        )


def sign_pss_text(private_key: rsa.RSAPrivateKey, text: str) -> str:
    signature = private_key.sign(
        text.encode("utf-8"),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return base64.b64encode(signature).decode("utf-8")


def build_auth_headers(
    credentials: KalshiCredentials,
    method: str,
    path: str,
    timestamp_ms: int,
) -> dict[str, str]:
    path_without_query = path.split("?", 1)[0]
    signing_string = f"{timestamp_ms}{method.upper()}{path_without_query}"
    private_key = _load_private_key(str(credentials.private_key_path))
    signature = sign_pss_text(private_key, signing_string)
    return {
        "KALSHI-ACCESS-KEY": credentials.api_key_id,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": str(timestamp_ms),
    }

