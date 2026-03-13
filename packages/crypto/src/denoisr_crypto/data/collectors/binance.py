"""Binance spot kline archive collection."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

from denoisr_crypto.types import FixedWindow, StorageLayout, iter_month_starts

log = logging.getLogger(__name__)
_BASE_URL = "https://data.binance.vision"


def build_monthly_kline_url(symbol: str, interval: str, month_start) -> str:
    month = f"{month_start.year:04d}-{month_start.month:02d}"
    return (
        f"{_BASE_URL}/data/spot/monthly/klines/"
        f"{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
    )


def build_monthly_checksum_url(symbol: str, interval: str, month_start) -> str:
    return f"{build_monthly_kline_url(symbol, interval, month_start)}.CHECKSUM"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_file(url: str, target: Path, *, timeout: int = 60) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as response, target.open("wb") as handle:  # noqa: S310
        shutil.copyfileobj(response, handle)


def _download_text(url: str, target: Path, *, timeout: int = 60) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=timeout) as response:  # noqa: S310
        text = response.read().decode("utf-8").strip()
    target.write_text(text)
    return text


def _load_checksum(path: Path) -> str:
    raw = path.read_text().strip()
    if not raw:
        raise ValueError(f"Empty checksum file: {path}")
    return raw.split()[0]


def collect_binance_klines(
    *,
    layout: StorageLayout,
    symbols: tuple[str, ...],
    interval: str,
    window: FixedWindow,
    skip_existing: bool = True,
    verify_checksums: bool = True,
) -> Path:
    """Download monthly Binance spot kline archives into the bronze layer."""
    collected_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    entries: list[dict[str, object]] = []

    for symbol in symbols:
        for month_start in iter_month_starts(window.start, window.end):
            url = build_monthly_kline_url(symbol, interval, month_start)
            checksum_url = build_monthly_checksum_url(symbol, interval, month_start)
            target = layout.bronze_archive_path(symbol, interval, month_start)
            checksum_target = layout.bronze_checksum_path(symbol, interval, month_start)
            metadata_target = layout.bronze_metadata_path(symbol, interval, month_start)

            if verify_checksums and (not checksum_target.exists() or not skip_existing):
                log.info("Downloading checksum %s -> %s", checksum_url, checksum_target)
                _download_text(checksum_url, checksum_target)
            expected_sha256 = (
                _load_checksum(checksum_target) if verify_checksums and checksum_target.exists() else None
            )
            actual_sha256 = _sha256_file(target) if target.exists() else None
            is_valid_existing = (
                target.exists()
                and skip_existing
                and (not verify_checksums or actual_sha256 == expected_sha256)
            )
            if is_valid_existing:
                status = "skipped"
            else:
                log.info("Downloading %s -> %s", url, target)
                _download_file(url, target)
                actual_sha256 = _sha256_file(target)
                if verify_checksums and expected_sha256 is not None and actual_sha256 != expected_sha256:
                    raise ValueError(
                        f"Checksum mismatch for {target}: expected {expected_sha256}, got {actual_sha256}"
                    )
                status = "downloaded"
            entry = {
                "symbol": symbol,
                "interval": interval,
                "month": month_start.isoformat(),
                "url": url,
                "checksum_url": checksum_url,
                "path": str(target),
                "checksum_path": str(checksum_target),
                "metadata_path": str(metadata_target),
                "status": status,
                "size_bytes": target.stat().st_size,
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
                "verified": (not verify_checksums) or (actual_sha256 == expected_sha256),
            }
            metadata_target.write_text(json.dumps(entry, indent=2, sort_keys=True))
            entries.append(entry)

    manifest = {
        "exchange": layout.exchange,
        "market": layout.market,
        "window": {"start": window.start.isoformat(), "end": window.end.isoformat()},
        "symbols": list(symbols),
        "interval": interval,
        "collected_at": collected_at,
        "entries": entries,
    }
    manifest_path = layout.manifest_path(f"collect_{collected_at}")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path
