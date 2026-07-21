"""Download the raw Amazon Reviews 2023 (Clothing/Shoes/Jewelry) source files.

This is a one-shot acquisition script, not part of the reusable pipeline
package surface — it fetches two large (multi-GB) gzip files from the
McAuley Lab's public host, verifies them with a streamed SHA-256 hash, and
records provenance (URL, size, hash, timestamp) in a JSON manifest so the
download is auditable and never silently re-hashed differently later.

Design notes:
- Resumable: downloads to a `.part` file and resumes with an HTTP Range
  request if interrupted, rather than restarting multi-GB transfers from
  scratch on every retry.
- Streamed hashing: SHA-256 is computed incrementally as bytes arrive (and
  incrementally over any pre-existing partial bytes on resume), so no
  second full-file read is needed after the download completes.
- Idempotent: re-running with the same config and without --force is a
  no-op if the manifest already has a verified entry for that file.

Usage:
    python -m cragb.data.download --config configs/data.yaml
    python -m cragb.data.download --only metadata --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cragb.utils.io import load_config, resolve_path

logger = logging.getLogger(__name__)

CHUNK_SIZE = 1024 * 1024  # 1 MiB
CONNECT_TIMEOUT_S = 10
READ_TIMEOUT_S = 30
MAX_RETRIES = 5


@dataclass(frozen=True)
class DownloadResult:
    url: str
    dest_path: str
    size_bytes: int
    sha256: str
    downloaded_at_utc: str


def _build_session() -> requests.Session:
    """A requests Session with retry/backoff for transient network errors."""
    session = requests.Session()
    retry = Retry(
        total=MAX_RETRIES,
        backoff_factor=2.0,  # 2s, 4s, 8s, 16s, 32s
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _remote_size(session: requests.Session, url: str) -> int | None:
    """HEAD the URL to get Content-Length, if the server reports one."""
    resp = session.head(url, timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S), allow_redirects=True)
    resp.raise_for_status()
    length = resp.headers.get("Content-Length")
    return int(length) if length is not None else None


def download_with_resume(url: str, dest: Path, session: requests.Session | None = None) -> DownloadResult:
    """Download `url` to `dest`, resuming a partial `.part` file if present.

    The SHA-256 hash is computed over the full file content (partial bytes
    already on disk are hashed first, then streamed new bytes are added),
    so the returned digest always covers the complete file regardless of
    how many resumes it took.

    Args:
        url: source URL to fetch.
        dest: final destination path. A sibling `<name>.part` file is used
            as the in-progress download target.
        session: an optional pre-built `requests.Session`; a retry-enabled
            one is created if not given.

    Returns:
        A `DownloadResult` with size and hash of the completed file.

    Raises:
        requests.HTTPError: on a non-2xx/206 response after retries.
        ValueError: if the server reports a total size and the final file
            size on disk doesn't match it (corruption/truncation guard).
    """
    session = session or _build_session()
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")

    remote_total = _remote_size(session, url)

    resume_from = part_path.stat().st_size if part_path.exists() else 0
    if remote_total is not None and resume_from >= remote_total:
        # Stale/complete partial from a previous run; start clean.
        resume_from = 0
        part_path.unlink(missing_ok=True)

    sha256 = hashlib.sha256()
    if resume_from:
        with part_path.open("rb") as f:
            for block in iter(lambda: f.read(CHUNK_SIZE), b""):
                sha256.update(block)
        logger.info("Resuming %s at byte %d of %s", dest.name, resume_from, remote_total or "unknown")

    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    with session.get(
        url,
        headers=headers,
        stream=True,
        timeout=(CONNECT_TIMEOUT_S, READ_TIMEOUT_S),
    ) as resp:
        if resume_from and resp.status_code != 206:
            # Server ignored our Range request; the byte stream we're about
            # to append would not align with what's already on disk.
            raise ValueError(
                f"Expected 206 Partial Content resuming {url}, got {resp.status_code}. "
                "Re-run with --force to restart the download from scratch."
            )
        resp.raise_for_status()
        # If the server applies Content-Encoding (e.g. gzip on top of the
        # payload), Content-Length reflects the wire size while
        # iter_content() yields decoded bytes — the two sizes legitimately
        # differ. Only trust the Content-Length size check when there's no
        # such transparent re-encoding in play.
        size_check_applies = "Content-Encoding" not in resp.headers

        mode = "ab" if resume_from else "wb"
        written = resume_from
        with part_path.open(mode) as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                sha256.update(chunk)
                written += len(chunk)

    if size_check_applies and remote_total is not None and written != remote_total:
        raise ValueError(
            f"Downloaded size {written} for {url} does not match reported "
            f"Content-Length {remote_total}; the file is likely truncated. "
            "Re-run to resume."
        )

    part_path.replace(dest)
    return DownloadResult(
        url=url,
        dest_path=str(dest),
        size_bytes=written,
        sha256=sha256.hexdigest(),
        downloaded_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_manifest(manifest_path: Path, manifest: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def acquire(config_path: str | Path, only: str = "all", force: bool = False) -> dict[str, DownloadResult]:
    """Download the configured raw file(s) and update the checksum manifest.

    Args:
        config_path: path to the data config (e.g. "configs/data.yaml").
        only: "reviews", "metadata", or "all".
        force: if True, delete any existing final/partial file first and
            redownload from scratch even if a manifest entry exists.

    Returns:
        Mapping of logical name ("reviews"/"metadata") to `DownloadResult`
        for whichever files were (re)downloaded or already verified.
    """
    cfg = load_config(config_path)
    manifest_path = resolve_path(cfg["paths"]["raw_checksums"])
    manifest = _load_manifest(manifest_path)

    targets = {
        "reviews": (cfg["source"]["reviews_url"], resolve_path(cfg["paths"]["reviews_raw"])),
        "metadata": (cfg["source"]["metadata_url"], resolve_path(cfg["paths"]["metadata_raw"])),
    }
    if only != "all":
        targets = {only: targets[only]}

    session = _build_session()
    results: dict[str, DownloadResult] = {}
    for name, (url, dest) in targets.items():
        if force:
            dest.unlink(missing_ok=True)
            dest.with_suffix(dest.suffix + ".part").unlink(missing_ok=True)
            manifest.pop(name, None)

        if not force and name in manifest and dest.exists():
            logger.info("Skipping %s: already downloaded and recorded in manifest (use --force to redo).", name)
            results[name] = DownloadResult(**manifest[name])
            continue

        logger.info("Downloading %s from %s", name, url)
        start = time.monotonic()
        result = download_with_resume(url, dest, session=session)
        elapsed = time.monotonic() - start
        logger.info(
            "Finished %s: %.1f MB in %.0fs (%.1f MB/s), sha256=%s",
            name,
            result.size_bytes / 1e6,
            elapsed,
            (result.size_bytes / 1e6) / max(elapsed, 1e-9),
            result.sha256[:12],
        )
        manifest[name] = asdict(result)
        _save_manifest(manifest_path, manifest)
        results[name] = result

    return results


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml", help="Path to data config YAML.")
    parser.add_argument(
        "--only",
        choices=["reviews", "metadata", "all"],
        default="all",
        help="Download only this file, or both (default).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if already present/recorded in the manifest.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        acquire(args.config, only=args.only, force=args.force)
    except (requests.HTTPError, ValueError) as exc:
        logger.error("Download failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
