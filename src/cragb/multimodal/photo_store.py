"""Content-addressed photo byte store for CRAGB (T6.1; PLAN.md §3 E7, M6.md T6.1).

`corpus_v1.parquet` holds `image_urls` — external `m.media-amazon.com` CDN
links (`data/processed/corpus_v1_datasheet.md`) — never image bytes. E7's
vision judge (T6.4) needs actual bytes to send to a multimodal model, so
this module is the one place that turns a URL into a verified, cached file
on disk, exactly once, with every failure recorded rather than silently
dropped.

Design notes, each tied to a specific failure this module must not have:

- **Content-addressed by URL, not by doc_id.** The same photo can be the
  first image of more than one review-image list in principle, and a
  doc_id-keyed cache would make "have I already fetched this URL" a query
  instead of a filename check. `photo_id(url)` is a stable, low-collision
  16-hex-char SHA-256 prefix, so the same URL never triggers a second HTTP
  request across runs, machines, or callers (T6.1 today, T6.3's control
  sampling and T6.4's judge calls later).
- **A 200 response is not proof of an image.** Amazon's CDN, like most
  CDNs, can return a 200 with an HTML error page, an empty body, or a
  format this pipeline can't send to a vision model. `fetch_photo` always
  decodes the downloaded bytes with Pillow before treating anything as
  `status="ok"` — a response that doesn't decode is `not_an_image`, not a
  silently-cached bad file.
- **A cap on bytes read, not bytes buffered after the fact.** `max_bytes`
  is enforced while streaming (`iter_content`), so a pathological huge
  response is abandoned mid-stream (`status="too_large"`) rather than
  fully downloaded into memory first and only then rejected.
- **The manifest is the coverage number.** `build_manifest` produces
  `n_attempted`/`n_ok`/`n_failed_by_status` alongside the per-URL rows —
  RQ4's honesty depends on this count being visible before any vision-judge
  quota is spent (M6.md's "print the funnel before spending quota" rule).

Usage:
    python -m cragb.multimodal.photo_store fetch --config configs/photo_store.yaml
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from PIL import Image, UnidentifiedImageError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from cragb.utils.io import REPO_ROOT, load_config, resolve_path

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT_S = 10

# Pillow's `Image.format` values this pipeline is willing to send to a
# vision model, mapped to the MIME type / file extension used to store and
# later re-serve them. Anything Pillow decodes but that isn't one of these
# (e.g. an animated GIF, a TIFF) is treated as `not_an_image` rather than
# silently accepted — a format the multimodal client (T6.2) was never
# tested against is a new failure mode, not a free win.
_FORMAT_TO_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_FORMAT_TO_EXT = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
_EXT_TO_MIME = {ext: mime for mime, ext in zip(_FORMAT_TO_MIME.values(), _FORMAT_TO_EXT.values())}

VALID_STATUSES = ("ok", "http_error", "timeout", "not_an_image", "too_large")


def photo_id(url: str) -> str:
    """Stable content-address for `url`.

    Args:
        url: the image URL (e.g. an `m.media-amazon.com` CDN link).

    Returns:
        The first 16 hex characters of `sha256(url)` — deterministic across
        runs and machines, short enough to be a friendly filename stem,
        long enough that a collision between two distinct URLs actually
        fetched by this project is not a practical concern.
    """
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class PhotoRecord:
    """One row of the fetch manifest: the outcome of attempting one URL.

    `path`/`bytes_len`/`mime`/`width`/`height` are only populated when
    `status == "ok"`; every other status leaves them `None` so a manifest
    reader can't mistake a failed attempt for a usable file.
    """

    photo_id: str
    url: str
    status: str  # one of VALID_STATUSES
    path: str | None = None
    bytes_len: int | None = None
    mime: str | None = None
    width: int | None = None
    height: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_session(max_retries: int) -> requests.Session:
    """A `requests.Session` with retry/backoff for transient network errors.

    Mirrors `cragb.data.download._build_session`: retries only 429/5xx (a
    real 404/403 from the CDN is a terminal `http_error`, not something to
    hammer on).
    """
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@dataclass
class PhotoStore:
    """Fetches, verifies, and disk-caches review photo bytes by URL.

    Args:
        photos_dir: directory for cached image files, repo-root-relative
            or absolute (resolved via `cragb.utils.io.resolve_path`).
        max_bytes: hard cap on bytes read per image; exceeding it while
            streaming aborts the fetch as `status="too_large"` without
            buffering the rest of the response.
        timeout_s: read timeout in seconds (connect timeout is fixed at
            `CONNECT_TIMEOUT_S`, matching `cragb.data.download`).
        max_retries: retry attempts on 429/5xx before giving up.
        request_delay_s: fixed pause between *live* (non-cache-hit)
            requests, so `fetch_many` doesn't hammer Amazon's CDN.
        allowed_mime: MIME types accepted as `status="ok"`; a successfully
            decoded image in any other format is `not_an_image`.
    """

    photos_dir: str = "data/photos"
    max_bytes: int = 5_000_000
    timeout_s: int = 20
    max_retries: int = 5
    request_delay_s: float = 0.2
    allowed_mime: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._session = _build_session(self.max_retries)

    # -- paths ---------------------------------------------------------

    def _dir(self) -> Path:
        d = resolve_path(self.photos_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _existing_path(self, pid: str) -> Path | None:
        """The cached file for `pid`, if one was already written, else None.

        Globs on the id stem rather than trying every extension, since the
        extension actually written depends on the image's decoded format,
        not something the caller chooses.
        """
        matches = sorted(self._dir().glob(f"{pid}.*"))
        return matches[0] if matches else None

    def _record_from_path(self, pid: str, url: str, path: Path) -> PhotoRecord:
        """Build a `status="ok"` record for an already-cached file.

        Re-decodes the cached bytes (cheap; these files are small) rather
        than trusting the extension alone, so a cache directory that was
        tampered with or corrupted on disk is caught here instead of being
        silently handed to the vision judge later.
        """
        payload = path.read_bytes()
        try:
            img = Image.open(io.BytesIO(payload))
            img.load()
        except (UnidentifiedImageError, OSError):
            return PhotoRecord(
                photo_id=pid,
                url=url,
                status="not_an_image",
                bytes_len=len(payload),
                error="cached file failed to decode",
            )
        fmt = (img.format or "").upper()
        return PhotoRecord(
            photo_id=pid,
            url=url,
            status="ok",
            path=str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path),
            bytes_len=len(payload),
            mime=_FORMAT_TO_MIME.get(fmt),
            width=img.width,
            height=img.height,
        )

    # -- fetch -----------------------------------------------------------

    def fetch_photo(self, url: str, *, force: bool = False) -> PhotoRecord:
        """Fetch and verify one photo, or return the cached record if present.

        Args:
            url: the image URL to fetch.
            force: redownload even if a cached file already exists for
                this URL's `photo_id`.

        Returns:
            A `PhotoRecord`. Only `status="ok"` implies a usable file on
            disk; every other status is a real, reportable outcome, not an
            exception — a caller processing many URLs (`fetch_many`) must
            not have one bad CDN link abort the whole batch.
        """
        pid = photo_id(url)
        if not force:
            existing = self._existing_path(pid)
            if existing is not None:
                return self._record_from_path(pid, url, existing)

        try:
            with self._session.get(
                url, stream=True, timeout=(CONNECT_TIMEOUT_S, self.timeout_s)
            ) as resp:
                if resp.status_code >= 400:
                    return PhotoRecord(
                        photo_id=pid, url=url, status="http_error", error=f"HTTP {resp.status_code}"
                    )

                data = bytearray()
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    data.extend(chunk)
                    if len(data) > self.max_bytes:
                        return PhotoRecord(
                            photo_id=pid, url=url, status="too_large", bytes_len=len(data)
                        )
        except requests.Timeout as exc:
            logger.warning("timeout fetching %s: %s", url, exc)
            return PhotoRecord(photo_id=pid, url=url, status="timeout", error=str(exc))
        except requests.RequestException as exc:
            logger.warning("request error fetching %s: %s", url, exc)
            return PhotoRecord(photo_id=pid, url=url, status="http_error", error=str(exc))

        payload = bytes(data)
        try:
            img = Image.open(io.BytesIO(payload))
            img.load()
        except (UnidentifiedImageError, OSError):
            return PhotoRecord(photo_id=pid, url=url, status="not_an_image", bytes_len=len(payload))

        fmt = (img.format or "").upper()
        mime = _FORMAT_TO_MIME.get(fmt)
        ext = _FORMAT_TO_EXT.get(fmt)
        if mime is None or ext is None or mime not in self.allowed_mime:
            return PhotoRecord(
                photo_id=pid,
                url=url,
                status="not_an_image",
                bytes_len=len(payload),
                error=f"unsupported image format: {fmt or 'unknown'}",
            )

        dest = self._dir() / f"{pid}.{ext}"
        dest.write_bytes(payload)
        logger.debug("cached %s -> %s (%d bytes)", url, dest, len(payload))
        return PhotoRecord(
            photo_id=pid,
            url=url,
            status="ok",
            path=str(dest.relative_to(REPO_ROOT)) if dest.is_relative_to(REPO_ROOT) else str(dest),
            bytes_len=len(payload),
            mime=mime,
            width=img.width,
            height=img.height,
        )

    def fetch_many(self, urls: list[str], *, force: bool = False) -> list[PhotoRecord]:
        """Fetch every URL in `urls`, deduplicated, with a polite delay between live calls.

        Args:
            urls: URLs to fetch. Duplicates are fetched (and delayed) once.
            force: passed through to every `fetch_photo` call.

        Returns:
            One `PhotoRecord` per unique URL, in first-seen order.
        """
        records: list[PhotoRecord] = []
        seen: set[str] = set()
        n_unique = len({u for u in urls})
        for url in urls:
            if url in seen:
                continue
            seen.add(url)

            cache_hit = not force and self._existing_path(photo_id(url)) is not None
            record = self.fetch_photo(url, force=force)
            records.append(record)

            if not cache_hit:
                time.sleep(self.request_delay_s)
            if len(records) % 20 == 0 or len(records) == n_unique:
                logger.info("fetched %d/%d", len(records), n_unique)
        return records

    # -- reading back for T6.2/T6.4/T6.6 -----------------------------------

    def photo_path(self, photo_id_: str) -> Path | None:
        """Resolved on-disk path to the cached file for `photo_id`, or `None` if
        not cached.

        For callers that need a human-openable file path rather than bytes
        (T6.6's spot-check worksheet: photos are referenced by path so a
        human can open them directly, never re-encoded or re-served).
        """
        return self._existing_path(photo_id_)

    def load_photo_bytes(self, photo_id_: str) -> bytes:
        """Read the cached bytes for a `photo_id` already fetched successfully.

        Raises:
            FileNotFoundError: no cached file exists for this id.
        """
        path = self._existing_path(photo_id_)
        if path is None:
            raise FileNotFoundError(f"No cached photo for photo_id={photo_id_!r} in {self._dir()}")
        return path.read_bytes()

    def to_data_part(self, photo_id_: str) -> dict[str, Any]:
        """Build the provider-neutral inline-image part T6.2's client expects.

        Raises:
            FileNotFoundError: no cached file exists for this id.
        """
        path = self._existing_path(photo_id_)
        if path is None:
            raise FileNotFoundError(f"No cached photo for photo_id={photo_id_!r} in {self._dir()}")
        data = path.read_bytes()
        mime = _EXT_TO_MIME.get(path.suffix.lstrip("."))
        if mime is None:
            img = Image.open(io.BytesIO(data))
            mime = _FORMAT_TO_MIME.get((img.format or "").upper())
        return {
            "type": "image",
            "photo_id": photo_id_,
            "mime": mime,
            "data_b64": base64.b64encode(data).decode("ascii"),
        }


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def build_manifest(records: list[PhotoRecord], *, config_path: str | None = None) -> dict[str, Any]:
    """Aggregate fetch records into the committed coverage manifest shape.

    Args:
        records: one `PhotoRecord` per attempted URL.
        config_path: the config used for this run, recorded for provenance.

    Returns:
        A dict with `created_at_utc`, `config_path`, `n_attempted`, `n_ok`,
        `n_failed_by_status` (every non-"ok" status and its count), and
        `entries` (the full per-URL rows).
    """
    by_status: dict[str, int] = {}
    for r in records:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path,
        "n_attempted": len(records),
        "n_ok": by_status.get("ok", 0),
        "n_failed_by_status": {status: count for status, count in sorted(by_status.items()) if status != "ok"},
        "entries": [r.to_dict() for r in records],
    }


def save_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")


def load_manifest(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Candidate URL collection (CRAGB-specific)
# --------------------------------------------------------------------------


def collect_candidate_urls(
    questions_in: str,
    pools_in: str,
    corpus_in: str,
    *,
    image_col: str = "has_image",
    image_urls_col: str = "image_urls",
) -> list[str]:
    """URLs worth fetching for the M6 multimodal pilot.

    Scope: the first image of every `has_image` doc that appears in the
    retrieval pool (`pools_v1.jsonl`, T2.6) of a question T2.3 flagged
    `image_target: true`. This is deliberately the *candidate* universe —
    T6.3's `surfaced_photo` still has to pick the specific doc the RAG-small
    pipeline would actually surface, and `control_photo` samples outside
    it — but every URL either of those needs bytes for was a retrieval
    candidate for an image-target question, so fetching this set up front
    means T6.3 never blocks on a live HTTP call.

    Args:
        questions_in: path to `cragb_questions_v1.jsonl` (has `image_target`).
        pools_in: path to `pools_v1.jsonl` (has `question_id`, `doc_ids`).
        corpus_in: path to `corpus_v1.parquet`.
        image_col: corpus boolean column flagging an image-bearing review.
        image_urls_col: corpus column holding each review's parsed URL list.

    Returns:
        A sorted, deduplicated list of URLs.
    """
    with resolve_path(questions_in).open("r", encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    image_target_ids = {q["id"] for q in questions if q.get("image_target")}

    candidate_doc_ids: set[str] = set()
    with resolve_path(pools_in).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            pool = json.loads(line)
            if pool["question_id"] in image_target_ids:
                candidate_doc_ids.update(pool["doc_ids"])

    corpus = pd.read_parquet(resolve_path(corpus_in))
    wanted = {int(doc_id) for doc_id in candidate_doc_ids}
    subset = corpus.loc[corpus.index.isin(wanted) & corpus[image_col].astype(bool)]

    urls: set[str] = set()
    for _, row in subset.iterrows():
        row_urls = row[image_urls_col]
        if row_urls is not None and len(row_urls) > 0:
            urls.add(str(row_urls[0]))
    return sorted(urls)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/photo_store.yaml", help="Path to photo-store config YAML.")
    parser.add_argument("--force", action="store_true", help="Refetch every URL even if already cached.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Fetch only the first N candidate URLs (smoke testing)."
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.add_argument("command", nargs="?", default="fetch", choices=["fetch"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    urls = collect_candidate_urls(
        cfg["paths"]["questions_in"],
        cfg["paths"]["pools_in"],
        cfg["paths"]["corpus_in"],
        image_col=cfg["corpus"]["image_col"],
        image_urls_col=cfg["corpus"]["image_urls_col"],
    )
    if args.limit is not None:
        urls = urls[: args.limit]
    logger.info("candidate URLs: %d", len(urls))

    store = PhotoStore(
        photos_dir=cfg["store"]["photos_dir"],
        max_bytes=cfg["store"]["max_bytes"],
        timeout_s=cfg["store"]["timeout_s"],
        max_retries=cfg["store"]["max_retries"],
        request_delay_s=cfg["store"]["request_delay_s"],
        allowed_mime=tuple(cfg["store"]["allowed_mime"]),
    )
    records = store.fetch_many(urls, force=args.force)

    manifest = build_manifest(records, config_path=args.config)
    save_manifest(cfg["paths"]["manifest_out"], manifest)

    logger.info(
        "attempted %d, ok %d (%.1f%%), failed by status %s",
        manifest["n_attempted"],
        manifest["n_ok"],
        100.0 * manifest["n_ok"] / max(manifest["n_attempted"], 1),
        manifest["n_failed_by_status"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
