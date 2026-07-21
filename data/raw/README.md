# Raw data — Amazon Reviews 2023 (Clothing, Shoes & Jewelry)

This directory holds the **unmodified source files** for the project. Files
here are never hand-edited; the only way they change is by re-running the
download script or bumping the source dataset version.

## Source

- **Dataset:** Amazon Reviews 2023 — https://amazon-reviews-2023.github.io/
- **Citation:** Hou, Y. et al. (2024). *Bridging Language and Items for
  Retrieval and Recommendation*. arXiv:2403.03952.
- **Category:** Clothing, Shoes and Jewelry
- **License:** see the dataset site; redistribution of the raw files is not
  covered by this repo's license — only the code that processes them is.

## Files

| File | Contents | Source URL |
|---|---|---|
| `reviews_clothing.jsonl.gz` | Raw review records (`rating`, `text`, `images`, `parent_asin`, ...) | `review_categories/Clothing_Shoes_and_Jewelry.jsonl.gz` |
| `meta_clothing.jsonl.gz` | Raw product metadata (`title`, `category`, `parent_asin`, ...) | `meta_categories/meta_Clothing_Shoes_and_Jewelry.jsonl.gz` |

Both files are **gzip-compressed JSON Lines** — one JSON object per line, no
top-level array. They are large (multi-gigabyte compressed) and are
git-ignored; see `checksums.json` (generated, not hand-written) for the
exact size, SHA-256 hash, and UTC download timestamp of whatever copy is
currently on disk, so anyone can verify their local copy matches what was
used to build `corpus_v1`.

## How to (re)download

```bash
# from the repo root, with the cragb conda env active
python -m cragb.data.download --config configs/data.yaml

# force a clean redownload of just one file
python -m cragb.data.download --only metadata --force
```

The script is resumable (safe to re-run after a network interruption) and
idempotent (skips files already recorded in `checksums.json` unless
`--force` is passed). Source URLs, destination paths, and the manifest
location are all configured in `configs/data.yaml` under `source:` and
`paths:` — nothing is hardcoded in the script itself.

## Verifying integrity

```bash
python -c "
import json, hashlib, pathlib
m = json.load(open('data/raw/checksums.json'))
for name, info in m.items():
    h = hashlib.sha256(pathlib.Path(info['dest_path']).read_bytes()).hexdigest()
    assert h == info['sha256'], f'{name} hash mismatch!'
    print(name, 'OK', info['sha256'][:12])
"
```

(For multi-GB files prefer streaming this rather than `read_bytes()` in one
call; the check above is illustrative — `download.py` itself hashes in a
streamed, memory-safe way.)
