"""One-time model fetch.  Run once, with network; `run.py` then works offline.

The assignment forbids network access during the scored run, so every weight
must be on disk beforehand.  Each repository is resolved to an immutable commit
SHA and recorded in `models/lockfile.json`; `run.py` loads by that SHA, so a
later upstream force-push cannot silently change what gets scored.

    python prepare_models.py            # fetch everything
    python prepare_models.py --cpu      # only what the CPU pipeline needs
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

# Symlinks need Developer Mode on Windows; copies are correct everywhere.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODELS_DIR = pathlib.Path(__file__).parent / "models"
LOCKFILE = pathlib.Path(__file__).parent / "model_lock.json"

# Weights that never load together are still all fetched: the pipeline picks
# among them at run time and a missing repo would fail the offline run.
CPU_MODELS = [
    "BAAI/bge-reranker-v2-m3",
    "Alibaba-NLP/gte-multilingual-reranker-base",
]
# Only the reranker: the GPU pipeline uses the same lexical stage 1 as the CPU
# one, because dense retrieval measured as a regression here (see README 4.1).
# Fetching an unused 8 GB embedding model would just slow setup down.
GPU_MODELS = [
    "Qwen/Qwen3-Reranker-4B",
]

# Formats we never load.  Skipping them roughly halves the download, and it
# also dodges a Windows failure mode: `.eval_results/*.yaml` are MTEB metadata
# that hf_hub tries to symlink, which needs Developer Mode or admin rights and
# otherwise aborts the whole snapshot with WinError 1314.
IGNORE = [
    "*.onnx", "*.onnx_data", "onnx/*", "openvino/*", "*.msgpack", "*.h5",
    "*.tflite", ".eval_results/*", "*.eval_results/*", "imgs/*", "*.png",
]


def fetch(repos: list[str]) -> dict[str, str]:
    from huggingface_hub import snapshot_download

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    lock: dict[str, str] = {}
    if LOCKFILE.exists():
        lock = json.loads(LOCKFILE.read_text(encoding="utf-8"))

    for repo in repos:
        print(f"[fetch] {repo} ...", flush=True)
        try:
            path = snapshot_download(
                repo,
                revision=lock.get(repo),  # honour a pinned SHA when one exists
                cache_dir=str(MODELS_DIR),
                ignore_patterns=IGNORE,
                max_workers=8,
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[fail ] {repo}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        # snapshot path ends with the resolved commit SHA
        lock[repo] = os.path.basename(path.rstrip("/\\"))
        print(f"[ok   ] {repo} @ {lock[repo]}", flush=True)

    LOCKFILE.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    return lock


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="fetch only CPU-pipeline weights")
    ap.add_argument("--gpu", action="store_true", help="fetch only GPU-pipeline weights")
    ap.add_argument(
        "--no-proxy", action="store_true",
        help="ignore HTTP(S)_PROXY for this fetch.  A local proxy that is down "
             "surfaces as ProxyError/WinError 10048 rather than as a clear "
             "network failure, and direct access usually still works.",
    )
    args = ap.parse_args()

    if args.no_proxy:
        for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                    "http_proxy", "https_proxy", "all_proxy"):
            os.environ.pop(var, None)
        os.environ["NO_PROXY"] = "*"

    repos = CPU_MODELS if args.cpu else GPU_MODELS if args.gpu else CPU_MODELS + GPU_MODELS
    lock = fetch(repos)
    print(f"\n{len(lock)} repositories pinned in {LOCKFILE}")


if __name__ == "__main__":
    main()
