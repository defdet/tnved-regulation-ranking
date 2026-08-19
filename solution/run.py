"""Rank TN VED regulations against customs declarations.

    python run.py --data ./data --out ./out

Writes `<out>/predictions.csv` with columns declaration_id, rank,
regulation_id, score -- exactly ten unique regulations per declaration, ranks
1..10, scores finite and non-increasing with rank.

The pipeline is chosen automatically: GPU if a CUDA device is visible, CPU
otherwise.  Override with --pipeline.

    gpu   4B reranker at fp16 plus LLM-based extraction of numeric predicates.
          Needs ~14 GB of VRAM.
    cpu   fallback.  Lexical retrieval, a 568M multilingual cross-encoder, and
          a regex parser for numeric predicates.  Sized for 30 min / 8 GB with
          no accelerator.

The run is offline by construction: weights load from ./models pinned to the
commit SHAs in models/lockfile.json, with HF_HUB_OFFLINE forced on.  Fetch them
once beforehand with `python prepare_models.py`.
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="./data", help="directory holding the three input files")
    ap.add_argument("--out", default="./out", help="directory for predictions.csv")
    ap.add_argument("--pipeline", default="auto", choices=["auto", "cpu", "gpu", "lexical"],
                    help="auto (default): gpu if CUDA is visible, else cpu. "
                         "cpu | gpu | lexical force a specific path.")
    ap.add_argument("--n-candidates", type=int, default=None,
                    help="override how many stage-1 survivors are reranked")
    args = ap.parse_args()

    started = time.time()
    data_dir = pathlib.Path(args.data).resolve()
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in ("declarations.jsonl", "regulations.jsonl", "tnved_knowledge.txt"):
        if not (data_dir / name).is_file():
            print(f"error: {data_dir / name} not found", file=sys.stderr)
            return 2

    from src.pipeline import CPU_CONFIG, GPU_CONFIG, PipelineConfig, run_pipeline
    from src.records import load_declarations, load_regulations
    from src.validate import validate_predictions

    import torch

    cuda = torch.cuda.is_available()
    choice = "gpu" if (args.pipeline == "auto" and cuda) else args.pipeline
    if choice == "auto":
        choice = "cpu"

    if choice == "gpu":
        # Only an explicit --pipeline gpu can fail here; "auto" has already
        # fallen back to cpu above.  Asking for a device that is not there is
        # an error rather than a silent downgrade.
        if not cuda:
            print("error: --pipeline gpu was requested but no CUDA device is visible.\n"
                  "       Drop the flag to select the CPU path automatically.",
                  file=sys.stderr)
            return 2
        cfg = GPU_CONFIG
        print(f"[run] gpu pipeline on {torch.cuda.get_device_name(0)}")
    elif choice == "lexical":
        cfg = PipelineConfig(name="lexical")
    else:
        cfg = CPU_CONFIG
        why = "no CUDA device visible" if args.pipeline == "auto" else "forced by --pipeline cpu"
        print(f"[run] cpu pipeline ({why})")

    if args.n_candidates is not None:
        cfg.n_candidates = args.n_candidates

    regs = load_regulations(str(data_dir))
    decs = load_declarations(str(data_dir))
    print(f"[run] {len(decs)} declarations, {len(regs)} regulations")

    results = run_pipeline(cfg, regs, decs)

    out_path = out_dir / "predictions.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["declaration_id", "rank", "regulation_id", "score"])
        for d in decs:  # input order, so the file is stable across runs
            for rank, (reg_id, score) in enumerate(results[d.declaration_id], start=1):
                writer.writerow([d.declaration_id, rank, reg_id, f"{score:.6f}"])

    # Format is scored as pass/fail independently of ranking quality, so the
    # gate runs on every execution rather than only in tests.
    validate_predictions(
        str(out_path),
        {d.declaration_id for d in decs},
        {r.regulation_id for r in regs},
    )

    elapsed = time.time() - started
    print(f"[run] wrote {out_path} — format OK — {elapsed:.1f}s elapsed (budget 1800s)")
    if elapsed > 1800:
        print("warning: exceeded the 30 minute budget", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    raise SystemExit(main())
