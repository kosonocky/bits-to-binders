import os
import csv
import time
import torch
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# Config
DIRNAMES = [
    "autoregwithoutseq_proteinmpnn",
    "singleaascorewithoutseq_proteinmpnn",
    "autoregwithoutseq_solublempnn",
    "singleaascorewithoutseq_solublempnn",
    "autoregwithseq_proteinmpnn",
    "singleaascorewithseq_proteinmpnn",
    "autoregwithseq_solublempnn",
    "singleaascorewithseq_solublempnn",
]
ROOT = "../../LigandMPNN/outputs/bits_to_binders"
OUT_CSV = "../data/needs_recomputing/mpnn/consolidated_mpnn_scores_v2.csv"
CACHE_DIR = "../data/mpnn_score_cache"   # will store one tiny .csv per .pt
USE_PROCESSES = True                     # set False to try threads
MAX_WORKERS = min(8, os.cpu_count() or 4)  # start conservative for disk

AAS = ("A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y")
POS_KEYS_80 = tuple(f"A{i+1}" for i in range(80))

os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(dirname, fname):
    safe_dir = dirname.replace("/", "_")
    return os.path.join(CACHE_DIR, f"{safe_dir}__{fname}.csv")

def score_one(dirname, fpath, cutoff_len=80):
    """Fast path: pure-Python loop, no NumPy build. Returns dict."""

    # cache check
    fname = os.path.basename(fpath)
    cpath = cache_path(dirname, fname)
    if os.path.exists(cpath):
        try:
            with open(cpath, "r", newline="") as fh:
                rd = next(csv.DictReader(fh))
            return {"dirname": dirname, "filename": fname,
                    "total_score": None if rd["total_score"]=="" else float(rd["total_score"])}
        except Exception:
            pass  # fall back to recompute if cache is corrupt

    try:
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        data = torch.load(fpath, map_location="cpu", weights_only=False)

        seq = data["native_sequence"][:cutoff_len]
        L = len(seq)
        if L == 0:
            res = {"dirname": dirname, "filename": fname, "total_score": None}
        else:
            mop = data["mean_of_probs"]  # local var lookup is faster
            total = 0.0
            # tight Python loop, no dict allocations in the loop body
            for i, aa_idx in enumerate(seq):
                total += mop[POS_KEYS_80[i]][AAS[aa_idx]]
            res = {"dirname": dirname, "filename": fname, "total_score": total / L}

        # write tiny per-file cache
        try:
            with open(cpath, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=["dirname","filename","total_score"])
                wr.writeheader()
                wr.writerow(res)
        except Exception:
            pass

        return res

    except Exception:
        # return None score on any failure, still cache it to avoid repeated work
        res = {"dirname": dirname, "filename": fname, "total_score": None}
        try:
            with open(cpath, "w", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=["dirname","filename","total_score"])
                wr.writeheader()
                wr.writerow(res)
        except Exception:
            pass
        return res

def build_tasks():
    tasks = []
    for dirname in DIRNAMES:
        dpath = os.path.join(ROOT, dirname)
        if not os.path.isdir(dpath):
            continue
        with os.scandir(dpath) as it:
            for ent in it:
                if ent.is_file() and ent.name.endswith(".pt"):
                    tasks.append((dirname, ent.path))
    return tasks

def run_pool(tasks):
    rows = []
    Executor = ProcessPoolExecutor if USE_PROCESSES else ThreadPoolExecutor
    with Executor(max_workers=MAX_WORKERS) as ex:
        futs = [ex.submit(score_one, d, p) for d, p in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Scoring files", unit="file"):
            rows.append(fut.result())
    return rows

if __name__ == "__main__":
    # optional: set env vars before importing torch in a fresh process:
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"

    t0 = time.time()
    tasks = build_tasks()
    rows = run_pool(tasks)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"Done in {time.time()-t0:.1f}s, rows={len(df)}")