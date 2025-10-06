#!/usr/bin/env python3
"""
Compute central B/C backbone axis (PCA) and angles to A-chain N/C-terminal local axes.

- Central B/C axis: 1st PCA component of backbone atoms (N, CA, C) from chains B & C,
  oriented from the B/C termini side ("bottom") toward the opposite side ("top").
- A-chain residue axis:
    * Internal i:  CA(i-1) -> CA(i+1)
    * N-terminus:  OUTWARD  CA(1) -> CA(2)  (vector used = CA(1) - CA(2))
    * C-terminus:  OUTWARD  CA(n) -> CA(n-1) (vector used = CA(n) - CA(n-1))
- Reports two angles (degrees, 0–180): A C-term vs oriented B/C axis, and A N-term vs oriented B/C axis.

Directory layout supported:
- root/
  ├─ binder08109_CD20_46-217/
  │   ├─ binder08109_CD20_46-217_model_0.cif
  │   └─ other files...
  └─ ...

By default, recursively finds *.cif (and *.pdb). You can restrict via flags.

Requires: numpy, biopython
Optional: tqdm  (for a nicer progress bar)
Install:  pip install numpy biopython tqdm
Usage:
  python axis_angle_bc_vs_a_termini.py /path/to/root --out results.csv --jobs 4
  python axis_angle_bc_vs_a_termini.py /root --only-cif --cif-pattern "*_model_0.cif"
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser

def find_atom(res, name: str):
    if name not in res:
        return None
    try:
        return np.asarray(res[name].get_coord(), dtype=float)
    except Exception:
        return None

def is_std_res(res) -> bool:
    # Skip HETATM/waters; standard residues have res.id[0] == ' '
    return res.id[0] == ' '

def get_chain_ordered_residues(structure, chain_id: str):
    """Return ordered standard residues for the chain in model 0 (or None)."""
    for model in structure:
        if chain_id not in model:
            return None
        chain = model[chain_id]
        return [res for res in chain if is_std_res(res)]
    return None

def get_chain_ca_list(structure, chain_id: str) -> Optional[List[np.ndarray]]:
    residues = get_chain_ordered_residues(structure, chain_id)
    if residues is None:
        return None
    cas = []
    for res in residues:
        ca = find_atom(res, 'CA')
        if ca is not None:
            cas.append(ca)
    return cas if cas else None

def residue_axis_from_ca_list(ca_list: List[np.ndarray], idx: int) -> Optional[np.ndarray]:
    """
    Local backbone tangent using CA positions.
    OUTWARD at termini:
      - N-term:  CA(1) - CA(2)
      - C-term:  CA(n) - CA(n-1)
    Internal: central difference CA(i+1) - CA(i-1)
    """
    n = len(ca_list)
    if n < 2:
        return None
    if idx <= 0:
        v = ca_list[0] - ca_list[1]      # outward N-term
    elif idx >= n - 1:
        v = ca_list[-1] - ca_list[-2]    # outward C-term
    else:
        v = ca_list[idx + 1] - ca_list[idx - 1]
    norm = np.linalg.norm(v)
    if not np.isfinite(norm) or norm == 0:
        return None
    return v / norm

def collect_chain_backbone_coords(structure, chain_ids: List[str]) -> np.ndarray:
    coords = []
    for model in structure:
        for cid in chain_ids:
            if cid not in model:
                continue
            chain = model[cid]
            for res in chain:
                if not is_std_res(res):
                    continue
                for atom_name in ('N', 'CA', 'C'):
                    a = find_atom(res, atom_name)
                    if a is not None:
                        coords.append(a)
        break  # first model only
    return np.array(coords, dtype=float) if coords else np.empty((0, 3), dtype=float)

def principal_axis(coords: np.ndarray) -> Optional[np.ndarray]:
    if coords.shape[0] < 3:
        return None
    centered = coords - coords.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    axis = vt[0]
    n = np.linalg.norm(axis)
    if not np.isfinite(n) or n == 0:
        return None
    return axis / n

def angle_degrees(u: np.ndarray, v: np.ndarray) -> float:
    """0–180° angle; no acute-forcing."""
    dot = float(np.clip(np.dot(u, v), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))

def parse_structure(struct_path: Path):
    """Auto-select parser based on extension; return Bio.PDB structure or raise."""
    stem = struct_path.stem
    if struct_path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    elif struct_path.suffix.lower() == ".pdb":
        parser = PDBParser(QUIET=True, PERMISSIVE=True)
    else:
        raise ValueError(f"Unsupported file type: {struct_path.suffix}")
    return parser.get_structure(stem, str(struct_path))

# ---- Helpers to orient the B/C axis bottom->top ----------------------------

def chain_term_ca_coords(structure, chain_id: str):
    """Return CA of first and last standard residues for a chain, if present."""
    residues = get_chain_ordered_residues(structure, chain_id)
    if not residues:
        return (None, None)
    first_ca = None
    for res in residues:
        ca = find_atom(res, 'CA')
        if ca is not None:
            first_ca = ca
            break
    last_ca = None
    for res in reversed(residues):
        ca = find_atom(res, 'CA')
        if ca is not None:
            last_ca = ca
            break
    return (first_ca, last_ca)

def bc_termini_centroid(structure, chains_bc: Tuple[str, str]):
    """Average the B/C termini CA positions to estimate the 'bottom' side."""
    pts = []
    for cid in chains_bc:
        first_ca, last_ca = chain_term_ca_coords(structure, cid)
        if first_ca is not None:
            pts.append(first_ca)
        if last_ca is not None:
            pts.append(last_ca)
    if not pts:
        return None
    return np.mean(np.vstack(pts), axis=0)

def orient_axis_from_bottom_to_top(axis: np.ndarray, coords: np.ndarray, bottom_point: np.ndarray):
    """
    Make axis point from the 'bottom' (termini centroid) toward the bulk ('top').
    Heuristic: ensure dot(axis, mean(coords) - bottom_point) > 0
    """
    mu = coords.mean(axis=0)
    if np.dot(axis, (mu - bottom_point)) < 0:
        return -axis
    return axis

# ---------------------------------------------------------------------------

def process_pdb(struct_path: Path,
                chain_a: str = 'A',
                chains_bc: Tuple[str, str] = ('B', 'C')) -> dict:
    out = {
        "file": str(struct_path),
        "n_backbone_BC": None,
        "angle_Cterm_deg": None,
        "angle_Nterm_deg": None,
        "status": "ok",
        "note": ""
    }
    try:
        structure = parse_structure(struct_path)
    except Exception as e:
        out["status"] = "parse_error"
        out["note"] = f"{e.__class__.__name__}: {e}"
        return out

    # Central B/C backbone axis (N, CA, C)
    bc_coords = collect_chain_backbone_coords(structure, list(chains_bc))
    out["n_backbone_BC"] = int(bc_coords.shape[0])
    bc_axis = principal_axis(bc_coords)
    if bc_axis is None:
        out["status"] = "no_bc_axis"
        out["note"] = "Insufficient B+C backbone coords for PCA"
        return out

    # Orient B/C axis from termini side ("bottom") to the opposite side ("top")
    bottom = bc_termini_centroid(structure, chains_bc)
    if bottom is not None:
        bc_axis = orient_axis_from_bottom_to_top(bc_axis, bc_coords, bottom)

    # A-chain Cα list and termini indices
    ca_list = get_chain_ca_list(structure, chain_a)
    if not ca_list or len(ca_list) < 2:
        out["status"] = "no_chainA_cas"
        out["note"] = f"Chain {chain_a} missing or has <2 CA atoms"
        return out

    n_idx = 0
    c_idx = len(ca_list) - 1

    # Local residue axes at A-chain termini (OUTWARD)
    axis_nterm = residue_axis_from_ca_list(ca_list, n_idx)
    axis_cterm = residue_axis_from_ca_list(ca_list, c_idx)
    if axis_cterm is None:
        out["status"] = "no_cterm_axis"
        out["note"] = "Could not compute C-term axis"
        return out
    if axis_nterm is None:
        out["status"] = "no_nterm_axis"
        out["note"] = "Could not compute N-term axis"
        return out

    out["angle_Cterm_deg"] = angle_degrees(axis_cterm, bc_axis)
    out["angle_Nterm_deg"] = angle_degrees(axis_nterm, bc_axis)
    return out

def iter_struct_files(root: Path,
                      recurse: bool = True,
                      only_cif: bool = False,
                      only_pdb: bool = False,
                      cif_pattern: Optional[str] = None) -> List[Path]:
    """
    Find structure files under root. Defaults to both *.cif and *.pdb recursively.
    If cif_pattern is set (e.g. "*_model_0.cif"), it's used for CIFs.
    """
    paths = []
    if recurse:
        if not only_pdb:
            if cif_pattern:
                paths += list(root.glob(f"**/{cif_pattern}"))
            else:
                paths += list(root.glob("**/*.cif"))
        if not only_cif:
            paths += list(root.glob("**/*.pdb"))
    else:
        if not only_pdb:
            if cif_pattern:
                paths += list(root.glob(cif_pattern))
            else:
                paths += list(root.glob("*.cif"))
        if not only_cif:
            paths += list(root.glob("*.pdb"))
    return sorted(set(paths))

# --- TOP-LEVEL WORKER (picklable) -------------------------------------------
def _worker_star(args):
    p, chain_a, chains_bc = args
    return process_pdb(p, chain_a=chain_a, chains_bc=chains_bc)
# ---------------------------------------------------------------------------

def main():
    import multiprocessing as mp
    from itertools import repeat

    # Try tqdm; fallback to None
    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    ap = argparse.ArgumentParser(description="Angles between A-chain termini axes and oriented central B/C backbone axis over many structures (mmCIF/PDB).")
    ap.add_argument("pdb_dir", type=Path, help="Root directory containing subfolders with .cif/.pdb files")
    ap.add_argument("--out", type=Path, default=Path("axis_angles_aterm_vs_bc.csv"), help="Output CSV path")
    ap.add_argument("--chainA", default="A", help="Chain ID for A (default A)")
    ap.add_argument("--chainB", default="B", help="Chain ID for B (default B)")
    ap.add_argument("--chainC", default="C", help="Chain ID for C (default C)")
    ap.add_argument("--no-recurse", action="store_true", help="Do not search subdirectories")
    ap.add_argument("--only-cif", action="store_true", help="Only search for .cif files")
    ap.add_argument("--only-pdb", action="store_true", help="Only search for .pdb files")
    ap.add_argument("--cif-pattern", default="*_model_0.cif", help="Glob for CIF files (default '*_model_0.cif'); ignored if empty")
    ap.add_argument("--jobs", "-j", type=int, default=1, help="Parallel workers (>=1). Default 1")
    ap.add_argument("--verbose", "-v", action="store_true", help="Print a line per file with its status")
    args = ap.parse_args()

    if args.only_cif and args.only_pdb:
        raise SystemExit("--only-cif and --only-pdb are mutually exclusive")

    cif_pat = args.cif_pattern if args.cif_pattern else None
    files = iter_struct_files(
        args.pdb_dir,
        recurse=not args.no_recurse,
        only_cif=args.only_cif,
        only_pdb=args.only_pdb,
        cif_pattern=cif_pat
    )
    total = len(files)
    if total == 0:
        print("No structure files found.")
        return

    print(f"Discovered {total} structure file(s).", flush=True)

    chains_bc = (args.chainB, args.chainC)
    results = []
    ok = 0
    failed = 0

    if args.jobs and args.jobs > 1:
        with mp.Pool(processes=args.jobs) as pool:
            iterable = zip(files, repeat(args.chainA), repeat(chains_bc))
            iterator = pool.imap_unordered(_worker_star, iterable)
            if tqdm is not None:
                for rec in tqdm(iterator, total=total, desc="Processing", unit="file"):
                    results.append(rec)
                    if rec["status"] == "ok":
                        ok += 1
                    else:
                        failed += 1
                    if args.verbose:
                        print(f"{rec['status']:>12}  {rec['file']}")
            else:
                processed = 0
                for rec in iterator:
                    results.append(rec)
                    processed += 1
                    if rec["status"] == "ok":
                        ok += 1
                    else:
                        failed += 1
                    if args.verbose:
                        print(f"{rec['status']:>12}  {rec['file']}")
                    if processed % 25 == 0:
                        print(f"Processed {processed}/{total} ... OK={ok} failed={failed}", flush=True)
    else:
        iterable = files
        if tqdm is not None:
            iterable = tqdm(files, total=total, desc="Processing", unit="file")
        for p in iterable:
            rec = process_pdb(p, chain_a=args.chainA, chains_bc=chains_bc)
            results.append(rec)
            if rec["status"] == "ok":
                ok += 1
            else:
                failed += 1
            if args.verbose:
                print(f"{rec['status']:>12}  {rec['file']}")

    # Write CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "n_backbone_BC", "angle_Cterm_deg", "angle_Nterm_deg", "status", "note"]
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Wrote {len(results)} rows to {args.out}")
    print(f"OK: {ok} | failed: {failed}", flush=True)

if __name__ == "__main__":
    main()