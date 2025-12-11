import os
import sys
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from Bio.PDB import MMCIFParser, DSSP


def analyze_cif(cif_path, dssp_exe="mkdssp"):
    """
    Run DSSP on a CIF file and return a single string of
    DSSP secondary structure codes for Chain A in residue order.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("struct", cif_path)

    model = next(structure.get_models())
    dssp = DSSP(model, cif_path, dssp=dssp_exe)

    ss_codes = []

    # dssp keys are (chain_id, residue_id); sort to be safe
    for key in sorted(dssp.keys(), key=lambda k: (k[0], k[1][1], k[1][2])):
        chain_id = key[0]
        if chain_id != "A":
            continue

        dssp_entry = dssp[key]
        # index 2 = secondary structure (0=index, 1=AA, 2=SS)
        ss_code = dssp_entry[2]
        ss_codes.append(ss_code)

    dssp_string = "".join(ss_codes)
    return dssp_string


def _analyze_cif_wrapper(args):
    """
    Helper for parallel execution.
    Returns (global_id, cif_path, dssp_string, error_message_or_None)
    """
    global_id, cif_path, dssp_exe = args
    try:
        dssp_string = analyze_cif(cif_path, dssp_exe=dssp_exe)
        return global_id, cif_path, dssp_string, None
    except Exception as e:
        return global_id, cif_path, None, str(e)


def process_files(out_csv, global_ids, cif_paths, dssp_exe="mkdssp", n_workers=1):
    """
    Take lists of global_ids and .cif paths and write a CSV with:
      - global_id
      - dssp: DSSP code string for Chain A

    If n_workers > 1, process files in parallel.
    """
    records = []

    if n_workers is None or n_workers <= 1:
        # Original sequential version
        for global_id, cif_path in zip(global_ids, cif_paths):
            try:
                dssp_string = analyze_cif(cif_path, dssp_exe=dssp_exe)
            except Exception as e:
                print(f"Error for global_id {global_id} ({cif_path}): {e}", file=sys.stderr)
                dssp_string = None

            records.append(
                {
                    "global_id": global_id,
                    "dssp": dssp_string,
                }
            )
    else:
        # Parallel version
        tasks = [(global_id, cif_path, dssp_exe) for global_id, cif_path in zip(global_ids, cif_paths)]
        # Tune chunksize if needed
        chunksize = 20

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            for global_id, cif_path, dssp_string, error in executor.map(
                _analyze_cif_wrapper, tasks, chunksize=chunksize
            ):
                if error is not None:
                    print(f"Error for global_id {global_id} ({cif_path}): {error}", file=sys.stderr)

                records.append(
                    {
                        "global_id": global_id,
                        "dssp": dssp_string,
                    }
                )

    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    out_csv = "12k_dssp.csv"

    global_ids = []
    cif_paths = []

    for global_id in range(1, 12002):
        global_id_str = str(global_id).zfill(5)
        cif_paths.append(
            f"../data/jakub/data/boltz/predictions/"
            f"binder{global_id_str}_CD20_46-217/"
            f"binder{global_id_str}_CD20_46-217_model_0.cif"
        )
        global_ids.append(global_id)

    # Set n_workers to the number of processes you want, for example:
    process_files(out_csv, global_ids, cif_paths, n_workers=30)