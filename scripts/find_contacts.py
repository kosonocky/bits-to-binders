import numpy as np
import pandas as pd
from Bio.PDB import MMCIFParser
from itertools import combinations
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


AROMATIC = {"PHE", "TYR", "TRP", "HIS"}

RING_ATOMS = {
    "PHE": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "TYR": ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    "HIS": ("CG", "ND1", "CD2", "CE1", "NE2"),
    # simple choice: six membered ring of Trp
    "TRP": ("CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"),
}

def ring_centroid_and_normal(res):
    names = RING_ATOMS[res.get_resname()]
    coords = []
    for name in names:
        if name in res:
            coords.append(res[name].coord)
    if len(coords) < 3:
        return None
    coords = np.array(coords)
    centroid = coords.mean(axis=0)
    v1 = coords[1] - coords[0]
    v2 = coords[2] - coords[0]
    normal = np.cross(v1, v2)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-6:
        return None
    normal = normal / n_norm
    return centroid, normal

def find_pipi_contacts(model, d_max=7.0, angle_cutoff=30.0):
    aro_res = []
    for chain in model:
        for res in chain:
            if res.get_resname() in AROMATIC:
                info = ring_centroid_and_normal(res)
                if info is None:
                    continue
                centroid, normal = info
                # res.id is (hetflag, resseq, icode)
                _, resseq, icode = res.id
                aro_res.append((chain.id, res.get_resname(), resseq, icode, centroid, normal))

    within_chain = []
    between_chain = []

    for (ch1, name1, resi1, ic1, c1, n1), (ch2, name2, resi2, ic2, c2, n2) in combinations(aro_res, 2):
        d = np.linalg.norm(c1 - c2)
        if d > d_max:
            continue

        cos_theta = np.dot(n1, n2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(abs(cos_theta)))
        if angle > angle_cutoff:
            continue

        # same skip rule you used for disulfides
        if (ch1 in ["B", "C"] and ch2 in ["B", "C"]):
            continue

        pair = (ch1, name1, resi1,
                ch2, name2, resi2)

        if ch1 == ch2:
            within_chain.append(pair)
        else:
            between_chain.append(pair)

    return within_chain, between_chain

def find_disulfides_basic(model, threshold=3.0):
    cys_residues = [
        res for res in model.get_residues()
        if res.get_resname() == "CYS" and "SG" in res
    ]

    within_chain = []
    between_chain = []

    for res1, res2 in combinations(cys_residues, 2):
        d = res1["SG"] - res2["SG"]
        if d > threshold:
            continue

        chain1 = res1.get_parent().id
        chain2 = res2.get_parent().id

        pair = (
            chain1, res1.id[1],
            chain2, res2.id[1],
        )

        # dont include BB and CC disulfides; but do include AA
        if (chain1 in ["B", "C"] and chain2 in ["B", "C"]):
            continue

        if chain1 == chain2:
            within_chain.append(pair)
        else:
            between_chain.append(pair)

    return within_chain, between_chain


def process_one(global_id):
    """Worker for a single protein index. Returns one row dict or None."""
    prot_id = str(global_id).zfill(5)
    pdb_path = (
        f"../data/jakub/data/boltz/predictions/"
        f"binder{prot_id}_CD20_46-217/"
        f"binder{prot_id}_CD20_46-217_model_0.cif"
    )

    parser = MMCIFParser(QUIET=True)

    try:
        structure = parser.get_structure("prot", pdb_path)
    except Exception:
        # could not parse this file
        return None

    try:
        model = next(structure.get_models())
    except StopIteration:
        return None

    within_chain, between_chain = find_disulfides_basic(model)
    pipi_within, pipi_between = find_pipi_contacts(model)

    return {
        "global_id": global_id,
        "pdb_path": pdb_path,
        "disulfide_within_chain": within_chain,
        "disulfide_between_chain": between_chain,
        "pipi_within_chain": pipi_within,
        "pipi_between_chain": pipi_between,
        "num_disulfide_within_chain": len(within_chain),
        "num_disulfide_between_chain": len(between_chain),
        "num_pipi_within_chain": len(pipi_within),
        "num_pipi_between_chain": len(pipi_between),
    }


def main():
    ids = list(range(1, 12002))
    rows = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_one, i): i for i in ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing proteins"):
            row = fut.result()
            if row is not None:
                rows.append(row)

    disulf_df = pd.DataFrame(rows)
    disulf_df = disulf_df.sort_values("global_id").reset_index(drop=True)
    disulf_df.to_csv("../data/disulfide_and_pipi_contacts.csv", index=False)


if __name__ == "__main__":
    main()