import math
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


# -----------------------------
# config
# -----------------------------

METRICS_PATH = "../data/12k_all_metrics.csv"
RESULTS_PATH = "../data/12k_all_results.csv"
N_JOBS = -1  # for joblib Parallel
TOP_N_FOR_COMBOS_DETECTED = 30  # only for leah_12k_detected combos
MAX_COMBO_SIZE = 3


# -----------------------------
# load data
# -----------------------------

metrics_df = pd.read_csv(METRICS_PATH)
metrics_df["closest_ab_pident"] = metrics_df["closest_ab_pident"].fillna(0)
metrics_df["ubiquitin_pident"] = metrics_df["ubiquitin_pident"].fillna(0)

results_df = pd.read_csv(RESULTS_PATH)
results_df["leah_12k_Significant"] = results_df["leah_12k_Significant"].fillna(False)


# -----------------------------
REQUIRED_FEATURES = [
    "dna_sequence_entropy",
    "aa_sequence_entropy",          # "aa seq entropy"
    "seq_kl_vs_human",
    "longest_dup_substr_len",
    "total_duplicated_residues",
    "gravy_score",
    "num_cysteines",
    "closest_ab_pident",
    "msa_depth",
    "num_pipi_within_chain",
    "num_disulfide_within_chain",
    "longest_dup_occurrences",
    "dna_gc_content",
    "seq_charge",
    "ratio_K",
    "ratio_E",
    "ratio_A",
    "ratio_KE",
    "ratio_AKE",
    "dssp_alpha_ratio",
    "dssp_beta_ratio",
    "dssp_other_ratio",
    "dssp_ke_alpha_ratio",
    "dssp_kae_alpha_ratio",
]

# numeric feature list (exclude global_id)
feature_cols = [
    c
    for c in metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    if c != "global_id"
]

# -----------------------------
# helper: build X, y for binary target
# -----------------------------

def build_binary_data(feat_subset, target_col):
    X_all = metrics_df[list(feat_subset)]
    y_all = results_df.loc[X_all.index, target_col]

    valid = X_all.notna().all(axis=1) & y_all.notna()
    X = X_all.loc[valid]
    y = y_all.loc[valid].astype(int)

    return X, y


def binary_auc_for_features(feat_subset, target_col, test_size=0.1, random_state=838975):
    X, y = build_binary_data(feat_subset, target_col)

    if len(np.unique(y)) < 2:
        return np.nan

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    )

    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test,  y_proba)
    return roc_auc


# -----------------------------
# helper: multiclass AUC for fourclass
# -----------------------------

le = LabelEncoder()
le.fit(results_df["fourclass"].dropna())


def build_multiclass_data(feat_subset, target_col="fourclass"):
    X_all = metrics_df[list(feat_subset)]
    y_all = results_df.loc[X_all.index, target_col]

    valid = X_all.notna().all(axis=1) & y_all.notna()
    X = X_all.loc[valid]
    y = y_all.loc[valid]

    y_enc = le.transform(y)
    return X, y_enc


def multiclass_auc_for_features(
    feat_subset, test_size=0.1, random_state=838975, target_col="fourclass"
):
    X, y_enc = build_multiclass_data(feat_subset, target_col=target_col)

    classes = np.unique(y_enc)
    if len(classes) < 2:
        n_classes = len(le.classes_)
        return np.nan, [np.nan] * n_classes

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=test_size,
        random_state=random_state,
        stratify=y_enc,
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    multi_class="multinomial",
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)

    y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))

    micro_auc = roc_auc_score(
        y_test_bin,
        y_proba,
        average="micro",
        multi_class="ovr",
    )

    per_class_auc = []
    for k in range(y_test_bin.shape[1]):
        per_class_auc.append(roc_auc_score(y_test_bin[:, k], y_proba[:, k]))

    return micro_auc, per_class_auc


# -----------------------------
# generic scanning helpers (parallel)
# -----------------------------

def scan_binary_target(
    target_col,
    single_auc_csv,
    combo_auc_csv,
    top_n_for_combos=None,
    test_size=0.1,
    random_state=838975,
    n_jobs=N_JOBS,
):
    # univariate
    def _single_auc(feat):
        auc = binary_auc_for_features([feat], target_col, test_size, random_state)
        return feat, auc

    results = Parallel(n_jobs=n_jobs)(
        delayed(_single_auc)(feat) for feat in feature_cols
    )

    single_rows = [
        {"feature": feat, f"roc_auc_{target_col}": auc} for feat, auc in results
    ]

    single_auc_df = pd.DataFrame(single_rows).sort_values(
        by=f"roc_auc_{target_col}", ascending=False
    ).reset_index(drop=True)

    single_auc_df.to_csv(single_auc_csv, index=False)
    single_auc_df = single_auc_df[single_auc_df["feature"] != "global_id"]

    if top_n_for_combos is None:
        base_features = single_auc_df["feature"].tolist()
    else:
        base_features = single_auc_df["feature"].head(top_n_for_combos).tolist()

    # ensure required features are also included (if they exist in metrics_df)
    required_present = [f for f in REQUIRED_FEATURES if f in feature_cols]

    top_features = []
    for f in required_present + base_features:
        if f not in top_features:
            top_features.append(f)


    # combinations up to MAX_COMBO_SIZE
    combo_rows = []
    max_k = min(MAX_COMBO_SIZE, len(top_features))

    for k in range(1, max_k + 1):
        feat_subsets = list(combinations(top_features, k))

        def _combo_auc(subset):
            subset = list(subset)
            auc = binary_auc_for_features(subset, target_col, test_size, random_state)
            return subset, auc

        combo_results = Parallel(n_jobs=n_jobs)(
            delayed(_combo_auc)(subset) for subset in feat_subsets
        )

        for subset, auc in combo_results:
            combo_rows.append(
                {
                    "features": tuple(subset),
                    "n_features": len(subset),
                    f"roc_auc_{target_col}": auc,
                }
            )

    combo_df = pd.DataFrame(combo_rows).sort_values(
        by=f"roc_auc_{target_col}", ascending=False
    ).reset_index(drop=True)

    combo_df.to_csv(combo_auc_csv, index=False)


def scan_fourclass(
    single_auc_csv,
    combo_auc_csv,
    top_n_for_combos=None,
    test_size=0.1,
    random_state=838975,
    n_jobs=N_JOBS,
):
    # univariate
    def _single(feat):
        mc_auc_micro, mc_auc_per_class = multiclass_auc_for_features(
            [feat], test_size=test_size, random_state=random_state
        )
        bin_auc = binary_auc_for_features(
            [feat], target_col="leah_12k_detected", test_size=test_size, random_state=random_state
        )
        return feat, mc_auc_micro, mc_auc_per_class, bin_auc

    results = Parallel(n_jobs=n_jobs)(
        delayed(_single)(feat) for feat in feature_cols
    )

    rows = []
    for feat, mc_auc_micro, mc_auc_per_class, bin_auc in results:
        row = {
            "feature": feat,
            "n_features": 1,
            "roc_auc_4class_micro": mc_auc_micro,
            "roc_auc_binary_detected": bin_auc,
        }
        for cls_name, auc_val in zip(le.classes_, mc_auc_per_class):
            col_name = f"roc_auc_4class_{cls_name}"
            row[col_name] = auc_val
        rows.append(row)

    single_auc_df = pd.DataFrame(rows).sort_values(
        by="roc_auc_4class_micro", ascending=False
    ).reset_index(drop=True)

    single_auc_df.to_csv(single_auc_csv, index=False)

    if top_n_for_combos is None:
        top_features = single_auc_df["feature"].tolist()
    else:
        top_features = single_auc_df["feature"].head(top_n_for_combos).tolist()

    # combinations up to MAX_COMBO_SIZE
    combo_rows = []
    max_k = min(MAX_COMBO_SIZE, len(top_features))

    for k in range(1, max_k + 1):
        feat_subsets = list(combinations(top_features, k))

        def _combo(subset):
            subset = list(subset)
            mc_auc_micro, mc_auc_per_class = multiclass_auc_for_features(
                subset, test_size=test_size, random_state=random_state
            )
            bin_auc = binary_auc_for_features(
                subset,
                target_col="leah_12k_detected",
                test_size=test_size,
                random_state=random_state,
            )
            row = {
                "features": tuple(subset),
                "n_features": len(subset),
                "roc_auc_4class_micro": mc_auc_micro,
                "roc_auc_binary_detected": bin_auc,
            }
            for cls_name, auc_val in zip(le.classes_, mc_auc_per_class):
                col_name = f"roc_auc_4class_{cls_name}"
                row[col_name] = auc_val
            return row

        combo_results = Parallel(n_jobs=n_jobs)(
            delayed(_combo)(subset) for subset in feat_subsets
        )
        combo_rows.extend(combo_results)

    combo_df = pd.DataFrame(combo_rows).sort_values(
        by="roc_auc_4class_micro", ascending=False
    ).reset_index(drop=True)

    combo_df.to_csv(combo_auc_csv, index=False)


# -----------------------------
# main
# -----------------------------

def main():
    # 1) Twist DNA detected
    scan_binary_target(
        target_col="leah_12k_twist_dna_detected",
        single_auc_csv="../results/analysis/feature_single_aucs_leah_12k_twist_dna_detected.csv",
        combo_auc_csv="../results/analysis/feature_combination_aucs_leah_12k_twist_dna_detected.csv",
        top_n_for_combos=30,
    )

    # 2) 4 class outcome + detected
    scan_fourclass(
        single_auc_csv="../results/analysis/feature_single_aucs_fourclass_leah_12k.csv",
        combo_auc_csv="../results/analysis/feature_combination_aucs_fourclass_leah_12k.csv",
        top_n_for_combos=30,
    )

    # 3) Binary leah_12k_detected with top N features for combos
    scan_binary_target(
        target_col="leah_12k_detected",
        single_auc_csv="../results/analysis/feature_single_aucs_leah_12k_detected.csv",
        combo_auc_csv="../results/analysis/feature_combination_aucs_leah_12k_detected.csv",
        top_n_for_combos=30,
    )


if __name__ == "__main__":
    main()