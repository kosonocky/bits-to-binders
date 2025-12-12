import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


# -----------------------------
# config
# -----------------------------

METRICS_PATH = "../data/12k_all_metrics.csv"
RESULTS_PATH = "../data/12k_all_results.csv"

FEATURE_COMBO = [
    # "dna_gc_content",
    # "aa_sequence_entropy",
    "dna_sequence_entropy",
    "dssp_ke_alpha_ratio",
    # "dssp_beta_ratio",
    # "dssp_other_ratio",
    "seq_kl_vs_human",
    "boltz_rosetta_complex_normalized",
    # "gravy_score",
    # "boltz_sap_score",
    # "seq_charge",
    # "num_disulfide_within_chain",
    # "num_cysteines",
    # "esm2_full_LL_alone",
    # "boltz_rosetta_A_BC_iptm",
]

USE_CV = True
N_SPLITS = 10

RANDOM_STATE = 838975

# RFC settings (simple)
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 1
RF_N_JOBS = -1


# -----------------------------
# load data
# -----------------------------

metrics_df = pd.read_csv(METRICS_PATH)
metrics_df["closest_ab_pident"] = metrics_df["closest_ab_pident"].fillna(0)
metrics_df["ubiquitin_pident"] = metrics_df["ubiquitin_pident"].fillna(0)

results_df = pd.read_csv(RESULTS_PATH)
results_df["leah_12k_Significant"] = results_df["leah_12k_Significant"].fillna(False)


# -----------------------------
# label encoder for fourclass
# -----------------------------

le = LabelEncoder()
le.fit(results_df["fourclass"].dropna().astype(str))


# -----------------------------
# model factories
# -----------------------------

def make_lr():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def make_rf():
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=RF_N_JOBS,
        random_state=RANDOM_STATE,
    )


# -----------------------------
# helpers
# -----------------------------

def build_binary_data(feat_subset, target_col):
    X_all = metrics_df[list(feat_subset)]
    y_all = results_df.loc[X_all.index, target_col]

    valid = X_all.notna().all(axis=1) & y_all.notna()
    X = X_all.loc[valid]
    y = y_all.loc[valid].astype(int)

    return X, y


def binary_auc_for_features(
    feat_subset,
    target_col,
    clf,
    test_size=0.1,
    random_state=RANDOM_STATE,
    use_cv=False,
    n_splits=5,
):
    X, y = build_binary_data(feat_subset, target_col)

    if len(np.unique(y)) < 2:
        return np.nan

    if not use_cv:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_proba)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_proba = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        clf.fit(X_train, y_train)
        y_proba[test_idx] = clf.predict_proba(X_test)[:, 1]

    return roc_auc_score(y, y_proba)


def build_multiclass_data(feat_subset, target_col="fourclass"):
    X_all = metrics_df[list(feat_subset)]
    y_all = results_df.loc[X_all.index, target_col]

    valid = X_all.notna().all(axis=1) & y_all.notna()
    X = X_all.loc[valid]
    y = y_all.loc[valid].astype(str)

    y_enc = le.transform(y)
    return X, y_enc


def multiclass_auc_for_features(
    feat_subset,
    clf,
    test_size=0.1,
    random_state=RANDOM_STATE,
    target_col="fourclass",
    use_cv=False,
    n_splits=5,
):
    X, y_enc = build_multiclass_data(feat_subset, target_col=target_col)

    classes_present = np.unique(y_enc)
    n_classes = len(le.classes_)

    if len(classes_present) < 2:
        return np.nan, [np.nan] * n_classes

    if not use_cv:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y_enc,
            test_size=test_size,
            random_state=random_state,
            stratify=y_enc,
        )

        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)

        y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
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

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_proba_all = np.zeros((len(y_enc), n_classes), dtype=float)

    for train_idx, test_idx in cv.split(X, y_enc):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y_enc[train_idx]
        clf.fit(X_train, y_train)
        y_proba_all[test_idx, :] = clf.predict_proba(X_test)

    y_bin_all = label_binarize(y_enc, classes=np.arange(n_classes))
    micro_auc = roc_auc_score(
        y_bin_all,
        y_proba_all,
        average="micro",
        multi_class="ovr",
    )

    per_class_auc = []
    for k in range(y_bin_all.shape[1]):
        per_class_auc.append(roc_auc_score(y_bin_all[:, k], y_proba_all[:, k]))

    return micro_auc, per_class_auc


def build_up_vs_rest_data(feat_subset):
    X_all = metrics_df[list(feat_subset)]
    y_all = results_df.loc[X_all.index, "fourclass"]

    valid = X_all.notna().all(axis=1) & y_all.notna()
    X = X_all.loc[valid]
    y_str = y_all.loc[valid].astype(str)

    y = (y_str.str.lower() == "up").astype(int)
    return X, y


def up_vs_rest_auc_for_features(
    feat_subset,
    clf,
    test_size=0.1,
    random_state=RANDOM_STATE,
    use_cv=False,
    n_splits=5,
):
    X, y = build_up_vs_rest_data(feat_subset)

    if len(np.unique(y)) < 2:
        return np.nan

    if not use_cv:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, y_proba)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_proba = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        clf.fit(X_train, y_train)
        y_proba[test_idx] = clf.predict_proba(X_test)[:, 1]

    return roc_auc_score(y, y_proba)


# -----------------------------
# main
# -----------------------------

def main():
    feats = FEATURE_COMBO

    print(f"Using features: {feats}")
    print(f"USE_CV={USE_CV}, N_SPLITS={N_SPLITS}")

    models = [
        ("LR", make_lr()),
        ("RF", make_rf()),
    ]

    for name, model in models:
        print(f"\n== {name} ==")

        auc_twist = binary_auc_for_features(
            feats,
            target_col="leah_12k_twist_dna_detected",
            clf=model,
            use_cv=USE_CV,
            n_splits=N_SPLITS,
        )
        print(f"ROC AUC leah_12k_twist_dna_detected: {auc_twist:.4f}")

        auc_detected = binary_auc_for_features(
            feats,
            target_col="leah_12k_detected",
            clf=model,
            use_cv=USE_CV,
            n_splits=N_SPLITS,
        )
        print(f"ROC AUC leah_12k_detected: {auc_detected:.4f}")

        mc_micro, mc_per_class = multiclass_auc_for_features(
            feats,
            clf=model,
            use_cv=USE_CV,
            n_splits=N_SPLITS,
        )
        print(f"ROC AUC fourclass micro: {mc_micro:.4f}")
        for cls_name, auc_val in zip(le.classes_, mc_per_class):
            print(f"ROC AUC fourclass {cls_name}: {auc_val:.4f}")

        auc_up_vs_rest = up_vs_rest_auc_for_features(
            feats,
            clf=model,
            use_cv=USE_CV,
            n_splits=N_SPLITS,
        )
        print(f"ROC AUC fourclass up_vs_rest: {auc_up_vs_rest:.4f}")


if __name__ == "__main__":
    main()