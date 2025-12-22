import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize


# -----------------------------
# config
# -----------------------------

METRICS_PATH = "../data/12k_all_metrics.csv"
RESULTS_PATH = "../data/12k_all_results.csv"
OUTPUT_CSV = "../results/analysis/feature_single_aucs_10fold_lr_rf_up_vs_rest_threeclass.csv"

N_JOBS = -1
N_SPLITS = 10
RANDOM_STATE = 838975

DNA_DETECTED_COL = "leah_12k_twist_dna_detected"
RECOVERED_COL = "leah_12k_detected"
FOURCLASS_COL = "fourclass"

# simple RFC defaults (tweak if you want)
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None
RF_MIN_SAMPLES_LEAF = 1


# -----------------------------
# load data
# -----------------------------

metrics_df = pd.read_csv(METRICS_PATH)
results_df = pd.read_csv(RESULTS_PATH)

# align and filter: drop rows where either detected flag is False
results_df[DNA_DETECTED_COL] = results_df[DNA_DETECTED_COL].fillna(False).astype(bool)
results_df[RECOVERED_COL] = results_df[RECOVERED_COL].fillna(False).astype(bool)

keep_mask = results_df[DNA_DETECTED_COL] & results_df[RECOVERED_COL]
metrics_df = metrics_df.loc[keep_mask].reset_index(drop=True)
results_df = results_df.loc[keep_mask].reset_index(drop=True)

# numeric feature list (exclude global_id)
feature_cols = [
    c for c in metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    if c != "global_id"
]


# -----------------------------
# derive 3-class target: Up / Not_Sig / Down
# -----------------------------

def _to_threeclass(s):
    s = str(s).strip().lower()
    if s == "up":
        return "Up"
    if s == "down":
        return "Down"
    return "Not_Sig"


threeclass_series = results_df[FOURCLASS_COL].fillna("").map(_to_threeclass)

le3 = LabelEncoder()
le3.fit(threeclass_series.astype(str))
THREECLASS_LABELS = list(le3.classes_)


# -----------------------------
# CV helpers
# -----------------------------

def _effective_n_splits(y, requested_splits):
    y = np.asarray(y)
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min()) if len(counts) else 0
    return max(2, min(requested_splits, min_count))


def _make_lr():
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )


def _make_rf():
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )


def _fit_predict_oof_binary(X, y, clf, n_splits, random_state):
    n_splits_eff = _effective_n_splits(y, n_splits)
    if n_splits_eff < 2:
        return None

    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    y_proba = np.zeros(len(y), dtype=float)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        clf.fit(X_train, y_train)
        y_proba[test_idx] = clf.predict_proba(X_test)[:, 1]

    return y_proba


def _fit_predict_oof_multiclass(X, y_enc, clf, n_splits, random_state, n_classes):
    n_splits_eff = _effective_n_splits(y_enc, n_splits)
    if n_splits_eff < 2:
        return None

    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)
    y_proba_all = np.zeros((len(y_enc), n_classes), dtype=float)

    for train_idx, test_idx in cv.split(X, y_enc):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y_enc[train_idx]
        clf.fit(X_train, y_train)
        y_proba_all[test_idx, :] = clf.predict_proba(X_test)

    return y_proba_all


def up_vs_rest_auc_cv_single_feature(feature, model_name):
    X_all = metrics_df[[feature]]
    y_all = results_df.loc[X_all.index, FOURCLASS_COL]

    valid = X_all.notna().all(axis=1) & y_all.notna()
    X = X_all.loc[valid]
    y_str = y_all.loc[valid].astype(str)

    y = (y_str.str.strip().str.lower() == "up").astype(int)

    if len(np.unique(y)) < 2:
        return np.nan

    if model_name == "lr":
        clf = _make_lr()
    elif model_name == "rf":
        clf = _make_rf()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    try:
        y_proba = _fit_predict_oof_binary(X, y, clf, N_SPLITS, RANDOM_STATE)
        if y_proba is None:
            return np.nan
        return roc_auc_score(y, y_proba)
    except ValueError:
        return np.nan


def threeclass_micro_and_perclass_auc_cv_single_feature(feature, model_name):
    X_all = metrics_df[[feature]]
    y_all = results_df.loc[X_all.index, FOURCLASS_COL]

    valid = X_all.notna().all(axis=1) & y_all.notna()
    X = X_all.loc[valid]
    y_three = y_all.loc[valid].map(_to_threeclass).astype(str)

    if y_three.nunique() < 2:
        return np.nan, {lbl: np.nan for lbl in THREECLASS_LABELS}

    y_enc = le3.transform(y_three)
    n_classes = len(le3.classes_)

    if model_name == "lr":
        clf = _make_lr()
    elif model_name == "rf":
        clf = _make_rf()
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    try:
        y_proba_all = _fit_predict_oof_multiclass(X, y_enc, clf, N_SPLITS, RANDOM_STATE, n_classes)
        if y_proba_all is None:
            return np.nan, {lbl: np.nan for lbl in THREECLASS_LABELS}

        y_bin_all = label_binarize(y_enc, classes=np.arange(n_classes))

        micro_auc = roc_auc_score(
            y_bin_all,
            y_proba_all,
            average="micro",
            multi_class="ovr",
        )

        per_class = {}
        for k, lbl in enumerate(THREECLASS_LABELS):
            try:
                per_class[lbl] = roc_auc_score(y_bin_all[:, k], y_proba_all[:, k])
            except ValueError:
                per_class[lbl] = np.nan

        return micro_auc, per_class
    except ValueError:
        return np.nan, {lbl: np.nan for lbl in THREECLASS_LABELS}


# -----------------------------
# scan
# -----------------------------

def _score_feature(feature):
    row = {"feature": feature}

    for model_name in ("lr", "rf"):
        auc_up = up_vs_rest_auc_cv_single_feature(feature, model_name=model_name)
        auc_3c_micro, auc_3c_perclass = threeclass_micro_and_perclass_auc_cv_single_feature(
            feature, model_name=model_name
        )

        row[f"roc_auc_{model_name}_up_vs_rest_10cv"] = auc_up
        row[f"roc_auc_{model_name}_threeclass_micro_10cv"] = auc_3c_micro

        for lbl in THREECLASS_LABELS:
            safe_lbl = str(lbl).replace(" ", "_")
            row[f"roc_auc_{model_name}_threeclass_{safe_lbl}_10cv"] = auc_3c_perclass.get(lbl, np.nan)

    return row


def main():
    rows = Parallel(n_jobs=N_JOBS)(delayed(_score_feature)(feat) for feat in feature_cols)

    out_df = pd.DataFrame(rows).sort_values(
        by="roc_auc_lr_threeclass_micro_10cv",
        ascending=False,
    ).reset_index(drop=True)

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()