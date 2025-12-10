# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import Counter
# from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
# from scipy.stats import pearsonr
# from statannotations.Annotator import Annotator
# from Bio.SeqUtils.ProtParam import ProteinAnalysis
# from itertools import combinations
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import roc_auc_score


# metrics_df = pd.read_csv("../data/12k_all_metrics.csv")
# metrics_df["closest_ab_pident"] = metrics_df["closest_ab_pident"].fillna(0)
# metrics_df["ubiquitin_pident"] = metrics_df["ubiquitin_pident"].fillna(0)

# results_df = pd.read_csv("../data/12k_all_results.csv")
# results_df["leah_12k_isup"] = results_df["leah_12k_2fold_threshold"] == "Up"
# results_df["leah_12k_Significant"] = results_df["leah_12k_Significant"].fillna(False)
# results_df["leah_12k_twist_dna_detected"] = results_df["twist_dna_read_percentile"].apply(lambda x: x > 0)
# results_df["leah_12k_2fold_threshold_int"] = results_df["leah_12k_2fold_threshold"].map(
#     {"Up": 1, "Not Sig": 0, "Down": -1, np.nan: np.nan}
# )


# def compute_human_frequencies():
#     human_freqs = Counter()
#     total_aa = 0
#     with open("../data/homo_sapiens_UP000005640_9606.fasta", "r") as f:
#         seq_parts = []
#         for line in f:
#             line = line.strip()
#             if line.startswith(">"):
#                 if seq_parts:
#                     seq = "".join(seq_parts)
#                     human_freqs.update(seq)
#                     total_aa += len(seq)
#                     seq_parts = []
#             else:
#                 seq_parts.append(line)

#         if seq_parts:
#             seq = "".join(seq_parts)
#             human_freqs.update(seq)
#             total_aa += len(seq)

#     freqs = {aa: human_freqs[aa] / total_aa for aa in human_freqs}
#     return freqs

# human_freqs = compute_human_frequencies()
# print(human_freqs)
    


# def kl_between_seq_and_human(seq):
#     """
#     Compute KL divergence between the amino acid composition of seq
#     and the human proteome frequencies.
#     """
#     pa = ProteinAnalysis(seq)
#     freqs = pa.get_amino_acids_percent()

#     kl_div = 0.0
#     for aa in human_freqs.keys():
#         p = freqs.get(aa, 0.0)
#         q = human_freqs[aa]
#         if p > 0:
#             kl_div += p * math.log2(p / q)
#     return kl_div

# def compute_gravy(sequence: str) -> float:
#     """
#     Compute GRAVY (Grand Average of Hydropathy) for an amino acid sequence.
#     """
#     seq = sequence.replace("*", "").upper()  # remove stop if present
#     analyzed_seq = ProteinAnalysis(seq)
#     return analyzed_seq.gravy()

# def sequence_complexity(seq):
#     """
#     Shannon entropy over amino acid frequencies.
#     Higher entropy means higher complexity.
#     """
#     seq = seq.replace("*", "").upper()
#     pa = ProteinAnalysis(seq)
#     freqs = pa.get_amino_acids_percent()

#     entropy = 0.0
#     for aa, p in freqs.items():
#         if p > 0:
#             entropy += -p * math.log2(p)
#     return entropy

# def dna_sequence_entropy(seq):
#     """
#     Shannon entropy over nucleotide frequencies.
#     Higher entropy means higher complexity.
#     """
#     seq = seq.upper()
#     length = len(seq)
#     freqs = Counter(seq)
    
#     entropy = 0.0
#     for nucleotide, count in freqs.items():
#         p = count / length
#         if p > 0:
#             entropy += -p * math.log2(p)
#     return entropy

# def num_cysteines(seq):
#     """
#     Count the number of cysteine residues in the sequence.
#     """
#     seq = seq.replace("*", "").upper()
#     return seq.count('C')

# def compute_charge(sequence: str) -> float:
#     """
#     Compute the net charge of an amino acid sequence at pH 7.0.
#     """
#     seq = sequence.replace("*", "").upper()  # remove stop if present
#     analyzed_seq = ProteinAnalysis(seq)
#     return analyzed_seq.charge_at_pH(7.0)

# def gc_content(sequence: str) -> float:
#     """
#     Compute the GC content of a DNA sequence.
#     """
#     sequence = sequence.upper()
#     g_count = sequence.count('G')
#     c_count = sequence.count('C')
#     gc_count = g_count + c_count
#     return gc_count / len(sequence) if len(sequence) > 0 else 0.0

# def contains_linker(seq):
#     linker_seqs = ["GGGGS", "GGGS", "GGS", "GGGGGS"]
#     for linker in linker_seqs:
#         if linker in seq:
#             return True
#     return False

# def longest_duplicated_substring(seq):
#     """
#     Find the longest duplicated substring in the sequence such that
#     the two defining occurrences do not overlap.

#     Return: (length, substring, total_occurrences)
#     where total_occurrences counts all (possibly overlapping) matches.
#     """
#     def count_overlapping(haystack, needle):
#         if not needle:
#             return 0
#         count = 0
#         i = 0
#         while True:
#             i = haystack.find(needle, i)
#             if i == -1:
#                 break
#             count += 1
#             i += 1  # move by 1 to allow overlaps
#         return count

#     n = len(seq)
#     suffixes = sorted((seq[i:], i) for i in range(n))

#     max_len = 0
#     best_substring = ""

#     for i in range(1, n):
#         s1, idx1 = suffixes[i - 1]
#         s2, idx2 = suffixes[i]

#         # compute LCP between neighboring suffixes
#         j = 0
#         limit = min(len(s1), len(s2))
#         while j < limit and s1[j] == s2[j]:
#             j += 1

#         # prevent overlap between the two suffixes used to define the match
#         distance = abs(idx1 - idx2)
#         lcp_no_overlap = min(j, distance)

#         if lcp_no_overlap > max_len:
#             max_len = lcp_no_overlap
#             start = min(idx1, idx2)
#             best_substring = seq[start:start + max_len]

#     if max_len == 0:
#         return 0, "", 0

#     total_occurrences = count_overlapping(seq, best_substring)
#     return max_len, best_substring, total_occurrences

# def map_four_class_int_to_label(v):
#     if pd.isna(v):
#         return "Not Recovered"
#     if v == 0:
#         return "Not Sig"
#     if v == 1:
#         return "Up"
#     if v == -1:
#         return "Down"
#     return "Not Recovered"

# metrics_df["gravy_score"] = metrics_df["sequence"].apply(compute_gravy)
# metrics_df["aa_sequence_entropy"] = metrics_df["sequence"].apply(sequence_complexity)
# metrics_df["dna_sequence_entropy"] = results_df["dna_sequence"].apply(dna_sequence_entropy)
# metrics_df["num_cysteines"] = metrics_df["sequence"].apply(num_cysteines)
# metrics_df["seq_kl_vs_human"] = metrics_df["sequence"].apply(kl_between_seq_and_human)
# metrics_df["seq_charge"] = metrics_df["sequence"].apply(compute_charge)
# metrics_df["dna_gc_content"] = results_df["dna_sequence"].apply(gc_content)
# metrics_df["is_linker"] = metrics_df["sequence"].apply(contains_linker)
# metrics_df[["longest_dup_substr_len", "longest_dup_substr", "longest_dup_occurrences"]] = metrics_df["sequence"].apply(
#     lambda seq: pd.Series(longest_duplicated_substring(seq))
# )
# metrics_df["total_duplicated_residues"] = metrics_df["longest_dup_substr_len"] * metrics_df["longest_dup_occurrences"]

# results_df["fourclass"] = results_df["leah_12k_2fold_threshold_int"].apply(map_four_class_int_to_label)

# # -----------------------------
# # feature set: all numeric columns in metrics_df
# # -----------------------------
# feature_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()


# # -----------------------------
# # helper to build X, y for binary target leah_12k_twist_dna_detected
# # -----------------------------
# def build_binary_data_twist(feat_subset):
#     X_all = metrics_df[list(feat_subset)]
#     y_all = results_df.loc[X_all.index, "leah_12k_twist_dna_detected"]

#     valid = X_all.notna().all(axis=1) & y_all.notna()
#     X = X_all.loc[valid]
#     y = y_all.loc[valid].astype(int)

#     return X, y

# # -----------------------------
# # helper to train and eval binary AUC
# # -----------------------------
# def binary_auc_for_features_twist(feat_subset, test_size=0.1, random_state=838975):
#     X, y = build_binary_data_twist(feat_subset)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X,
#         y,
#         test_size=test_size,
#         random_state=random_state,
#         stratify=y,
#     )

#     clf = Pipeline(
#         steps=[
#             ("scaler", StandardScaler()),
#             (
#                 "logreg",
#                 LogisticRegression(
#                     max_iter=1000,
#                 ),
#             ),
#         ]
#     )

#     clf.fit(X_train, y_train)
#     y_proba = clf.predict_proba(X_test)[:, 1]

#     roc_auc = roc_auc_score(y_test, y_proba)
#     return roc_auc

# # -----------------------------
# # 1) univariate scan across all numeric features
# # -----------------------------
# single_rows_twist = []

# for feat in feature_cols:
#     auc = binary_auc_for_features_twist([feat])
#     single_rows_twist.append(
#         {
#             "feature": feat,
#             "roc_auc_leah_12k_twist_dna_detected": auc,
#         }
#     )

# single_auc_twist_df = pd.DataFrame(single_rows_twist)
# single_auc_twist_df = single_auc_twist_df.sort_values(
#     by="roc_auc_leah_12k_twist_dna_detected",
#     ascending=False,
# ).reset_index(drop=True)

# single_auc_twist_df.to_csv(
#     "../results/analysis/feature_single_aucs_leah_12k_twist_dna_detected.csv",
#     index=False,
# )

# top_features_twist = single_auc_twist_df["feature"].tolist()

# # -----------------------------
# # 2) combinations up to 3 features within top_features_twist
# # -----------------------------
# rows_twist = []

# max_k = min(3, len(top_features_twist))
# for k in range(1, max_k + 1):
#     for feat_subset in combinations(top_features_twist, k):
#         feat_subset = list(feat_subset)

#         bin_auc = binary_auc_for_features_twist(feat_subset)

#         row = {
#             "features": tuple(feat_subset),
#             "n_features": len(feat_subset),
#             "roc_auc_leah_12k_twist_dna_detected": bin_auc,
#         }

#         rows_twist.append(row)

# auc_twist_df = pd.DataFrame(rows_twist)

# auc_twist_df = auc_twist_df.sort_values(
#     by="roc_auc_leah_12k_twist_dna_detected",
#     ascending=False,
# ).reset_index(drop=True)

# auc_twist_df.to_csv(
#     "../results/analysis/feature_combination_aucs_leah_12k_twist_dna_detected.csv",
#     index=False,
# )



# # -----------------------------
# # 4 class labels
# # -----------------------------
# results_df["leah_12k_2fold_threshold_int"] = results_df["leah_12k_2fold_threshold"].map(
#     {"Up": 1, "Not Sig": 0, "Down": -1, np.nan: np.nan}
# )

# def map_four_class(v):
#     if pd.isna(v):
#         return "Not Recovered"
#     if v == -1:
#         return "Down"
#     if v == 0:
#         return "Not Sig"
#     if v == 1:
#         return "Up"
#     return "Not Recovered"

# results_df["fourclass"] = results_df["leah_12k_2fold_threshold_int"].apply(map_four_class)

# # feature set: all numeric columns in metrics_df
# feature_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()

# # label encoder fit once on all available 4 class labels
# le = LabelEncoder()
# le.fit(results_df["fourclass"].dropna())

# # -----------------------------
# # 1) univariate scan across all numeric features
# # -----------------------------
# single_rows = []

# for feat in feature_cols:
#     mc_auc_micro, mc_auc_per_class = multiclass_auc_for_features([feat])
#     bin_auc = binary_auc_for_features([feat])

#     row = {
#         "feature": feat,
#         "n_features": 1,
#         "roc_auc_4class_micro": mc_auc_micro,
#         "roc_auc_binary_detected": bin_auc,
#     }

#     # add per class AUCs
#     for cls_name, auc_val in zip(le.classes_, mc_auc_per_class):
#         col_name = f"roc_auc_4class_{cls_name}"
#         row[col_name] = auc_val

#     single_rows.append(row)

# single_auc_df = pd.DataFrame(single_rows)
# single_auc_df = single_auc_df.sort_values(
#     by="roc_auc_4class_micro",
#     ascending=False,
# ).reset_index(drop=True)

# single_auc_df.to_csv(
#     "../results/analysis/feature_single_aucs_fourclass_leah_12k.csv",
#     index=False,
# )

# top_features = single_auc_df["feature"].tolist()

# # -----------------------------
# # 2) combinations up to 3 features within top_features
# # -----------------------------
# rows = []

# max_k = min(3, len(top_features))
# for k in range(1, max_k + 1):
#     for feat_subset in combinations(top_features, k):
#         feat_subset = list(feat_subset)

#         mc_auc_micro, mc_auc_per_class = multiclass_auc_for_features(feat_subset)
#         bin_auc = binary_auc_for_features(feat_subset)

#         row = {
#             "features": tuple(feat_subset),
#             "n_features": len(feat_subset),
#             "roc_auc_4class_micro": mc_auc_micro,
#             "roc_auc_binary_detected": bin_auc,
#         }

#         for cls_name, auc_val in zip(le.classes_, mc_auc_per_class):
#             col_name = f"roc_auc_4class_{cls_name}"
#             row[col_name] = auc_val

#         rows.append(row)

# auc_df = pd.DataFrame(rows)

# auc_df = auc_df.sort_values(
#     by="roc_auc_4class_micro",
#     ascending=False,
# ).reset_index(drop=True)

# auc_df.to_csv(
#     "../results/analysis/feature_combination_aucs_fourclass_leah_12k.csv",
#     index=False,
# )


# # -----------------------------
# # feature set: all numeric columns in metrics_df
# # -----------------------------
# feature_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()

# # -----------------------------
# # 1) univariate scan across all numeric features
# # -----------------------------
# single_rows = []

# for feat in feature_cols:
#     auc = binary_auc_for_features([feat])
#     single_rows.append(
#         {
#             "feature": feat,
#             "roc_auc_leah_12k_detected": auc,
#         }
#     )

# single_auc_df = pd.DataFrame(single_rows)
# single_auc_df = single_auc_df.sort_values(
#     by="roc_auc_leah_12k_detected",
#     ascending=False,
# ).reset_index(drop=True)

# # optional: save univariate results
# single_auc_df.to_csv(
#     "../results/analysis/feature_single_aucs_leah_12k_detected.csv",
#     index=False,
# )

# top_features = single_auc_df["feature"].tolist()

# # -----------------------------
# # 2) combinations up to 3 features within top_features
# # -----------------------------
# rows = []

# max_k = min(3, len(top_features))
# for k in range(1, max_k + 1):
#     for feat_subset in combinations(top_features, k):
#         feat_subset = list(feat_subset)

#         bin_auc = binary_auc_for_features(feat_subset)

#         row = {
#             "features": tuple(feat_subset),
#             "n_features": len(feat_subset),
#             "roc_auc_leah_12k_detected": bin_auc,
#         }

#         rows.append(row)

# auc_df = pd.DataFrame(rows)

# # optional: sort by something
# auc_df = auc_df.sort_values(
#     by="roc_auc_leah_12k_detected",
#     ascending=False,
# ).reset_index(drop=True)

# auc_df.to_csv(
#     "../results/analysis/feature_combination_aucs_leah_12k_detected.csv",
#     index=False,
# )



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
results_df["leah_12k_isup"] = results_df["leah_12k_2fold_threshold"] == "Up"
results_df["leah_12k_Significant"] = results_df["leah_12k_Significant"].fillna(False)
results_df["leah_12k_twist_dna_detected"] = (
    results_df["twist_dna_read_percentile"].apply(lambda x: x > 0)
)

results_df["leah_12k_2fold_threshold_int"] = results_df[
    "leah_12k_2fold_threshold"
].map({"Up": 1, "Not Sig": 0, "Down": -1, np.nan: np.nan})


# -----------------------------
# human proteome frequencies
# -----------------------------

def compute_human_frequencies():
    human_freqs = Counter()
    total_aa = 0
    with open("../data/homo_sapiens_UP000005640_9606.fasta", "r") as f:
        seq_parts = []
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if seq_parts:
                    seq = "".join(seq_parts)
                    human_freqs.update(seq)
                    total_aa += len(seq)
                    seq_parts = []
            else:
                seq_parts.append(line)

        if seq_parts:
            seq = "".join(seq_parts)
            human_freqs.update(seq)
            total_aa += len(seq)

    freqs = {aa: human_freqs[aa] / total_aa for aa in human_freqs}
    return freqs


human_freqs = compute_human_frequencies()


# -----------------------------
# sequence feature functions
# -----------------------------

def kl_between_seq_and_human(seq):
    """
    KL divergence between amino acid composition of seq
    and human proteome frequencies (in bits).
    """
    pa = ProteinAnalysis(seq.replace("*", "").upper())
    freqs = pa.get_amino_acids_percent()

    kl_div = 0.0
    for aa in human_freqs.keys():
        p = freqs.get(aa, 0.0)
        q = human_freqs[aa]
        if p > 0:
            kl_div += p * math.log2(p / q)
    return kl_div


def compute_gravy(sequence: str) -> float:
    seq = sequence.replace("*", "").upper()
    analyzed_seq = ProteinAnalysis(seq)
    return analyzed_seq.gravy()


def sequence_complexity(seq):
    """
    Shannon entropy over amino acid frequencies.
    """
    seq = seq.replace("*", "").upper()
    pa = ProteinAnalysis(seq)
    freqs = pa.get_amino_acids_percent()

    entropy = 0.0
    for p in freqs.values():
        if p > 0:
            entropy += -p * math.log2(p)
    return entropy


def dna_sequence_entropy(seq):
    """
    Shannon entropy over nucleotide frequencies.
    """
    seq = seq.upper()
    length = len(seq)
    if length == 0:
        return 0.0

    freqs = Counter(seq)
    entropy = 0.0
    for count in freqs.values():
        p = count / length
        if p > 0:
            entropy += -p * math.log2(p)
    return entropy


def num_cysteines(seq):
    seq = seq.replace("*", "").upper()
    return seq.count("C")


def compute_charge(sequence: str) -> float:
    seq = sequence.replace("*", "").upper()
    analyzed_seq = ProteinAnalysis(seq)
    return analyzed_seq.charge_at_pH(7.0)


def gc_content(sequence: str) -> float:
    sequence = sequence.upper()
    if not sequence:
        return 0.0
    g = sequence.count("G")
    c = sequence.count("C")
    return (g + c) / len(sequence)


def contains_linker(seq):
    linker_seqs = ["GGGGS", "GGGS", "GGS", "GGGGGS"]
    seq = seq.upper()
    return any(linker in seq for linker in linker_seqs)


def longest_duplicated_substring(seq):
    """
    Longest non overlapping duplicated substring.
    Return (length, substring, total_occurrences).
    """
    def count_overlapping(haystack, needle):
        if not needle:
            return 0
        count = 0
        i = 0
        while True:
            i = haystack.find(needle, i)
            if i == -1:
                break
            count += 1
            i += 1
        return count

    n = len(seq)
    suffixes = sorted((seq[i:], i) for i in range(n))

    max_len = 0
    best_substring = ""

    for i in range(1, n):
        s1, idx1 = suffixes[i - 1]
        s2, idx2 = suffixes[i]

        j = 0
        limit = min(len(s1), len(s2))
        while j < limit and s1[j] == s2[j]:
            j += 1

        distance = abs(idx1 - idx2)
        lcp_no_overlap = min(j, distance)

        if lcp_no_overlap > max_len:
            max_len = lcp_no_overlap
            start = min(idx1, idx2)
            best_substring = seq[start : start + max_len]

    if max_len == 0:
        return 0, "", 0

    total_occurrences = count_overlapping(seq, best_substring)
    return max_len, best_substring, total_occurrences


# -----------------------------
# apply feature functions
# -----------------------------

metrics_df["gravy_score"] = metrics_df["sequence"].apply(compute_gravy)
metrics_df["aa_sequence_entropy"] = metrics_df["sequence"].apply(sequence_complexity)
metrics_df["dna_sequence_entropy"] = results_df["dna_sequence"].apply(
    dna_sequence_entropy
)
metrics_df["num_cysteines"] = metrics_df["sequence"].apply(num_cysteines)
metrics_df["seq_kl_vs_human"] = metrics_df["sequence"].apply(kl_between_seq_and_human)
metrics_df["seq_charge"] = metrics_df["sequence"].apply(compute_charge)
metrics_df["dna_gc_content"] = results_df["dna_sequence"].apply(gc_content)
metrics_df["is_linker"] = metrics_df["sequence"].apply(contains_linker)

metrics_df[
    ["longest_dup_substr_len", "longest_dup_substr", "longest_dup_occurrences"]
] = metrics_df["sequence"].apply(
    lambda seq: pd.Series(longest_duplicated_substring(seq))
)
metrics_df["total_duplicated_residues"] = (
    metrics_df["longest_dup_substr_len"] * metrics_df["longest_dup_occurrences"]
)


# -----------------------------
# fourclass labels
# -----------------------------

def map_four_class(v):
    if pd.isna(v):
        return "Not Recovered"
    if v == -1:
        return "Down"
    if v == 0:
        return "Not Sig"
    if v == 1:
        return "Up"
    return "Not Recovered"


results_df["fourclass"] = results_df["leah_12k_2fold_threshold_int"].apply(
    map_four_class
)


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