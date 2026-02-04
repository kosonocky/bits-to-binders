"""
Simple feature search with Logistic Regression.
Tests all single and double feature combinations with ROC-AUC.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from itertools import combinations
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# Config
METRICS_PATH = "../data/12k_all_metrics.csv"
RESULTS_PATH = "../data/12k_all_results.csv"
OUTPUT_PATH = "comprehensive_feature_search_results.csv"
N_SPLITS = 5
N_JOBS = min(cpu_count() - 1, 20)
RANDOM_STATE = 838975
INCLUDE_DOUBLES = True  # Set to False to only test single features
MAX_DOUBLE_COMBOS = 0  # Set to limit doubles (e.g., 100 for quick testing), None for all
# Load data
print("Loading data...")
metrics_df = pd.read_csv(METRICS_PATH)
results_df = pd.read_csv(RESULTS_PATH)
results_df["leah_12k_Significant"] = results_df["leah_12k_Significant"].fillna(False)

# Get numeric features
exclude = ['global_id', 'team', 'sequence', 'dna_sequence', 'dssp', 
           'longest_dup_substr', 'longest_dup_substr_dna', 'is_linker']
features = [col for col in metrics_df.columns 
            if metrics_df[col].dtype in ['float64', 'int64', 'bool'] 
            and col not in exclude]

print(f"Samples: {len(metrics_df)}, Features: {len(features)}")


def get_xy(feat_list, target, filter_mask=None):
    """Get X, y for a task."""
    if filter_mask is not None:
        X = metrics_df.loc[filter_mask, feat_list]
        y = results_df.loc[filter_mask, target]
    else:
        X = metrics_df[feat_list]
        y = results_df[target]
    
    valid = X.notna().all(axis=1) & y.notna()
    return X.loc[valid], y.loc[valid]


def cv_auc_binary(X, y):
    """Binary classification AUC with CV."""
    if len(X) < 50 or len(np.unique(y)) < 2:
        return np.nan
    
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    y_proba = np.zeros(len(y))
    
    for train_idx, test_idx in cv.split(X, y):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ])
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_proba[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]
    
    return roc_auc_score(y, y_proba)


def cv_auc_multiclass(X, y_str):
    """3-class OvR (one-vs-rest) using sklearn's OneVsRestClassifier."""
    if len(X) < 50 or y_str.isna().any() or len(y_str.unique()) < 2:
        return np.nan, np.nan, np.nan, np.nan
    
    label_map = {'Down': 0, 'Not Sig': 1, 'Up': 2}
    y = y_str.map(label_map).values
    
    try:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        y_proba = np.zeros((len(y), 3))
        
        for train_idx, test_idx in cv.split(X, y):
            # Create base estimator
            base_lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
            model = OneVsRestClassifier(base_lr)
            
            # Fit on training data (handle scaling separately)
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            
            # Scale the data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model.fit(X_train_scaled, y[train_idx])
            y_proba[test_idx] = model.predict_proba(X_test_scaled)
        
        # Compute per-class AUC
        y_bin = label_binarize(y, classes=[0, 1, 2])
        
        try:
            auc_down = roc_auc_score(y_bin[:, 0], y_proba[:, 0])
        except:
            auc_down = np.nan
        
        try:
            auc_notsig = roc_auc_score(y_bin[:, 1], y_proba[:, 1])
        except:
            auc_notsig = np.nan
        
        try:
            auc_up = roc_auc_score(y_bin[:, 2], y_proba[:, 2])
        except:
            auc_up = np.nan
        
        # Average the three AUCs
        aucs = [auc_down, auc_notsig, auc_up]
        valid_aucs = [a for a in aucs if not np.isnan(a)]
        avg_auc = np.mean(valid_aucs) if valid_aucs else np.nan
        
        return avg_auc, auc_down, auc_notsig, auc_up
    except Exception as e:
        return np.nan, np.nan, np.nan, np.nan


def eval_combo(feat_list):
    """Evaluate one feature combination on all tasks."""
    try:
        # Filters
        dna_mask = results_df["leah_12k_twist_dna_detected"] == True
        recovered_mask = dna_mask & (results_df["leah_12k_detected"] == True)
        
        # Task 1: DNA detected (no filter)
        X, y = get_xy(feat_list, "leah_12k_twist_dna_detected")
        auc_dna = cv_auc_binary(X, y.astype(int))
        n_dna = len(X)
        
        # Task 2: Leah detected (filter DNA)
        X, y = get_xy(feat_list, "leah_12k_detected", dna_mask)
        auc_leah = cv_auc_binary(X, y.astype(int))
        n_leah = len(X)
        
        # Task 3: 3-class (filter DNA + recovered)
        X, y = get_xy(feat_list, "fourclass", recovered_mask)
        y = y[y.isin(['Down', 'Not Sig', 'Up'])]
        X = X.loc[y.index]
        auc_3m, auc_d, auc_n, auc_u = cv_auc_multiclass(X, y)
        n_3class = len(X)
        
        return {
            'feature_combo': '|'.join(feat_list),
            'n_features': len(feat_list),
            'n_dna_detected': n_dna,
            'n_leah_detected': n_leah,
            'n_3class': n_3class,
            'auc_dna_detected': auc_dna,
            'auc_leah_detected': auc_leah,
            'auc_3class_avg': auc_3m,
            'auc_3class_down': auc_d,
            'auc_3class_notsig': auc_n,
            'auc_3class_up': auc_u,
        }
    except Exception as e:
        return None


# Main
if __name__ == "__main__":
    start = time.time()
    
    # Generate combos
    print("\nGenerating feature combinations...")
    single = [[f] for f in features]
    double = [list(combo) for combo in combinations(features, 2)][:MAX_DOUBLE_COMBOS] if INCLUDE_DOUBLES else []
    all_combos = single + double
    print(f"Total: {len(all_combos)} ({len(single)} single + {len(double)} double)")
    print(f"Using {N_JOBS} workers\n")
    
    # Run parallel
    results = []
    with Pool(processes=N_JOBS) as pool:
        for i, result in enumerate(pool.imap_unordered(eval_combo, all_combos, chunksize=10)):
            if result is not None:
                results.append(result)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (len(all_combos) - i - 1) / rate
                print(f"Progress: {i+1}/{len(all_combos)} ({100*(i+1)/len(all_combos):.1f}%) - "
                      f"Elapsed: {elapsed/60:.1f}m - Remaining: {remaining/60:.1f}m")
    
    # Save
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)
    
    # Summary
    print(f"\n{'='*80}")
    print("TOP RESULTS")
    print(f"{'='*80}")
    
    single_df = df[df['n_features'] == 1]
    double_df = df[df['n_features'] == 2]
    
    for task, col in [
        ('DNA Detected', 'auc_dna_detected'),
        ('Leah Detected', 'auc_leah_detected'),
        ('3-Class (Avg OvR)', 'auc_3class_avg'),
    ]:
        print(f"\n{task}:")
        print("  Single:", single_df.nlargest(3, col)[['feature_combo', col]].to_string(index=False, header=False))
        print("  Double:", double_df.nlargest(3, col)[['feature_combo', col]].to_string(index=False, header=False))
    
    print(f"\nDone in {(time.time()-start)/60:.1f}m. Saved to {OUTPUT_PATH}")
