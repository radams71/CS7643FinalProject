"""
Taken from: https://github.com/yzhuoning/DeepAUC_OGB_Challenge
"""

import os
import shutil
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from ogb.graphproppred import GraphPropPredDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score


def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def main(args):
    all_probs = {}
    all_ap = {}
    all_rocs = {}
    train_label_props = {}

    n_estimators = 1000
    max_tasks = None
    run_times = 10

    eval_scores = []
    test_scores = []

    mgf_file = "./dataset/%s/mgf_feat.npy" % (args.dataset_name.replace("-", "_"))
    maccs_file = "./dataset/%s/maccs_feat.npy" % (args.dataset_name.replace("-", "_"))
    mgf_feat = np.load(mgf_file)
    maccs_feat = np.load(maccs_file)

    dataset = GraphPropPredDataset(name=args.dataset_name, root="./dataset/")
    smiles_file = "./dataset/%s/mapping/mol.csv.gz" % (args.dataset_name.replace("-", "_"))
    df_smi = pd.read_csv(smiles_file)
    smiles = df_smi["smiles"]
    outcomes = df_smi.set_index("smiles").drop(["mol_id"], axis=1)

    feat = np.concatenate([mgf_feat, maccs_feat], axis=1)
    print("features size:", feat.shape[1])

    X = pd.DataFrame(feat,
                     index=smiles,
                     columns=[i for i in range(feat.shape[1])])

    # Split into train/val/test
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]

    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]

    for rep in range(run_times):
        for oo in tqdm(outcomes.columns[:max_tasks]):
            # Get probabilities
            val_key = args.dataset_name, oo, rep, "val"
            test_key = args.dataset_name, oo, rep, "test"

            # If re-running, skip finished runs
            if val_key in all_probs:
                print("Skipping", val_key[:-1])
                continue

            # Split outcome in to train/val/test
            Y = outcomes[oo]
            y_train, y_val, y_test = Y.loc[X_train.index], Y.loc[X_val.index], Y.loc[X_test.index]

            # Skip outcomes with no positive training examples
            if y_train.sum() == 0:
                continue

            # Remove missing labels in validation
            y_val, y_test = y_val.dropna(), y_test.dropna()
            X_v, X_t = X_val.loc[y_val.index], X_test.loc[y_test.index]

            # Remove missing values in the training labels, and downsample imbalance to cut runtime
            y_tr = y_train.dropna()
            train_label_props[args.dataset_name, oo, rep] = y_tr.mean()
            print(f"Sampled label balance:\n{y_tr.value_counts()}")

            # Fit model
            print("Fitting model...")
            rf = RandomForestClassifier(min_samples_leaf=2,
                                        n_estimators=n_estimators,
                                        n_jobs=-1,
                                        criterion='entropy',
                                        class_weight={0: 1, 1: 10},
                                        random_state=rep
                                        )
            rf.fit(X_train.loc[y_tr.index], y_tr)

            # Calculate probabilities
            all_probs[val_key] = pd.Series(rf.predict_proba(X_v)[:, 1], index=X_v.index)
            all_probs[test_key] = pd.Series(rf.predict_proba(X_t)[:, 1], index=X_t.index)

            if y_val.sum() > 0:
                all_ap[val_key] = average_precision_score(y_val, all_probs[val_key])
                all_rocs[val_key] = roc_auc_score(y_val, all_probs[val_key])

            if y_test.sum() > 0:
                all_ap[test_key] = average_precision_score(y_test, all_probs[test_key])
                all_rocs[test_key] = roc_auc_score(y_test, all_probs[test_key])

            print(
                f'{oo}, rep {rep}, AP (val, test): {all_ap.get(val_key, np.nan):.3f}, {all_ap.get(test_key, np.nan):.3f}')
            print(f'\tROC (val, test): {all_rocs.get(val_key, np.nan):.3f}, {all_rocs.get(test_key, np.nan):.3f}')
            eval_scores.append(all_rocs.get(val_key, np.nan))
            test_scores.append(all_rocs.get(test_key, np.nan))

            # save pred
            all_prob = np.array(rf.predict_proba(X))
            print(all_prob.shape)
            os.makedirs('rf_preds', exist_ok=True)
            np.save(
                "./rf_preds/rf_pred_auc_{:d}.npy".format(rep),
                all_prob)


#    final_index = np.argmax(eval_scores)
#    best_val_score = eval_scores[final_index]
#    final_test_score = test_scores[final_index]
#    shutil.copy2("./rf_preds/rf_pred_auc_{:0.4f}_{:0.4f}.npy".format(best_val_score, final_test_score), './rf_preds/rf_final_pred.npy')
#    print("Best preds saved in ./rf_preds/rf_final_pred.npy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='gnn')
    parser.add_argument("--dataset_name", type=str, default="ogbg-molhiv")
    args = parser.parse_args()
    main(args)