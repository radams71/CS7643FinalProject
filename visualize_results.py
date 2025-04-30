import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import os
 
def plot_training_metrics(train_losses, valid_aucs, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(valid_aucs, label="Valid ROC-AUC", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("ROC-AUC")
    plt.title("Validation ROC-AUC Over Epochs")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_metrics.png"))
    plt.close()
 
def plot_roc_curve(y_true, y_pred, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve on Test Set")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
 
def plot_attention_scores(data, att_scores, y_true, y_pred, idx, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Placeholder SMILES (requires actual SMILES retrieval for ogbg-molhiv)
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin as a placeholder
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not parse SMILES for graph {idx}")
        return
    
    att_scores = att_scores.cpu().numpy()
    att_scores = (att_scores - att_scores.min()) / (att_scores.max() - att_scores.min() + 1e-8)
    
    atom_colors = {}
    for i in range(min(mol.GetNumAtoms(), len(att_scores))):
        atom_colors[i] = (1.0, 1.0 - att_scores[i], 1.0 - att_scores[i])  # Red (high) to White (low)
    
    img = Draw.MolToImage(mol, size=(300, 300), highlightAtoms=list(range(mol.GetNumAtoms())), highlightMap=atom_colors)
    img.save(os.path.join(output_dir, f"molecule_{idx}_true_{y_true:.0f}_pred_{y_pred:.3f}.png"))
 