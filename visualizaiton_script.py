import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import os
from ogb.graphproppred import PygGraphPropPredDataset
import pandas as pd
from PIL import Image, ImageDraw
 
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
    plt.savefig(os.path.join(output_dir, "training_metrics.png"), dpi=300)
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
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300)
    plt.close()
 
def plot_attention_scores(data, att_scores, y_true, y_pred, idx, smiles_list, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Access the SMILES string for the given index
    smiles = smiles_list[idx]
    
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Could not parse SMILES for graph {idx}: {smiles}")
        return
    
    att_scores = att_scores.cpu().numpy() if torch.is_tensor(att_scores) else att_scores
    if np.isnan(att_scores).any() or np.isinf(att_scores).any():
        print(f"Attention scores for graph {idx} contain NaN or inf. Skipping.")
        return
    
    if len(att_scores.shape) > 1:
        print(f"Graph {idx} att_scores shape before reduction: {att_scores.shape}")
        att_scores = att_scores.mean(axis=-1)  # Reduce to (num_nodes,)
        print(f"Graph {idx} att_scores shape after reduction: {att_scores.shape}")
    
    att_scores = (att_scores - att_scores.min()) / (att_scores.max() - att_scores.min() + 1e-8)
    
    if att_scores.max() == att_scores.min():
        print(f"Attention scores for graph {idx} are uniform: {att_scores[0]}. Investigate model training.")
    
    # Map attention scores to atoms (blue for low, red for high)
    atom_colors = {}
    num_atoms = mol.GetNumAtoms()
    num_scores = len(att_scores)
    if num_atoms != num_scores:
        print(f"Warning: Number of atoms ({num_atoms}) does not match number of attention scores ({num_scores}) for graph {idx}.")
    
    for i in range(min(num_atoms, num_scores)):
        # Blue (low) to Red (high)
        blue = float(1.0 - att_scores[i])  # 1.0 (low attention) to 0.0 (high attention)
        green = 0.0                       # No green component
        red = float(att_scores[i])        # 0.0 (low attention) to 1.0 (high attention)
        atom_colors[i] = (red, green, blue)
    
    # Map attention scores to bond colors (blue for low, red for high)
    bond_colors = {}
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtomIdx()
        atom2 = bond.GetEndAtomIdx()
        if atom1 < num_scores and atom2 < num_scores:
            # Average attention scores of the two atoms for the bond
            bond_attention = (att_scores[atom1] + att_scores[atom2]) / 2
            # Blue (low) to Red (high)
            blue = float(1.0 - bond_attention)
            green = 0.0
            red = float(bond_attention)
            bond_colors[bond.GetIdx()] = (red, green, blue)
    
    # Create a drawing object using SVG for high-quality output
    drawer = Draw.MolDraw2DSVG(800, 800)
    draw_options = drawer.drawOptions()
    draw_options.setBackgroundColour((1.0, 1.0, 1.0)) 
    draw_options.bondLineWidth = 1.5  
    draw_options.annotationFontScale = 1.5  
    draw_options.fixedBondLength = 40  # Consistent bond length
    draw_options.scaleBondWidth = True  
    draw_options.addStereoAnnotation = True  # Add stereochemistry annotations
    
    # Draw the molecule with highlighted atoms and bonds
    Draw.PrepareMolForDrawing(mol)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(range(num_atoms)),
        highlightAtomColors=atom_colors,
        highlightBonds=list(bond_colors.keys()),
        highlightBondColors=bond_colors
    )
    drawer.FinishDrawing()
    
    svg_file = os.path.join(output_dir, f"molecule_{idx}_true_{y_true:.0f}_pred_{y_pred:.3f}.svg")
    with open(svg_file, 'w') as f:
        f.write(drawer.GetDrawingText())
 
def plot_diverse_attention_scores(test_dataset, attention_scores, y_true, y_pred, smiles_list, num_plots=5, output_dir="plots"):
    """
    Plot attention scores for a diverse subset of graphs, prioritizing positive samples.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    graph_sizes = [g.num_nodes for g in test_dataset]
    graph_labels = [g.y.item() for g in test_dataset]
    
    num_positive = sum(1 for label in graph_labels if label == 1)
    num_negative = len(graph_labels) - num_positive
    print(f"Test set label distribution: {num_positive} positive samples, {num_negative} negative samples")
    
    from collections import defaultdict
    graphs_by_label_and_size = defaultdict(list)
    for idx, (size, label) in enumerate(zip(graph_sizes, graph_labels)):
        graphs_by_label_and_size[(label, size // 10)].append(idx)  # Bucket sizes by tens
    
    selected_indices = []
    positive_indices = []
    negative_indices = []
    
    for (label, size_bucket), indices in graphs_by_label_and_size.items():
        if label == 1:
            positive_indices.extend(indices)
        else:
            negative_indices.extend(indices)
    
    # Ensure at least 3 positive samples (or as many as available) if num_plots >= 3
    num_positive_to_select = min(len(positive_indices), max(3, num_plots // 2))
    if positive_indices:
        selected_indices.extend(np.random.choice(positive_indices, num_positive_to_select, replace=False))
    
    remaining_slots = num_plots - len(selected_indices)
    if remaining_slots > 0 and negative_indices:
        selected_indices.extend(np.random.choice(negative_indices, min(remaining_slots, len(negative_indices)), replace=False))
    
    if len(selected_indices) < num_plots:
        remaining = [i for i in range(len(test_dataset)) if i not in selected_indices]
        selected_indices.extend(np.random.choice(remaining, num_plots - len(selected_indices), replace=False))
    
    selected_indices = sorted(selected_indices[:num_plots])
    
    # Print the selected sample distribution
    selected_labels = [graph_labels[idx] for idx in selected_indices]
    num_selected_positive = sum(1 for label in selected_labels if label == 1)
    print(f"Selected {num_selected_positive} positive samples out of {num_plots} total samples")
    
    # Plot attention scores for selected graphs
    for idx in selected_indices:
        data = test_dataset[idx]
        att_scores = torch.tensor(attention_scores[idx])  # Ensure tensor format
        y_t = y_true[idx].item()
        y_p = y_pred[idx].item()
        plot_attention_scores(data, att_scores, y_t, y_p, idx, smiles_list, output_dir)
 
if __name__ == "__main__":
    # Load test dataset
    dataset = PygGraphPropPredDataset(name="ogbg-molhiv", root="dataset/")
    split_idx = dataset.get_idx_split()
    test_dataset = dataset[split_idx["test"]]
    
    # Load SMILES strings from the raw data
    smiles_df = pd.read_csv(os.path.join("dataset", "ogbg_molhiv", "mapping", "mol.csv.gz"), compression="gzip")
    test_indices = split_idx["test"].numpy()
    smiles_list = smiles_df["smiles"].iloc[test_indices].tolist()
    
    # Load saved predictions and attention scores
    suffix = "edge_on_transform_on_atom_embed_on_pool_attention_dims_192_384_768_dropout_0.2_aggr_add_atom_embed_dim_64_edge_embed_dim_16"
    saved_data = torch.load(f"conv_gcn_pred_{suffix}.pt")
    y_true = saved_data["y_true"].numpy()
    y_pred = saved_data["y_pred"].numpy()
    attention_scores = saved_data["attention_scores"]
    
    # Plot diverse attention scores
    plot_diverse_attention_scores(test_dataset, attention_scores, y_true, y_pred, smiles_list, num_plots=5)
    
    # Plot ROC curve
    plot_roc_curve(y_true, y_pred)
 