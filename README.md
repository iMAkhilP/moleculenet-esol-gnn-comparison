# MoleculeNet ESOL GNN Architecture Comparison

This repository compares three graph neural network architectures for molecular solubility prediction on the **MoleculeNet ESOL** dataset using **PyTorch Geometric**.

The project is designed as a compact, end-to-end learning project for understanding molecular graph regression. Each molecule is represented as a graph, where atoms are nodes, chemical bonds are edges, atom descriptors are node features, and bond descriptors are edge attributes.

## Project goal

The goal is to predict the aqueous solubility value of a molecule from its graph representation and compare how different edge-aware GNN architectures perform under the same train/validation/test split and training protocol.

## Dataset

- **Dataset:** MoleculeNet ESOL
- **Task type:** Graph-level regression
- **Target:** Aqueous solubility value
- **Number of molecules:** 1128
- **Node features:** 9
- **Edge features:** 3
- **Split:** 60% train, 20% validation, 20% test
- **Train molecules:** 676
- **Validation molecules:** 225
- **Test molecules:** 227

## Architectures compared

| Model | Main idea | Edge/bond feature usage |
|---|---|---|
| `GINEConv` | Edge-aware extension of GIN for molecular message passing | Edge attributes are injected directly into the message-passing update |
| `AttentiveFP` | Attention-based molecular graph model with graph readout steps | Edge attributes are used through PyG's edge-aware AttentiveFP implementation |
| `NNConv` | Edge-conditioned convolution where bonds generate transformation filters | Edge attributes generate message transformation matrices |

## Training setup

All models are trained under the same experimental protocol:

- Same deterministic random seed
- Same 60/20/20 train/validation/test split
- Same batch size: `32`
- Same optimizer: `AdamW`
- Same base learning rate: `0.001`
- Same weight decay: `1e-4`
- Same learning-rate scheduler: cosine annealing
- Same early stopping criterion based on validation MSE
- Same evaluation metrics: MSE, RMSE, MAE, and R²

## Results

| Model | Parameters | Best epoch | Trained epochs | Best val MSE | Test MSE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **AttentiveFP** | 23,841 | 478 | 579 | 0.4054 | 0.3790 | **0.6156** | 0.4507 | **0.9197** |
| **NNConv** | 72,513 | 22 | 123 | 2.8837 | 2.8612 | 1.6915 | 1.3297 | 0.3934 |
| **GINEConv** | 7,041 | 1 | 102 | 3.4986 | 4.1352 | 2.0335 | 1.5557 | 0.1233 |

For this particular split and training run, **AttentiveFP** performs best, achieving the lowest test RMSE and highest test R².

## Test RMSE comparison

<p align="center">
  <img src="assets/RMSE comp.png" width="75%">
</p>

## Model diagnostics

Each row shows the learning behavior of one architecture: training/validation loss, learning-rate schedule, and predicted-vs-true ESOL values on the test set.

### GINEConv

| Training vs validation loss | Learning-rate schedule | Test predictions |
|---|---|---|
| <img src="assets/GINEConv train-val.png" width="100%"> | <img src="assets/GINEConv lrs.png" width="100%"> | <img src="assets/GINEConv test pred.png" width="100%"> |

### AttentiveFP

| Training vs validation loss | Learning-rate schedule | Test predictions |
|---|---|---|
| <img src="assets/AttentiveFP train-val.png" width="100%"> | <img src="assets/AttentiveFP lrs.png" width="100%"> | <img src="assets/AttentiveFP test pred.png" width="100%"> |

### NNConv

| Training vs validation loss | Learning-rate schedule | Test predictions |
|---|---|---|
| <img src="assets/NNConv train-val.png" width="100%"> | <img src="assets/NNConv lrs.png" width="100%"> | <img src="assets/NNConv test pred.png" width="100%"> |

## Sample molecule visualization

The notebook also uses **RDKit** to visualize a molecule from the ESOL dataset and connect the chemical structure to the graph representation used by PyTorch Geometric.

<p align="center">
  <img src="assets/sample molecule.png" width="55%">
</p>

## Repository structure

```text
.
├── ESOL_GNN.ipynb
├── README.md
└── assets/
    ├── attentivefp_lr_schedule.png
    ├── attentivefp_test_predictions.png
    ├── attentivefp_train_val_loss.png
    ├── gineconv_lr_schedule.png
    ├── gineconv_test_predictions.png
    ├── gineconv_train_val_loss.png
    ├── nnconv_lr_schedule.png
    ├── nnconv_test_predictions.png
    ├── nnconv_train_val_loss.png
    ├── rmse_comparison.png
    └── sample_molecule.png
```

## How to run

Open the notebook in Google Colab or Jupyter and run the cells from top to bottom.

The notebook installs and uses:

```python
torch
torch_geometric
rdkit
numpy
pandas
scikit-learn
matplotlib
seaborn
```

## Key takeaways

- Molecular property prediction can be naturally formulated as graph-level regression.
- Edge attributes are important because chemical bonds carry useful information for molecular graphs.
- The same train/validation/test split is essential for fair architecture comparison.
- In this run, `AttentiveFP` generalizes much better than `GINEConv` and `NNConv` on the held-out ESOL test set.
- For a stronger molecular machine-learning benchmark, a scaffold split would be a better next step because it tests generalization to chemically different molecular structures.

## Possible next steps

- Add scaffold split evaluation.
- Repeat experiments across multiple random seeds.
- Tune hyperparameters separately for each architecture.
- Add additional MoleculeNet datasets such as FreeSolv or Lipophilicity.
- Compare against simple molecular fingerprint baselines.
