# TS-SGCN 

TS-SGCN is a PyTorch project for earthquake magnitude estimation from three-channel seismic waveforms. The model extracts features from both the **temporal domain** and the **frequency domain** (STFT spectrogram), builds a sparse weighted adjacency matrix via self-attention with a learned interaction mask, performs sparse graph convolution over waveform windows, and fuses the two branches with a gated GRU-based estimator to regress the earthquake magnitude.

Datasets used: **NIED** (K-NET/KiK-net dataset, 1996-2025, National Research Institute for Earth Science and Disaster Resilience, Japan) and **STEAD** (Stanford Earthquake Dataset), in various window lengths (3s–30s).

---

## Project Structure

```
TS-SGCN/
├── model.py            # Model definition
├── train.py            # Training entry point
├── predict.py          # Test-set evaluation entry point
├── repeat.py           # Batch experiment runner
├── SeisDataloader.py   # Dataset / DataLoader utilities
├── loss.py             # Loss functions
├── utils.py            # Training utilities & config loader
├── hdf5_stru.py        # HDF5 file structure inspector (exploration utility)
├── config.yaml         # Default hyperparameters and run configuration
├── nied*/ stead*/       # Checkpoint directories (one per experiment)
├── dataset/nied3s       #K-NET/KiK-net dataset，The "dataset/nied3s" directory contains the 3s version of the K-NET/KiK-net dataset,
                          which includes the waveform 3 seconds after the P-wave arrives and the waveform 1 second before the P-wave arrives.
                          Please modify the DataLoader accordingly to load this dataset.
└── predict_result/     # Prediction CSV exports from predict.py
```

### Code Files

| File | Description |
|---|---|
| `model.py` | Defines `MagRegressor`, the main model. Pipeline: windowing → 1D CNN feature extraction (temporal + spectral branches) → `DualPool` (avg + max pooling) → `SparseWeightedAdjacency` (multi-head self-attention + asymmetric-conv interaction mask, normalized by `ZeroSoftmax`) → `SparseGraphConvolution` → `MagnitudeEstimator_mean_gate_stable` (bidirectional GRU + attention pooling per branch, gate fusion, residual MLP regressor). Also contains reusable modules: `SelfAttention`, `InteractionMask`, `AsymmetricConvolution`, `GraphConvolution`, `DualPool`, `PreNormResMLP1`, `ZeroSoftmax`. |
| `train.py` | Training loop. Runs `train_one_epoch` / `eval_one_epoch`, uses `ReduceLROnPlateau` LR scheduling, `EarlyStopping`, saves the best checkpoint by validation loss, and plots the learning curve. |
| `predict.py` | Loads a trained checkpoint, runs inference on the test split, prints MAE / RMSE, and exports per-sample predictions (metadata, ground truth, prediction) to a timestamped CSV. |
| `repeat.py` | Batch experiment runner: executes a list of `train.py` / `predict.py` commands sequentially, with a configurable repeat count. Also pins `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` to prevent libgomp errors and stabilize training. |
| `SeisDataloader.py` | `SeisDataset` (a `torch.utils.data.Dataset` over preloaded tensors) and `make_dataset` / `make_dataloaders`, which load cached `.pt` dataset files from disk into memory. |
| `loss.py` | `weightedHuberLoss`: Huber loss weighted by a log function of the target magnitude, so larger events contribute more to the loss. |
| `utils.py` | `EarlyStopping`, `save_checkpoint` (keeps only the best model), `plot_history`, and `load_config` (merges `config.yaml` with command-line overrides; CLI args take precedence). |
| `hdf5_stru.py` | Standalone utility to inspect the structure of raw HDF5 waveform archives (groups, datasets, attributes, sample data). Used during data exploration, not part of the training pipeline. |

### Configuration File

`config.yaml` holds all default hyperparameters. Key entries:

| Key | Meaning |
|---|---|
| `dataset` | Dataset name (must match the cache directory name, e.g. `nied3s`, `stead30s`) |
| `model_path` | Name of the checkpoint directory to save to / load from |
| `test_model` | Checkpoint filename used by `predict.py` (typically `best_model.pt`) |
| `epochs`, `batch_size`, `lr` | Training schedule |
| `window`, `stride` | Temporal windowing parameters |
| `n_fft`, `hop_length`, `fc` | STFT parameters (window length, hop, frequency cutoff bin) |
| `filters` | CNN channel sizes for the two conv layers |
| `embedding_dims`, `num_heads` | Attention / GCN embedding dimensions and heads |
| `dropout`, `seed`, `early_stopping`, `num_workers` | Misc. training settings |

Any key can be overridden on the command line, e.g. `python train.py --dataset stead30s --epochs 100`.

### Directories

| Directory | Description |
|---|---|
| `nied3s/`, `nied10s-1/`, `nied30s-10hz/`, `stead30s-1/`, ... | Checkpoint directories, one per experiment. Naming convention: `{dataset}-{sampling rate or run id}` (e.g. `nied30s-10hz-2` = NIED dataset, 30 s windows, 10 Hz, run 2). Each contains `best_model.pt` (checkpoint with `epoch`, `state_dict`, `optimizer`, `val_loss`) and `learning_curve.png`. |
| `predict_result/` | Prediction CSVs exported by `predict.py`, named `{dataset}_{model_path}_{timestamp}.csv`. Each row: event metadata, ground-truth magnitude, predicted magnitude. |
| `.idea/` | PyCharm/IDE configuration (not part of the pipeline). |

---

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- numpy, pandas, matplotlib, tqdm, PyYAML, h5py

```bash
pip install torch numpy pandas matplotlib tqdm pyyaml h5py
```

---

## Data Preparation

The pipeline reads **pre-cached tensor files**, not raw HDF5 directly. `SeisDataloader.make_dataset` loads:

```
/root/autodl-tmp/{dataset}/train_dataset.pt
/root/autodl-tmp/{dataset}/val_dataset.pt
/root/autodl-tmp/{dataset}/test_dataset.pt
```

> **Note:** the data root is hardcoded to `/root/autodl-tmp/` (an [AutoDL](https://www.autodl.com/) cloud-GPU convention). Adjust the path in `SeisDataloader.py` if running elsewhere.

Each `.pt` file must be a dict loadable by `torch.load` with keys:

- `"waveform"`: float tensor of shape `(N, 3, L)` — three-channel waveforms
- `"mag"`: array/tensor of shape `(N, 1)` — earthquake magnitudes
- `"meta"`: list of metadata entries (one per sample, e.g. event/station ID)

`hdf5_stru.py` can be used to inspect the raw HDF5 archives before converting them into the cached `.pt` format.

---

## How to Run

### 1. Train

```bash
python train.py --dataset nied3s --model_path nied3s
```

Uses `config.yaml` defaults; override any hyperparameter via CLI. Outputs (in `./{model_path}/`):

- `best_model.pt` — best checkpoint by validation loss
- `learning_curve.png` — train/val loss curve

### 2. Predict / Evaluate

```bash
python predict.py --dataset nied3s --model_path nied3s
```

Loads `{model_path}/{test_model}` (default `best_model.pt`), evaluates on the test split, prints MAE and RMSE, and writes a CSV into the working directory.

### 3. Batch Experiments

Edit the `para` list and `num` (repeat count) in `repeat.py`, then:

```bash
python repeat.py
```

This runs the full train → predict sequence for every `(dataset, model_path)` pair. To switch experiments, just edit the entries in `para`:

```python
para = [
    ("nied3s",  "nied3s"),
    ("stead30s", "stead30s-1"),
]
```

### Typical Workflow

```bash
# 1. (On the training server) make sure dataset caches exist:
#    /root/autodl-tmp/nied3s/{train,val,test}_dataset.pt

# 2. Train
python train.py --dataset nied3s --model_path nied3s

# 3. Evaluate and export predictions
python predict.py --dataset nied3s --model_path nied3s
```

---

## Notes

- `repeat.py` pins `OMP_NUM_THREADS=1` / `MKL_NUM_THREADS=1` to avoid libgomp thread errors and keep training stable.
- `save_checkpoint` only writes `best_model.pt` (per-step checkpoints are disabled in code).
- Checkpoints are saved with `weights_only=False` semantics; load them with `torch.load(..., weights_only=False)` on a trusted environment.
