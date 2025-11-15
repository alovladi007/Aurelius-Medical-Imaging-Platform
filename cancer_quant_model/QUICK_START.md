# Quick Start Guide

## Project Overview

Complete production-ready quantitative cancer histopathology modeling system with:
- **3 Model Architectures**: ResNet, EfficientNet, Vision Transformer
- **Quantitative Features**: 100+ classic features (color, texture, morphology) + deep embeddings
- **Explainability**: Grad-CAM visualizations
- **Full MLOps**: Experiment tracking, checkpointing, callbacks
- **REST API**: FastAPI server for deployment
- **56 files, 7,989 lines of code**

## Installation (5 minutes)

```bash
cd cancer_quant_model

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import timm; import albumentations; print('✓ All dependencies installed')"
```

## Your First Model (10 minutes)

### 1. Prepare Sample Dataset

For testing, create a minimal dataset:
```bash
mkdir -p data/raw/train/{0,1}
# Add a few sample images to each class folder
```

Or download a Kaggle dataset:
```bash
kaggle datasets download -d <dataset-name>
unzip <dataset-name>.zip -d data/raw/
```

### 2. Configure Dataset

Edit `config/dataset.yaml` to match your structure (folder_binary or csv_labels).

### 3. Prepare Data and Create Splits

```bash
python scripts/prepare_data.py --config config/dataset.yaml
python scripts/create_splits.py --config config/dataset.yaml
```

### 4. Train ResNet-50 (Quick Test)

```bash
python scripts/train.py \
  --model_config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml \
  --train_config config/train_default.yaml
```

### 5. Monitor Training

```bash
# In another terminal
mlflow ui --backend-store-uri experiments/mlruns --port 5000
# Open http://localhost:5000
```

### 6. Evaluate Model

```bash
python scripts/evaluate.py \
  --checkpoint experiments/logs/resnet/best.pt \
  --config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml \
  --output_dir experiments/evaluation
```

Results in `experiments/evaluation/`:
- `metrics.json` - All metrics
- `predictions.csv` - Per-image predictions
- `visualizations/` - ROC curves, confusion matrices, etc.

## Advanced Usage

### Extract Quantitative Features

```bash
python scripts/extract_quant_features.py \
  --checkpoint experiments/logs/resnet/best.pt \
  --split test \
  --output experiments/features_test.parquet \
  --config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml
```

### Single Image Inference with Grad-CAM

```bash
python scripts/infer_single_image.py \
  --image_path path/to/image.png \
  --checkpoint experiments/logs/resnet/best.pt \
  --output_dir experiments/inference \
  --config config/model_resnet.yaml
```

### Run Inference API

```bash
python src/cancer_quant_model/api/inference_api.py \
  experiments/logs/resnet/best.pt \
  config/model_resnet.yaml

# Test endpoint
curl -X POST "http://localhost:8000/predict" -F "file=@test.png"
```

## What's Included

### 📁 Project Structure

```
cancer_quant_model/
├── config/               # YAML configs (dataset, 3 models, train, eval)
├── src/                  # Main source code
│   ├── data/            # Dataset, transforms, data modules
│   ├── models/          # ResNet, EfficientNet, ViT + heads
│   ├── training/        # Training loops, callbacks, evaluator
│   ├── explainability/  # Grad-CAM implementation
│   ├── api/             # FastAPI server + batch inference
│   └── utils/           # Metrics, features, visualization, logging
├── scripts/             # CLI scripts (6 total)
├── tests/               # Unit tests
├── notebooks/           # 2 Jupyter notebooks for EDA
├── docs/                # Complete documentation
└── experiments/         # Logs, checkpoints, MLflow tracking
```

### 🧠 Model Architectures

1. **ResNet** (18, 34, 50, 101, 152)
   - Fast, proven architecture
   - Config: `config/model_resnet.yaml`

2. **EfficientNet** (B0-B7)
   - Best accuracy/efficiency tradeoff
   - Config: `config/model_efficientnet.yaml`

3. **Vision Transformer** (Tiny, Small, Base, Large)
   - Latest SOTA architecture
   - Config: `config/model_vit.yaml`

### 📊 Features Extracted

**Classic Features (100+):**
- Color: RGB, HSV, LAB statistics (mean, std, skew, kurtosis)
- Texture: Haralick, GLCM, Local Binary Patterns
- Morphology: Nuclei count, density, shape analysis

**Deep Features:**
- 512-2048 dimensional embeddings from model

### 📈 Metrics Tracked

- Accuracy, Precision, Recall, F1-Score
- AUC-ROC, AUC-PR
- Confusion Matrix
- Calibration (Brier score, ECE)
- Per-class metrics

### 🔬 Explainability

- **Grad-CAM**: Visualize which image regions influenced the prediction
- Supports ResNet and EfficientNet
- Generates heatmap overlays

## Key Commands

```bash
# Prepare data
python scripts/prepare_data.py --config config/dataset.yaml

# Create splits
python scripts/create_splits.py --config config/dataset.yaml

# Train model
python scripts/train.py \
  --model_config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml \
  --train_config config/train_default.yaml

# Evaluate
python scripts/evaluate.py \
  --checkpoint experiments/logs/resnet/best.pt \
  --config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml \
  --output_dir experiments/evaluation

# Extract features
python scripts/extract_quant_features.py \
  --checkpoint experiments/logs/resnet/best.pt \
  --split test \
  --output experiments/features_test.parquet

# Infer single image
python scripts/infer_single_image.py \
  --image_path path/to/image.png \
  --checkpoint experiments/logs/resnet/best.pt \
  --output_dir experiments/inference

# Run tests
pytest

# Start MLflow UI
mlflow ui --backend-store-uri experiments/mlruns
```

## Documentation

- **README.md** - Full documentation
- **docs/model_card.md** - Model details, intended use, limitations
- **docs/dataset_notes.md** - Dataset preparation guide
- **docs/experiments_guide.md** - Complete experiments workflow

## Troubleshooting

**Out of Memory:**
```yaml
# In config/model_*.yaml
training:
  batch_size: 16  # Reduce from 32
  use_amp: true   # Enable mixed precision
```

**Training Not Converging:**
```yaml
# In config/model_*.yaml
optimizer:
  lr: 1e-5  # Lower learning rate
```

**Class Imbalance:**
```yaml
# In config/dataset.yaml
split:
  stratify: true

# In config/model_*.yaml
loss:
  class_weights: [1.0, 2.0]  # Adjust based on imbalance
```

## Next Steps

1. ✅ **Read the docs**: Start with `README.md` and `docs/experiments_guide.md`
2. ✅ **Prepare your dataset**: Follow `docs/dataset_notes.md`
3. ✅ **Run first experiment**: Use commands above
4. ✅ **Compare models**: Train ResNet, EfficientNet, ViT and compare in MLflow
5. ✅ **Extract features**: Use quantitative features for downstream analysis
6. ✅ **Deploy**: Set up FastAPI server for production

## Support

- Issues: Check logs in `experiments/logs/`
- MLflow: View runs at http://localhost:5000
- Tests: Run `pytest -v` for detailed output

## Built With

- PyTorch 2.0+
- timm (PyTorch Image Models)
- Albumentations (augmentation)
- MLflow (experiment tracking)
- FastAPI (inference API)
- scikit-learn, pandas, numpy
- mahotas (texture features)

---

**Everything is ready to use!** Just add your dataset and start training.
