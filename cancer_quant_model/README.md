# Cancer Quantitative Histopathology Model

A production-ready quantitative cancer research model that uses histopathology tissue slide images for deep learning-based classification, feature extraction, and explainability.

## Features

- **Multiple Model Architectures**: ResNet, EfficientNet, and Vision Transformer (ViT)
- **Quantitative Feature Extraction**: Classic image features (color, texture, morphology) + deep embeddings
- **Explainability**: Grad-CAM visualization for model interpretability
- **Experiment Tracking**: MLflow integration for reproducible experiments
- **Production API**: FastAPI-based REST API for inference
- **Comprehensive Metrics**: AUC-ROC, PR-AUC, calibration, confusion matrices, per-class metrics

## Project Structure

```
cancer_quant_model/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── config/
│   ├── dataset.yaml
│   ├── model_resnet.yaml
│   ├── model_efficientnet.yaml
│   ├── model_vit.yaml
│   ├── train_default.yaml
│   └── eval_default.yaml
├── data/
│   ├── raw/              # Place your Kaggle dataset here
│   ├── processed/
│   └── splits/
├── src/cancer_quant_model/
│   ├── config.py
│   ├── utils/
│   ├── data/
│   ├── models/
│   ├── training/
│   ├── explainability/
│   └── api/
├── experiments/
│   ├── logs/
│   └── mlruns/
├── notebooks/
├── scripts/
│   ├── prepare_data.py
│   ├── create_splits.py
│   ├── train.py
│   ├── evaluate.py
│   ├── infer_single_image.py
│   └── extract_quant_features.py
├── tests/
└── docs/
```

## Installation

### 1. Create Virtual Environment

```bash
cd cancer_quant_model
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or with development dependencies:

```bash
pip install -e ".[dev,jupyter]"
```

## Quick Start

### 1. Prepare Your Dataset

Place your Kaggle histopathology dataset in `data/raw/` in one of these formats:

**Option A: Folder Structure**
```
data/raw/train/
├── 0/  # Non-cancer images
└── 1/  # Cancer images
```

**Option B: CSV Labels**
```
data/raw/images/  # All images
data/raw/labels.csv  # CSV with columns: image_id, label
```

### 2. Configure Dataset

Edit `config/dataset.yaml` to match your dataset structure.

### 3. Prepare Data and Create Splits

```bash
# Prepare and preprocess data
python scripts/prepare_data.py --config config/dataset.yaml

# Create train/val/test splits
python scripts/create_splits.py --config config/dataset.yaml
```

### 4. Train a Model

Train ResNet-50:
```bash
python scripts/train.py \
  --model_config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml \
  --train_config config/train_default.yaml
```

Train EfficientNet-B3:
```bash
python scripts/train.py \
  --model_config config/model_efficientnet.yaml \
  --dataset_config config/dataset.yaml \
  --train_config config/train_default.yaml
```

### 5. Evaluate Model

```bash
python scripts/evaluate.py \
  --checkpoint experiments/logs/resnet/best.pt \
  --config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml \
  --output_dir experiments/logs/evaluation
```

### 6. Extract Quantitative Features

```bash
python scripts/extract_quant_features.py \
  --checkpoint experiments/logs/resnet/best.pt \
  --split test \
  --output experiments/features_test.parquet \
  --config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml
```

### 7. Run Inference on Single Image

```bash
python scripts/infer_single_image.py \
  --image_path path/to/image.png \
  --checkpoint experiments/logs/resnet/best.pt \
  --output_dir experiments/inference \
  --config config/model_resnet.yaml
```

### 8. Start Inference API

```bash
# Load model and start server
python src/cancer_quant_model/api/inference_api.py \
  experiments/logs/resnet/best.pt \
  config/model_resnet.yaml

# Or use uvicorn directly
uvicorn cancer_quant_model.api.inference_api:app \
  --host 0.0.0.0 \
  --port 8000
```

Test the API:
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/image.png"
```

## Experiment Tracking with MLflow

Start MLflow UI to view experiments:

```bash
mlflow ui --backend-store-uri experiments/mlruns --host 0.0.0.0 --port 5000
```

Access at: http://localhost:5000

## Configuration

### Dataset Configuration (`config/dataset.yaml`)

- `dataset_type`: "folder_binary" or "csv_labels"
- `split`: Train/val/test ratios and stratification
- `image_settings`: Target size, augmentation, tiling options
- `classes`: Number of classes and class names

### Model Configuration (`config/model_*.yaml`)

- `model`: Architecture, backbone, pretrained weights
- `head`: Dropout, hidden layers, activation
- `optimizer`: Learning rate, weight decay
- `scheduler`: LR scheduling strategy
- `loss`: Loss function and label smoothing
- `training`: Batch size, epochs, mixed precision

### Training Configuration (`config/train_default.yaml`)

- `seed`: Random seed for reproducibility
- `augmentation`: Data augmentation strategies
- `preprocessing`: Image normalization parameters
- `metrics`: Metrics to track during training

## Model Architectures

### ResNet (config/model_resnet.yaml)
- Backbones: resnet18, resnet34, resnet50, resnet101, resnet152
- Pretrained on ImageNet
- Customizable classification head

### EfficientNet (config/model_efficientnet.yaml)
- Backbones: efficientnet_b0 through efficientnet_b7
- State-of-the-art efficiency
- Compound scaling

### Vision Transformer (config/model_vit.yaml)
- Backbones: vit_tiny, vit_small, vit_base, vit_large
- Transformer-based architecture
- Attention mechanisms

## Quantitative Features

### Classic Features
- **Color Statistics**: Mean, std, skew, kurtosis for RGB, HSV, LAB channels
- **Texture Features**: Haralick features, GLCM properties, Local Binary Patterns (LBP)
- **Morphological Features**: Nuclei detection, shape analysis, density metrics

### Deep Features
- Extracted from model's penultimate layer
- High-dimensional embeddings (512-2048 dimensions)
- Can be used for downstream ML tasks

## Explainability

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- Visualizes which regions the model focuses on
- Supports ResNet and EfficientNet
- Generates heatmap overlays on original images

## Testing

Run tests:
```bash
pytest
```

With coverage:
```bash
pytest --cov=cancer_quant_model --cov-report=html
```

## Documentation

- `docs/model_card.md`: Model documentation and intended use
- `docs/dataset_notes.md`: Dataset preparation instructions
- `docs/experiments_guide.md`: Guide to running experiments

## Examples

See Jupyter notebooks in `notebooks/`:
- `EDA_dataset_overview.ipynb`: Dataset exploration and statistics
- `EDA_features_viz.ipynb`: Feature visualization and analysis

## Development

### Code Quality

Format code:
```bash
black src/ scripts/ tests/
isort src/ scripts/ tests/
```

Lint:
```bash
flake8 src/ scripts/ tests/
mypy src/
```

## License

MIT License - see LICENSE file

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cancer_quant_model,
  title = {Cancer Quantitative Histopathology Model},
  year = {2025},
  author = {Research Team},
  license = {MIT}
}
```

## Support

For issues and questions:
- GitHub Issues: [Create an issue](https://github.com/your-org/cancer-quant-model/issues)
- Documentation: See `docs/` directory

## Acknowledgments

Built with PyTorch, timm, albumentations, MLflow, FastAPI, and scikit-learn.
