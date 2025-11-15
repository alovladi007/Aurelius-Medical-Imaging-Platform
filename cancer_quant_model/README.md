# Cancer Quantitative Histopathology Model

A production-ready, end-to-end quantitative cancer histopathology modeling pipeline using deep learning and classical computer vision techniques.

## Features

- **Multiple Model Architectures**: ResNet, EfficientNet, Vision Transformer (ViT)
- **Quantitative Feature Extraction**: Color statistics, texture (GLCM, LBP), morphological features, frequency domain features
- **Explainability**: Grad-CAM and Grad-CAM++ for visual explanations
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Config-Driven**: Flexible YAML-based configuration system
- **Production-Ready**: Mixed precision training, gradient accumulation, comprehensive metrics
- **Fully Reproducible**: Seed management, deterministic algorithms, experiment logging

## Project Structure

```
cancer_quant_model/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── pyproject.toml                     # Project dependencies and metadata
├── config/                            # Configuration files
│   ├── dataset.yaml                   # Dataset configuration
│   ├── model_resnet.yaml              # ResNet model config
│   ├── model_efficientnet.yaml        # EfficientNet model config
│   ├── model_vit.yaml                 # ViT model config
│   ├── train_default.yaml             # Training configuration
│   └── eval_default.yaml              # Evaluation configuration
├── data/                              # Data directory
│   ├── raw/                           # Raw Kaggle data (place here)
│   ├── processed/                     # Processed/tiled data
│   └── splits/                        # Train/val/test split CSVs
├── src/cancer_quant_model/            # Main source code
│   ├── config.py                      # Config management
│   ├── data/                          # Data handling
│   │   ├── dataset.py                 # Dataset classes
│   │   ├── transforms.py              # Data augmentations
│   │   └── datamodule.py              # DataModule wrapper
│   ├── models/                        # Model architectures
│   │   ├── resnet.py                  # ResNet implementation
│   │   ├── efficientnet.py            # EfficientNet implementation
│   │   ├── vit.py                     # ViT implementation
│   │   └── heads.py                   # Classification heads
│   ├── training/                      # Training & evaluation
│   │   ├── train_loop.py              # Training loop
│   │   └── eval_loop.py               # Evaluation loop
│   ├── explainability/                # Explainability methods
│   │   └── grad_cam.py                # Grad-CAM implementation
│   ├── api/                           # Inference API
│   │   └── inference_api.py           # Simple inference API
│   └── utils/                         # Utilities
│       ├── logging_utils.py           # Logging setup
│       ├── seed_utils.py              # Reproducibility
│       ├── metrics_utils.py           # Metrics computation
│       ├── viz_utils.py               # Visualization
│       ├── feature_utils.py           # Quantitative features
│       └── tiling_utils.py            # Image tiling
├── scripts/                           # Command-line scripts
│   ├── prepare_data.py                # Data preparation
│   ├── create_splits.py               # Create train/val/test splits
│   ├── train.py                       # Training script
│   ├── evaluate.py                    # Evaluation script
│   ├── infer_single_image.py          # Single image inference
│   └── extract_quant_features.py      # Extract quantitative features
├── experiments/                       # Experiment outputs
│   ├── mlruns/                        # MLflow tracking
│   ├── checkpoints/                   # Model checkpoints
│   ├── logs/                          # Training logs
│   └── outputs/                       # Other outputs
├── notebooks/                         # Jupyter notebooks
│   ├── EDA_dataset_overview.ipynb     # Dataset exploration
│   └── EDA_features_viz.ipynb         # Feature visualization
├── tests/                             # Unit tests
└── docs/                              # Documentation
```

## Installation

### Requirements

- Python 3.11+
- CUDA-capable GPU (recommended) or CPU

### Install Dependencies

```bash
cd cancer_quant_model

# Using pip
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"

# Or for notebook support
pip install -e ".[notebooks]"
```

## Quick Start

### 1. Prepare Your Data

Place your Kaggle histopathology dataset in one of two formats:

**Option A: Folder Binary Structure**
```
data/raw/train/
  ├── 0/  (non-cancer images)
  │   ├── image1.png
  │   ├── image2.png
  │   └── ...
  └── 1/  (cancer images)
      ├── image1.png
      ├── image2.png
      └── ...
```

**Option B: CSV Labels**
```
data/raw/
  ├── images/
  │   ├── image1.png
  │   ├── image2.png
  │   └── ...
  └── labels.csv  (columns: image_id, label)
```

### 2. Configure Dataset

Edit `config/dataset.yaml` to match your dataset:

```yaml
dataset:
  type: "folder_binary"  # or "csv_labels"
  paths:
    raw_data_dir: "data/raw"
    # ... other paths
```

### 3. Create Data Splits

```bash
python scripts/create_splits.py --config config/dataset.yaml
```

This creates `train.csv`, `val.csv`, `test.csv` in `data/splits/`.

### 4. Train a Model

```bash
# Train with ResNet-50
python scripts/train.py \
    --dataset-config config/dataset.yaml \
    --model-config config/model_resnet.yaml \
    --train-config config/train_default.yaml

# Train with EfficientNet-B0
python scripts/train.py \
    --model-config config/model_efficientnet.yaml

# Train with Vision Transformer
python scripts/train.py \
    --model-config config/model_vit.yaml
```

### 5. Monitor Training

```bash
# View MLflow UI
mlflow ui --backend-store-uri experiments/mlruns
# Open http://localhost:5000 in your browser
```

### 6. Evaluate Model

```bash
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/best_model.pt \
    --dataset-config config/dataset.yaml \
    --model-config config/model_resnet.yaml \
    --output-dir experiments/eval_results
```

### 7. Run Inference

```bash
# Single image inference with Grad-CAM
python scripts/infer_single_image.py \
    --image path/to/image.png \
    --checkpoint experiments/checkpoints/best_model.pt \
    --model-config config/model_resnet.yaml \
    --save-gradcam
```

### 8. Extract Quantitative Features

```bash
python scripts/extract_quant_features.py \
    --input-dir data/raw/train \
    --output experiments/features/features.parquet
```

## Usage Examples

### Training with Custom Config

```python
from pathlib import Path
import pandas as pd
from cancer_quant_model.data.datamodule import HistopathDataModule
from cancer_quant_model.models.resnet import ResNetModel
from cancer_quant_model.training.train_loop import Trainer
import torch.nn as nn
import torch.optim as optim

# Load data
train_df = pd.read_csv("data/splits/train.csv")
val_df = pd.read_csv("data/splits/val.csv")

# Create data module
datamodule = HistopathDataModule(
    train_df=train_df,
    val_df=val_df,
    batch_size=32,
    num_workers=4,
)

# Build model
model = ResNetModel(variant="resnet50", num_classes=2, pretrained=True)

# Setup training
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Train
trainer = Trainer(
    model=model,
    train_loader=datamodule.train_dataloader(),
    val_loader=datamodule.val_dataloader(),
    criterion=criterion,
    optimizer=optimizer,
    max_epochs=50,
)
trainer.train()
```

### Inference API

```python
from cancer_quant_model.api.inference_api import InferenceAPI
from cancer_quant_model.models.resnet import ResNetModel

# Load model
model = ResNetModel(variant="resnet50", num_classes=2)

# Create API
api = InferenceAPI.from_checkpoint(
    checkpoint_path="experiments/checkpoints/best_model.pt",
    model=model,
    class_names=["non_cancer", "cancer"],
)

# Run inference
result = api.predict(
    "path/to/image.png",
    return_features=True,
    return_gradcam=True,
)

print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: {result['probabilities']}")
```

### Extract Quantitative Features

```python
import numpy as np
from PIL import Image
from cancer_quant_model.utils.feature_utils import QuantitativeFeatureExtractor

# Load image
image = np.array(Image.open("image.png"))

# Extract features
extractor = QuantitativeFeatureExtractor()
features = extractor.extract_all_features(image)

# Features include:
# - Color statistics (RGB, HSV, LAB)
# - Texture features (GLCM, LBP)
# - Morphological features
# - Frequency domain features

print(f"Extracted {len(features)} features")
```

## Configuration

### Dataset Configuration (`config/dataset.yaml`)

Key settings:
- `dataset.type`: "folder_binary" or "csv_labels"
- `image.target_size`: Input image size [224, 224]
- `split.{train,val,test}_ratio`: Data split ratios
- `augmentation.*`: Data augmentation settings
- `dataloader.batch_size`: Batch size

### Model Configuration

Each model has its own config file. Key settings:
- `model.variant`: Specific model variant (e.g., resnet50, efficientnet_b0)
- `model.pretrained`: Use ImageNet pretrained weights
- `head.num_classes`: Number of output classes
- `head.dropout`: Dropout rate
- `optimizer.*`: Optimizer settings
- `scheduler.*`: Learning rate scheduler settings
- `training.max_epochs`: Maximum training epochs

### Training Configuration (`config/train_default.yaml`)

Key settings:
- `training.max_epochs`: Maximum epochs
- `training.precision`: "16-mixed" for mixed precision
- `training.early_stopping.patience`: Early stopping patience
- `hardware.device`: "auto", "cuda", or "cpu"
- `experiment.mlflow.*`: MLflow tracking settings

## Advanced Features

### Tiling Large Images

If your images are large whole-slide images, enable tiling:

```yaml
# In config/dataset.yaml
dataset:
  image:
    create_patches: true
    patch_size: 512
    patch_overlap: 0
    min_tissue_ratio: 0.5
```

Then run:
```bash
python scripts/prepare_data.py --config config/dataset.yaml --create-tiles
```

### Custom Augmentation

Modify `config/dataset.yaml`:

```yaml
dataset:
  augmentation:
    horizontal_flip: true
    vertical_flip: true
    rotation_degrees: 90
    brightness: 0.2
    contrast: 0.2
    gaussian_blur: 0.1
    elastic_transform: true
```

### Mixed Precision Training

Automatically enabled with:
```yaml
# In config/train_default.yaml
training:
  precision: "16-mixed"  # Use mixed precision
```

### Gradient Accumulation

For larger effective batch sizes:
```yaml
# In config/train_default.yaml
training:
  accumulate_grad_batches: 4  # Effective batch = 4 * batch_size
```

## Testing

Run tests:
```bash
# All tests
pytest tests/

# Specific test
pytest tests/test_dataset.py

# With coverage
pytest --cov=cancer_quant_model --cov-report=html
```

## Experiment Tracking

All experiments are tracked with MLflow:

- **Metrics**: Loss, accuracy, precision, recall, F1, AUROC, AUPRC
- **Parameters**: Learning rate, batch size, model architecture
- **Artifacts**: Model checkpoints, config files, plots
- **System Metrics**: CPU/GPU usage, memory

View experiments:
```bash
mlflow ui --backend-store-uri experiments/mlruns
```

## Model Architectures

### ResNet (Residual Networks)
- Variants: ResNet-18, 34, 50, 101, 152
- Best for: General-purpose, proven performance
- Config: `config/model_resnet.yaml`

### EfficientNet
- Variants: EfficientNet-B0 to B7
- Best for: Efficiency, mobile deployment
- Config: `config/model_efficientnet.yaml`

### Vision Transformer (ViT)
- Variants: ViT-Tiny, Small, Base, Large
- Best for: State-of-the-art performance, large datasets
- Config: `config/model_vit.yaml`

## Quantitative Features

Extracted features include:

**Color Features**
- Per-channel statistics (mean, std, median, quartiles)
- HSV and LAB color space features

**Texture Features**
- GLCM (Gray-Level Co-occurrence Matrix)
- Local Binary Patterns (LBP)
- Contrast, dissimilarity, homogeneity, energy, correlation

**Morphological Features**
- Object count, area, perimeter
- Eccentricity, solidity, circularity
- Edge density

**Frequency Features**
- FFT-based features
- Power in low/mid/high frequency bands

## Troubleshooting

### Out of Memory

1. Reduce batch size in `config/dataset.yaml`
2. Enable gradient accumulation in `config/train_default.yaml`
3. Use smaller model variant

### Slow Training

1. Increase `num_workers` in dataset config
2. Enable mixed precision training
3. Use smaller image size

### Poor Performance

1. Increase training epochs
2. Adjust learning rate
3. Try different model architectures
4. Check for class imbalance (automatic class weighting available)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cancer_quant_model,
  title = {Cancer Quantitative Histopathology Model},
  author = {Aurelius Medical Imaging Platform},
  year = {2025},
  license = {MIT}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation in `docs/`

## Acknowledgments

- Built with PyTorch, timm, albumentations, MLflow
- Inspired by state-of-the-art medical imaging research
