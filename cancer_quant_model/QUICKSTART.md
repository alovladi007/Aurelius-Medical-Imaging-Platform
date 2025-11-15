# Quick Start Guide

This guide will get you up and running with the cancer histopathology model in 10 minutes.

## Prerequisites

- Python 3.11+
- A Kaggle histopathology dataset downloaded

## Step 1: Installation (2 minutes)

```bash
# Clone the repository (if not already)
cd cancer_quant_model

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Verify installation
python -c "import cancer_quant_model; print('Installation successful!')"
```

## Step 2: Prepare Your Data (3 minutes)

### Option A: Folder Structure (Recommended)

Organize your Kaggle dataset like this:

```
cancer_quant_model/data/raw/train/
  ├── 0/  # Non-cancer images
  │   ├── image001.png
  │   ├── image002.png
  │   └── ...
  └── 1/  # Cancer images
      ├── image001.png
      ├── image002.png
      └── ...
```

### Option B: CSV Labels

Place images in `data/raw/images/` and create `data/raw/labels.csv`:

```csv
image_id,label
image001.png,0
image002.png,1
...
```

Then update `config/dataset.yaml`:
```yaml
dataset:
  type: "csv_labels"  # Change from "folder_binary"
  paths:
    images_folder: "data/raw/images"
    labels_csv: "data/raw/labels.csv"
```

## Step 3: Create Data Splits (30 seconds)

```bash
python scripts/create_splits.py --config config/dataset.yaml
```

This creates:
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

**Verify**: Check that splits exist:
```bash
ls data/splits/
# Should see: train.csv  val.csv  test.csv
```

## Step 4: Train Your First Model (5 minutes)

### Quick Train (ResNet-50, 10 epochs)

```bash
python scripts/train.py \
    --dataset-config config/dataset.yaml \
    --model-config config/model_resnet.yaml \
    --train-config config/train_default.yaml
```

**What happens:**
- Model trains for epochs specified in config (default: 50, reduce for quick test)
- Metrics logged to MLflow
- Checkpoints saved to `experiments/checkpoints/`
- Best model saved as `experiments/checkpoints/best.pt`

**To train faster (for testing):**

Edit `config/model_resnet.yaml`:
```yaml
model:
  training:
    max_epochs: 10  # Reduce from 50
```

## Step 5: Monitor Training

Open a new terminal:

```bash
mlflow ui --backend-store-uri experiments/mlruns --port 5000
```

Open browser: `http://localhost:5000`

You'll see:
- Loss curves
- Accuracy, AUROC, F1 scores
- All hyperparameters
- Model checkpoints

## Step 6: Evaluate Your Model

```bash
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/best.pt \
    --dataset-config config/dataset.yaml \
    --model-config config/model_resnet.yaml \
    --output-dir experiments/eval_results
```

**Output:**
- `experiments/eval_results/predictions.csv` - All predictions
- `experiments/eval_results/confusion_matrix.png` - Visual confusion matrix
- `experiments/eval_results/roc_curve.png` - ROC curve
- Console output with all metrics

## Step 7: Run Inference on Single Image

```bash
python scripts/infer_single_image.py \
    --image data/raw/train/1/some_image.png \
    --checkpoint experiments/checkpoints/best.pt \
    --model-config config/model_resnet.yaml \
    --save-gradcam \
    --output-dir experiments/inference
```

**Output:**
- Console: Predicted class and probabilities
- `experiments/inference/some_image_gradcam.png` - Grad-CAM visualization

## Step 8: Extract Quantitative Features

```bash
python scripts/extract_quant_features.py \
    --input-dir data/raw/train \
    --output experiments/features/features.parquet
```

**Output:**
- `experiments/features/features.parquet` - 100+ features per image
  - Color statistics
  - Texture features (GLCM, LBP)
  - Morphological features
  - Frequency domain features

## Next Steps

### Run Different Models

**EfficientNet (lighter, faster):**
```bash
python scripts/train.py --model-config config/model_efficientnet.yaml
```

**Vision Transformer (best accuracy, slower):**
```bash
python scripts/train.py --model-config config/model_vit.yaml
```

### Explore Your Data

```bash
# Start Jupyter
jupyter lab

# Open notebooks/EDA_dataset_overview.ipynb
# Analyze class distribution, image statistics
```

### Batch Inference

```bash
# On a CSV of images
python -m cancer_quant_model.api.batch_inference \
    --input data/splits/test.csv \
    --output experiments/batch_predictions.csv \
    --checkpoint experiments/checkpoints/best.pt \
    --model-config config/model_resnet.yaml

# On a directory
python -m cancer_quant_model.api.batch_inference \
    --input data/raw/test_images/ \
    --output experiments/batch_predictions.csv \
    --checkpoint experiments/checkpoints/best.pt \
    --save-gradcam
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific test
pytest tests/test_dataset.py -v

# With coverage
pytest --cov=cancer_quant_model --cov-report=html
```

## Troubleshooting

### "No module named 'cancer_quant_model'"

```bash
# Reinstall in editable mode
pip install -e .
```

### "CUDA out of memory"

Reduce batch size in `config/dataset.yaml`:
```yaml
dataset:
  dataloader:
    batch_size: 16  # Reduce from 32
```

Or use gradient accumulation in `config/train_default.yaml`:
```yaml
training:
  accumulate_grad_batches: 2  # Effective batch = 16 * 2 = 32
```

### "File not found" errors

Check your data paths in `config/dataset.yaml`:
```yaml
dataset:
  paths:
    raw_data_dir: "data/raw"  # Make sure this exists
```

### Training is slow

1. Use smaller model variant:
   ```yaml
   model:
     variant: "resnet18"  # Instead of resnet50
   ```

2. Reduce image size:
   ```yaml
   dataset:
     image:
       target_size: [128, 128]  # Instead of [224, 224]
   ```

3. Use fewer workers (if CPU bottleneck):
   ```yaml
   dataset:
     dataloader:
       num_workers: 2  # Reduce from 4
   ```

## Common Workflows

### Hyperparameter Tuning

1. Copy config:
   ```bash
   cp config/train_default.yaml config/train_experiment1.yaml
   ```

2. Edit hyperparameters:
   ```yaml
   model:
     optimizer:
       lr: 0.0001  # Try different learning rate
   ```

3. Train:
   ```bash
   python scripts/train.py --train-config config/train_experiment1.yaml
   ```

4. Compare in MLflow UI

### Production Deployment

1. Train best model
2. Evaluate thoroughly
3. Use InferenceAPI:

```python
from cancer_quant_model.api.inference_api import InferenceAPI
from cancer_quant_model.models.resnet import ResNetModel

# Load model
model = ResNetModel(variant="resnet50", num_classes=2)
api = InferenceAPI.from_checkpoint(
    checkpoint_path="experiments/checkpoints/best.pt",
    model=model,
    class_names=["non_cancer", "cancer"],
)

# Predict
result = api.predict("path/to/image.png")
print(f"Prediction: {result['predicted_label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Support

- **Documentation**: See `docs/` directory
- **Issues**: Open a GitHub issue
- **Examples**: Check `notebooks/` for analysis examples

## What's Next?

- Read the full [README.md](README.md)
- Explore [docs/experiments_guide.md](docs/experiments_guide.md)
- Check [docs/model_card.md](docs/model_card.md) for model details
- Try different architectures and hyperparameters!

Happy modeling! 🔬🧬
