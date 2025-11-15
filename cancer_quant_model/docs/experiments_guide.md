# Experiments Guide

Complete guide to running training experiments, evaluating models, and analyzing results.

## Quick Start Experiment

### 1. Minimal Working Example

```bash
# Navigate to project
cd cancer_quant_model

# Activate environment
source .venv/bin/activate

# Prepare data (assuming you have data in data/raw/)
python scripts/prepare_data.py --config config/dataset.yaml

# Create splits
python scripts/create_splits.py --config config/dataset.yaml

# Train ResNet-50 for 10 epochs (quick test)
python scripts/train.py \
  --model_config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml \
  --train_config config/train_default.yaml

# Check MLflow
mlflow ui --backend-store-uri experiments/mlruns
```

## Full Experiment Workflow

### Step 1: Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import timm; import albumentations; print('OK')"
```

### Step 2: Dataset Preparation

See `dataset_notes.md` for detailed instructions.

```bash
# Configure dataset (edit config/dataset.yaml first)
python scripts/prepare_data.py --config config/dataset.yaml
python scripts/create_splits.py --config config/dataset.yaml

# Verify splits
ls -lh data/splits/
head data/splits/train.csv
```

### Step 3: Configure Experiment

Edit model config (e.g., `config/model_resnet.yaml`):

```yaml
model:
  backbone: "resnet50"
  num_classes: 2

training:
  batch_size: 32
  num_epochs: 50

early_stopping:
  patience: 10
  monitor: "val_auc"

checkpoint:
  save_top_k: 3
  monitor: "val_auc"
```

### Step 4: Train Model

```bash
# Train with default configs
python scripts/train.py \
  --model_config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml \
  --train_config config/train_default.yaml
```

**Training Output:**
- Checkpoints: `experiments/logs/resnet/`
- MLflow logs: `experiments/mlruns/`
- Training logs: `experiments/logs/train.log`

### Step 5: Monitor Training

**Option A: MLflow UI**
```bash
mlflow ui --backend-store-uri experiments/mlruns --port 5000
# Visit http://localhost:5000
```

**Option B: Check logs**
```bash
tail -f experiments/logs/train.log
```

### Step 6: Evaluate Model

```bash
python scripts/evaluate.py \
  --checkpoint experiments/logs/resnet/best.pt \
  --config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml \
  --output_dir experiments/logs/evaluation
```

**Evaluation Outputs:**
- `experiments/logs/evaluation/metrics.json`
- `experiments/logs/evaluation/predictions.csv`
- `experiments/logs/evaluation/visualizations/`
  - `confusion_matrix.png`
  - `roc_curve.png`
  - `pr_curve.png`
  - `calibration_curve.png`

### Step 7: Extract Features

```bash
python scripts/extract_quant_features.py \
  --checkpoint experiments/logs/resnet/best.pt \
  --split test \
  --output experiments/features_test.parquet \
  --config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml
```

### Step 8: Analyze Features

Use Jupyter notebook:
```bash
jupyter notebook notebooks/EDA_features_viz.ipynb
```

## Model Comparison Experiments

### Train Multiple Architectures

```bash
# ResNet-50
python scripts/train.py \
  --model_config config/model_resnet.yaml \
  --dataset_config config/dataset.yaml \
  --train_config config/train_default.yaml

# EfficientNet-B3
python scripts/train.py \
  --model_config config/model_efficientnet.yaml \
  --dataset_config config/dataset.yaml \
  --train_config config/train_default.yaml

# Vision Transformer
python scripts/train.py \
  --model_config config/model_vit.yaml \
  --dataset_config config/dataset.yaml \
  --train_config config/train_default.yaml
```

### Compare Results in MLflow

1. Start MLflow UI: `mlflow ui --backend-store-uri experiments/mlruns`
2. Navigate to experiment
3. Select multiple runs
4. Click "Compare"
5. View metric charts and parameter differences

## Hyperparameter Tuning

### Manual Grid Search

Create configs for different hyperparameters:

```bash
# config/hp_lr_1e-3.yaml
optimizer:
  lr: 1e-3

# config/hp_lr_1e-4.yaml
optimizer:
  lr: 1e-4

# Train with each
for config in config/hp_*.yaml; do
  python scripts/train.py \
    --model_config $config \
    --dataset_config config/dataset.yaml \
    --train_config config/train_default.yaml
done
```

### Key Hyperparameters to Tune

1. **Learning Rate:** 1e-5, 3e-5, 1e-4, 3e-4, 1e-3
2. **Batch Size:** 16, 24, 32, 48, 64
3. **Augmentation Strength:** light, medium, strong
4. **Dropout:** 0.1, 0.2, 0.3, 0.4, 0.5
5. **Weight Decay:** 1e-5, 1e-4, 1e-3

## Advanced Techniques

### Transfer Learning from Custom Checkpoint

```python
import torch
from cancer_quant_model.models import create_resnet_model

# Load pretrained model
checkpoint = torch.load("path/to/pretrained.pt")
config = checkpoint["config"]

# Create model and load weights
model = create_resnet_model(config)
model.load_state_dict(checkpoint["model_state_dict"])

# Fine-tune on new dataset
# ... (adjust configs and train)
```

### Ensemble Methods

```python
import torch
import numpy as np

# Load multiple models
models = []
for checkpoint_path in ["model1.pt", "model2.pt", "model3.pt"]:
    checkpoint = torch.load(checkpoint_path)
    model = create_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    models.append(model)

# Ensemble prediction
def ensemble_predict(image_tensor):
    probs_list = []
    for model in models:
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits, dim=1)
            probs_list.append(probs)
    
    # Average probabilities
    ensemble_probs = torch.mean(torch.stack(probs_list), dim=0)
    return ensemble_probs
```

### Cross-Validation

Create multiple splits manually:

```bash
# Modify config/dataset.yaml for each fold
for fold in 1 2 3 4 5; do
  # Update split seed
  sed -i "s/seed: .*/seed: $((42 + fold))/" config/dataset.yaml
  
  # Create new splits
  python scripts/create_splits.py --config config/dataset.yaml
  
  # Train
  python scripts/train.py \
    --model_config config/model_resnet.yaml \
    --dataset_config config/dataset.yaml \
    --train_config config/train_default.yaml
done
```

## Best Practices

### 1. Reproducibility

```yaml
# In config/train_default.yaml
seed: 42
training:
  deterministic: true  # Slower but fully reproducible
  benchmark: false     # Don't use cudnn.benchmark
```

### 2. Experiment Naming

Use MLflow tags:

```yaml
mlflow:
  tags:
    experiment_type: "baseline"
    dataset: "pcam"
    notes: "Initial run with default params"
```

### 3. Save Everything

```python
# In training script
import yaml

# Save all configs
with open("experiments/logs/run_config.yaml", "w") as f:
    yaml.dump({
        "model": model_config,
        "dataset": dataset_config,
        "train": train_config,
    }, f)
```

### 4. Monitor GPU Usage

```bash
# During training
watch -n 1 nvidia-smi

# Or use gpustat
pip install gpustat
gpustat -cpi
```

## Troubleshooting

### Out of Memory

**Solutions:**
1. Reduce batch size
2. Use gradient accumulation
3. Use mixed precision (AMP)
4. Use smaller model variant
5. Reduce image size

```yaml
training:
  batch_size: 16  # Reduce from 32
  accumulate_grad_batches: 2  # Effective batch size = 32
  use_amp: true  # Mixed precision
```

### Training Not Converging

**Check:**
1. Learning rate too high/low
2. Data augmentation too aggressive
3. Class imbalance
4. Data quality issues

**Solutions:**
```yaml
optimizer:
  lr: 1e-5  # Lower learning rate

augmentation:
  train:
    strength: "light"  # Reduce augmentation

loss:
  class_weights: [1.0, 2.0]  # Handle imbalance
```

### Overfitting

**Indicators:**
- Val loss increases while train loss decreases
- Large gap between train and val metrics

**Solutions:**
1. More data augmentation
2. Higher dropout
3. More regularization
4. Early stopping

```yaml
head:
  dropout: 0.5  # Increase from 0.3

optimizer:
  weight_decay: 1e-3  # Increase regularization

early_stopping:
  patience: 5  # Stop earlier
```

## Performance Optimization

### Data Loading

```yaml
training:
  num_workers: 8  # Increase if CPU bottleneck
  pin_memory: true  # Faster GPU transfer
```

### Mixed Precision

```yaml
training:
  use_amp: true  # ~2x speedup on modern GPUs
```

### Gradient Checkpointing

For very large models:

```python
# In model definition
torch.utils.checkpoint.checkpoint(layer, x)
```

## Deployment

### Export to ONNX

```python
import torch

model.eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch_size"}},
)
```

### Run Inference API

```bash
# Start server
python src/cancer_quant_model/api/inference_api.py \
  experiments/logs/resnet/best.pt \
  config/model_resnet.yaml

# Test endpoint
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.png"
```

## Resources

- **MLflow Docs:** https://www.mlflow.org/docs/latest/index.html
- **PyTorch Training:** https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
- **Timm Models:** https://github.com/huggingface/pytorch-image-models

## Getting Help

1. Check logs: `experiments/logs/train.log`
2. Review MLflow runs for insights
3. Test on small subset first
4. Use `fast_dev_run=true` for debugging
