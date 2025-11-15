# Experiments Guide

This guide covers running experiments, tracking results, and analyzing model performance.

## Quick Reference

```bash
# Create data splits
python scripts/create_splits.py --config config/dataset.yaml

# Train model
python scripts/train.py \
    --dataset-config config/dataset.yaml \
    --model-config config/model_resnet.yaml \
    --train-config config/train_default.yaml

# Evaluate model
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/best_model.pt \
    --dataset-config config/dataset.yaml \
    --model-config config/model_resnet.yaml

# Run inference
python scripts/infer_single_image.py \
    --image path/to/image.png \
    --checkpoint experiments/checkpoints/best_model.pt \
    --model-config config/model_resnet.yaml

# Extract features
python scripts/extract_quant_features.py \
    --input-dir data/raw/train \
    --output experiments/features/features.parquet
```

## Experiment Tracking with MLflow

### View Experiments

Start MLflow UI:
```bash
mlflow ui --backend-store-uri experiments/mlruns --port 5000
```

Open browser to `http://localhost:5000`

### What's Tracked

**Metrics**:
- Training: loss, accuracy, precision, recall, f1
- Validation: loss, accuracy, precision, recall, f1, auroc, auprc
- Per-epoch values

**Parameters**:
- Model architecture and variant
- Learning rate, optimizer, scheduler
- Batch size, number of epochs
- Augmentation settings

**Artifacts**:
- Model checkpoints
- Configuration files
- Training plots
- Confusion matrices

**System Metrics** (if enabled):
- CPU usage
- GPU usage and memory
- Disk I/O

### Compare Experiments

In MLflow UI:
1. Select multiple runs
2. Click "Compare"
3. View metric plots, parallel coordinates
4. Download comparison data

## Running Different Model Architectures

### ResNet

```bash
# ResNet-18 (faster, fewer parameters)
python scripts/train.py --model-config config/model_resnet.yaml

# Edit config for other variants:
# model.variant: resnet18, resnet34, resnet50, resnet101, resnet152
```

**When to use**:
- Good baseline model
- Proven performance
- Fast training

### EfficientNet

```bash
python scripts/train.py --model-config config/model_efficientnet.yaml

# Variants: efficientnet_b0 to efficientnet_b7
```

**When to use**:
- Better accuracy per parameter
- Resource-constrained environments
- Mobile/edge deployment

### Vision Transformer (ViT)

```bash
python scripts/train.py --model-config config/model_vit.yaml

# Variants: vit_tiny_patch16_224, vit_small, vit_base, vit_large
```

**When to use**:
- State-of-the-art performance
- Large datasets (1000+ samples per class)
- Sufficient compute resources

## Hyperparameter Tuning

### Learning Rate

Test different learning rates:

```yaml
# In config/model_*.yaml
model:
  optimizer:
    lr: 0.0001  # Try: 0.0001, 0.0003, 0.001, 0.003
```

**Rule of thumb**:
- Start with 0.001
- Decrease if loss oscillates
- Increase if loss decreases too slowly

### Batch Size

```yaml
# In config/dataset.yaml
dataset:
  dataloader:
    batch_size: 32  # Try: 16, 32, 64, 128
```

**Trade-offs**:
- Larger batch = faster training, more memory
- Smaller batch = better generalization, slower training
- Use gradient accumulation for effective large batches

### Data Augmentation

```yaml
# In config/dataset.yaml
dataset:
  augmentation:
    horizontal_flip: true
    vertical_flip: true
    rotation_degrees: 90
    brightness: 0.2  # Try: 0.1, 0.2, 0.3
    contrast: 0.2
```

**Guidelines**:
- More augmentation = better generalization, slower training
- Start conservative, increase if overfitting
- Histopathology specific: rotation, flips are safe

## Advanced Training Techniques

### Mixed Precision Training

Automatically enabled:
```yaml
# In config/train_default.yaml
training:
  precision: "16-mixed"  # 16-bit mixed precision
```

**Benefits**:
- 2-3x faster training
- ~50% less memory
- Minimal accuracy loss

### Gradient Accumulation

For larger effective batch sizes:
```yaml
training:
  accumulate_grad_batches: 4  # Effective batch = batch_size * 4
```

**When to use**:
- GPU memory limited
- Want large batch sizes
- Training large models

### Learning Rate Scheduling

```yaml
# Cosine annealing (recommended)
model:
  scheduler:
    type: "cosine"
    warmup_epochs: 5
    min_lr: 0.000001

# Step decay
model:
  scheduler:
    type: "step"
    step_size: 10
    gamma: 0.1

# Reduce on plateau
model:
  scheduler:
    type: "plateau"
    patience: 5
    factor: 0.5
```

### Early Stopping

```yaml
# In config/train_default.yaml
training:
  early_stopping:
    enabled: true
    monitor: "val_auroc"  # or val_loss, val_accuracy
    patience: 10
    mode: "max"  # "max" for metrics to maximize, "min" for loss
```

## Handling Class Imbalance

### Automatic Class Weighting

Automatically computed from training data:
```python
# In training script, class weights are computed:
class_weights = datamodule.get_class_weights()
```

### Manual Class Weights

```yaml
# In config/dataset.yaml
dataset:
  labels:
    class_weights: [1.0, 2.5]  # Weight for each class
```

### Focal Loss

For severe imbalance:
```yaml
# In config/model_*.yaml
model:
  loss:
    type: "focal"
    focal_alpha: [0.25, 0.75]  # Per-class weights
    focal_gamma: 2.0  # Focusing parameter
```

## Model Evaluation

### Comprehensive Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/best_model.pt \
    --dataset-config config/dataset.yaml \
    --model-config config/model_resnet.yaml \
    --output-dir experiments/eval_results
```

**Generates**:
- Predictions CSV
- Confusion matrix (normalized and raw)
- ROC curve (binary classification)
- Precision-recall curve
- Classification report
- Feature embeddings (if model extracts)

### Metrics Explained

**Accuracy**: Overall correctness
- Good for balanced datasets
- Can be misleading with imbalance

**Balanced Accuracy**: Average of per-class accuracies
- Better for imbalanced datasets

**Precision**: Of predicted positives, how many are correct?
- High precision = few false positives
- Important when false positives are costly

**Recall/Sensitivity**: Of actual positives, how many are detected?
- High recall = few false negatives
- Important when false negatives are costly

**F1 Score**: Harmonic mean of precision and recall
- Good overall metric
- Balances precision and recall

**AUROC**: Area under ROC curve
- Threshold-independent
- 0.5 = random, 1.0 = perfect
- Good for comparing models

**AUPRC**: Area under precision-recall curve
- Better than AUROC for imbalanced data

### Error Analysis

The evaluator provides error analysis:
```python
# In evaluation results
error_analysis = {
    "num_errors": 45,
    "error_rate": 0.09,
    "high_conf_error_indices": [...],  # High confidence but wrong
    "low_conf_correct_indices": [...],  # Low confidence but correct
}
```

Review these cases to improve model:
- High confidence errors: Systematic mistakes
- Low confidence correct: Model uncertain regions

## Explainability

### Grad-CAM Visualization

Generate Grad-CAM for single image:
```bash
python scripts/infer_single_image.py \
    --image path/to/image.png \
    --checkpoint experiments/checkpoints/best_model.pt \
    --save-gradcam
```

### Programmatic Grad-CAM

```python
from cancer_quant_model.explainability.grad_cam import GradCAM, get_target_layer

# Get target layer
target_layers = get_target_layer(model, architecture="resnet")

# Create Grad-CAM
gradcam = GradCAM(model, target_layers, device="cuda")

# Generate heatmap
cam = gradcam(input_tensor, target_class=1)
```

### Interpret Grad-CAM

- **Red regions**: Important for prediction
- **Blue regions**: Less important
- Look for:
  - Cell nuclei highlighting
  - Tissue architecture patterns
  - Artifacts (if model focuses on wrong areas)

## Quantitative Feature Analysis

### Extract Features

```bash
python scripts/extract_quant_features.py \
    --input-dir data/raw/train \
    --output experiments/features/features.parquet
```

### Analyze Features

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load features
df = pd.read_csv("experiments/features/features.parquet")

# Merge with labels
train_df = pd.read_csv("data/splits/train.csv")
df = df.merge(train_df[['image_path', 'label']], on='image_path')

# Feature correlation with label
correlations = df.corr()['label'].sort_values(ascending=False)
print(correlations.head(20))

# Plot distributions
feature_cols = [c for c in df.columns if c.startswith('color_') or c.startswith('texture_')]

for feature in feature_cols[:10]:
    plt.figure()
    df.groupby('label')[feature].hist(alpha=0.6, bins=30)
    plt.title(feature)
    plt.legend(['Non-cancer', 'Cancer'])
    plt.show()
```

## Best Practices

### Experiment Organization

```
experiments/
├── exp001_resnet50_baseline/
│   ├── checkpoints/
│   ├── configs/
│   └── results/
├── exp002_efficientnet_augmented/
│   ├── checkpoints/
│   ├── configs/
│   └── results/
└── exp003_vit_large_dataset/
    ├── checkpoints/
    ├── configs/
    └── results/
```

### Configuration Versioning

Always save configs with experiments:
```python
# Configs are automatically logged to MLflow
# Also manually save:
import shutil
shutil.copy("config/dataset.yaml", "experiments/exp001/configs/")
shutil.copy("config/model_resnet.yaml", "experiments/exp001/configs/")
```

### Reproducibility Checklist

- [ ] Set seed in config
- [ ] Enable deterministic mode
- [ ] Log all hyperparameters
- [ ] Save model checkpoint
- [ ] Save configuration files
- [ ] Document dataset version
- [ ] Record environment (pip freeze)

### Performance Monitoring

Track these over time:
- Training/validation loss curves
- Accuracy plateau
- Overfitting (train vs val gap)
- Learning rate schedule
- Gradient norms

## Troubleshooting

### Model Not Learning

1. Check data loading (visualize batches)
2. Verify labels are correct
3. Check learning rate (too high/low)
4. Ensure loss is decreasing
5. Try smaller model first

### Overfitting

Symptoms: High train accuracy, low val accuracy

Solutions:
- Increase data augmentation
- Add dropout
- Reduce model size
- Early stopping
- L2 regularization (weight decay)

### Underfitting

Symptoms: Low train and val accuracy

Solutions:
- Increase model capacity
- Train longer
- Reduce regularization
- Check data quality

### Slow Training

- Increase batch size
- More workers for data loading
- Mixed precision training
- Profile code to find bottleneck
- Use faster model architecture

## Advanced: Custom Experiments

### Custom Loss Function

```python
# In scripts/train.py, add:
class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # Your custom loss
        return loss

criterion = CustomLoss()
```

### Custom Metrics

```python
# In utils/metrics_utils.py
def custom_metric(y_true, y_pred):
    # Your metric
    return score
```

### Ensemble Models

```python
from cancer_quant_model.api.inference_api import InferenceAPI

# Load multiple models
models = [
    InferenceAPI.from_checkpoint("checkpoint1.pt", ...),
    InferenceAPI.from_checkpoint("checkpoint2.pt", ...),
    InferenceAPI.from_checkpoint("checkpoint3.pt", ...),
]

# Ensemble prediction
def ensemble_predict(image):
    predictions = [model.predict(image) for model in models]
    probs = [p['probabilities'] for p in predictions]
    # Average probabilities
    avg_probs = {k: np.mean([p[k] for p in probs]) for k in probs[0].keys()}
    return avg_probs
```

## Continuous Improvement

1. **Start Simple**: Baseline with ResNet-50
2. **Establish Metrics**: Define success criteria
3. **Iterate**: Try different architectures, hyperparameters
4. **Analyze Errors**: Understand failure modes
5. **Add Data**: Collect more challenging examples
6. **Monitor Production**: Track performance over time

Remember: The best model is the one that works for your specific use case!
