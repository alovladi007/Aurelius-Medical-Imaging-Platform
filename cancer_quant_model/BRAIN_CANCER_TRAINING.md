# Training on Kaggle Brain Cancer Dataset

## Quick Setup & Training Guide

### Step 1: Locate Your Dataset

Find where you saved "Kaggle Brain Cancer Data.zip":

```bash
# If in your Downloads folder:
ls -lh ~/Downloads/"Kaggle Brain Cancer Data.zip"

# If somewhere else, search for it:
find ~ -name "*Kaggle*Brain*Cancer*.zip" 2>/dev/null
```

### Step 2: Extract & Setup

Once you have the path, run:

```bash
cd /home/user/Aurelius-Medical-Imaging-Platform/cancer_quant_model

# Option A: Quick test with sample (RECOMMENDED for first run)
python scripts/setup_dataset.py \
    --zip-path "/path/to/Kaggle Brain Cancer Data.zip" \
    --create-sample \
    --sample-size 200

# Option B: Extract full dataset
python scripts/setup_dataset.py \
    --zip-path "/path/to/Kaggle Brain Cancer Data.zip"
```

**What this does:**
- Extracts the zip file
- Analyzes the structure
- Counts images per class
- (Optional) Creates a small sample for quick testing

### Step 3: Update Configuration

Edit `config/dataset.yaml`:

```bash
# Open in your editor
nano config/dataset.yaml
# OR
vim config/dataset.yaml
# OR
code config/dataset.yaml
```

**Update these lines:**

```yaml
dataset:
  # If using sample:
  paths:
    raw_data_dir: "data/raw/sample"

  # If using full dataset:
  # paths:
  #   raw_data_dir: "data/raw/extracted_folder_name"

  labels:
    num_classes: 4  # Or however many classes brain cancer dataset has
    class_names: ["glioma", "meningioma", "pituitary", "no_tumor"]  # Update with actual classes
```

### Step 4: Create Train/Val/Test Splits

```bash
python scripts/create_splits.py --config config/dataset.yaml
```

**Expected output:**
```
Train: XXXX samples
Val: XXXX samples
Test: XXXX samples
```

**Verify splits created:**
```bash
ls -lh data/splits/
# Should show: train.csv  val.csv  test.csv
```

### Step 5: Quick Verification (Optional but Recommended)

```bash
# Check splits are valid
python -c "
import pandas as pd
train = pd.read_csv('data/splits/train.csv')
print(f'Train: {len(train)} samples')
print(train['label'].value_counts())
"
```

### Step 6: Train Your First Model

**Option A: Quick test (ResNet-18, 5 epochs)**

First, create a quick test config:

```bash
cp config/model_resnet.yaml config/model_resnet_test.yaml
```

Edit `config/model_resnet_test.yaml`:
```yaml
model:
  variant: "resnet18"  # Faster
  training:
    max_epochs: 5      # Quick test
```

Then train:
```bash
python scripts/train.py \
    --dataset-config config/dataset.yaml \
    --model-config config/model_resnet_test.yaml \
    --train-config config/train_default.yaml
```

**Option B: Full training (ResNet-50, 50 epochs)**

```bash
python scripts/train.py \
    --dataset-config config/dataset.yaml \
    --model-config config/model_resnet.yaml \
    --train-config config/train_default.yaml
```

### Step 7: Monitor Training

Open a new terminal:

```bash
cd /home/user/Aurelius-Medical-Imaging-Platform/cancer_quant_model

mlflow ui --backend-store-uri experiments/mlruns --port 5000
```

Then open in browser: `http://localhost:5000`

You'll see:
- Real-time loss curves
- Accuracy, AUROC, F1 scores
- All hyperparameters
- Model checkpoints

### Step 8: Evaluate Your Model

After training completes:

```bash
python scripts/evaluate.py \
    --checkpoint experiments/checkpoints/best.pt \
    --dataset-config config/dataset.yaml \
    --model-config config/model_resnet.yaml \
    --output-dir experiments/brain_cancer_results
```

**Results will be in:**
- `experiments/brain_cancer_results/predictions.csv`
- `experiments/brain_cancer_results/confusion_matrix.png`
- `experiments/brain_cancer_results/roc_curve.png`
- Console output with all metrics

### Step 9: Test Inference on Single Image

```bash
# Pick any test image
python scripts/infer_single_image.py \
    --image data/raw/sample/class_name/some_image.jpg \
    --checkpoint experiments/checkpoints/best.pt \
    --model-config config/model_resnet.yaml \
    --save-gradcam \
    --output-dir experiments/brain_cancer_inference
```

**Output:**
- Predicted class
- Confidence scores
- Grad-CAM heatmap showing what the model looks at

### Step 10: Extract Quantitative Features

```bash
python scripts/extract_quant_features.py \
    --input-dir data/raw/sample \
    --output experiments/brain_cancer_features.parquet
```

**Then analyze in Python or notebook:**
```python
import pandas as pd
features = pd.read_parquet('experiments/brain_cancer_features.parquet')
print(f"Extracted {len(features.columns)} features for {len(features)} images")
```

---

## Expected Timeline

### With Sample Dataset (200 images/class)

| Step | Time | GPU | CPU |
|------|------|-----|-----|
| Extract & setup | 2 min | - | - |
| Create splits | 10 sec | - | - |
| Train (5 epochs, ResNet-18) | 5 min | 15 min | - |
| Evaluate | 1 min | 2 min | - |
| Extract features | 2 min | 5 min | - |
| **Total** | **~10 min** | **~25 min** | - |

### With Full Dataset (typical: 2000+ images/class)

| Step | Time (GPU) | Time (CPU) |
|------|-----------|-----------|
| Extract & setup | 5-10 min | 5-10 min |
| Create splits | 30 sec | 30 sec |
| Train (50 epochs, ResNet-50) | 2-4 hours | 20-30 hours |
| Evaluate | 5 min | 15 min |
| Extract features | 10 min | 30 min |
| **Total** | **~3-5 hours** | **~25-35 hours** |

---

## Troubleshooting

### "File not found"
```bash
# Check the zip file exists
ls -lh "/path/to/Kaggle Brain Cancer Data.zip"

# If it has spaces in the name, use quotes:
python scripts/setup_dataset.py --zip-path "/path with spaces/file.zip"
```

### "Out of memory" during training
```bash
# Reduce batch size in config/dataset.yaml:
dataset:
  dataloader:
    batch_size: 16  # or 8
```

### "CUDA out of memory"
```bash
# Use smaller model or enable gradient accumulation
# Edit config/train_default.yaml:
training:
  accumulate_grad_batches: 2  # Doubles effective batch size without more memory
```

### Training too slow on CPU
```bash
# Use the sample dataset first
# Or use a smaller model (ResNet-18 instead of ResNet-50)
# Or reduce image size in config/dataset.yaml:
dataset:
  image:
    target_size: [128, 128]  # Instead of [224, 224]
```

---

## Advanced: Try Different Models

### EfficientNet (More efficient)
```bash
python scripts/train.py \
    --model-config config/model_efficientnet.yaml \
    --dataset-config config/dataset.yaml
```

### Vision Transformer (Best accuracy)
```bash
python scripts/train.py \
    --model-config config/model_vit.yaml \
    --dataset-config config/dataset.yaml
```

---

## Compare All Models in MLflow

1. Train ResNet-50, EfficientNet, and ViT
2. Open MLflow UI
3. Select all runs
4. Click "Compare"
5. See which performs best!

---

## Next Steps After Training

1. **Analyze mistakes**: Look at misclassified images
2. **Feature analysis**: Use the notebooks to visualize extracted features
3. **Ensemble models**: Combine predictions from multiple models
4. **Production deployment**: Use the inference API
5. **Research paper**: Use the results for publication

---

## One-Line Complete Workflow

```bash
# Once you have the path to your zip file, run:
cd /home/user/Aurelius-Medical-Imaging-Platform/cancer_quant_model && \
python scripts/setup_dataset.py --zip-path "/path/to/Kaggle Brain Cancer Data.zip" --create-sample && \
python scripts/create_splits.py --config config/dataset.yaml && \
python scripts/train.py --dataset-config config/dataset.yaml --model-config config/model_resnet.yaml && \
python scripts/evaluate.py --checkpoint experiments/checkpoints/best.pt --dataset-config config/dataset.yaml --model-config config/model_resnet.yaml
```

---

**Ready to train? Just provide the path to your zip file and let's go!**
