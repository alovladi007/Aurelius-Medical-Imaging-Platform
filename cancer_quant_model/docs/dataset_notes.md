# Dataset Setup Guide

## Supported Dataset Formats

This pipeline supports two dataset formats for maximum flexibility.

### Format 1: Folder Binary Structure

Organize your images in a folder structure by class:

```
data/raw/train/
  ├── 0/  (class 0: non-cancer)
  │   ├── image_001.png
  │   ├── image_002.png
  │   └── ...
  └── 1/  (class 1: cancer)
      ├── image_001.png
      ├── image_002.png
      └── ...
```

**Configuration**:
```yaml
# In config/dataset.yaml
dataset:
  type: "folder_binary"
  paths:
    train_folder: "data/raw/train"
```

### Format 2: CSV Labels

Store all images in one folder with a CSV file containing labels:

```
data/raw/
  ├── images/
  │   ├── image_001.png
  │   ├── image_002.png
  │   └── ...
  └── labels.csv
```

**labels.csv format**:
```csv
image_id,label
image_001.png,0
image_002.png,1
...
```

**Configuration**:
```yaml
# In config/dataset.yaml
dataset:
  type: "csv_labels"
  paths:
    images_folder: "data/raw/images"
    labels_csv: "data/raw/labels.csv"
    csv_image_col: "image_id"
    csv_label_col: "label"
```

## Kaggle Dataset Examples

### Example 1: BreakHis Dataset

**Source**: Breast Cancer Histopathological Database (BreakHis)

**Structure**:
```
raw/
  ├── benign/
  │   ├── adenosis/
  │   ├── fibroadenoma/
  │   └── ...
  └── malignant/
      ├── ductal_carcinoma/
      ├── lobular_carcinoma/
      └── ...
```

**Setup**:
1. Download from Kaggle
2. Reorganize into binary classes:
   ```bash
   mkdir -p data/raw/train/0  # benign
   mkdir -p data/raw/train/1  # malignant

   # Move all benign images to data/raw/train/0/
   # Move all malignant images to data/raw/train/1/
   ```
3. Run create_splits.py

### Example 2: PatchCamelyon (PCam)

**Source**: Histopathologic Cancer Detection

**Structure**: Already in binary format with CSV

**Setup**:
1. Download from Kaggle
2. Place in `data/raw/`
3. Configure for CSV format
4. Run create_splits.py

## Data Preparation Workflow

### Step 1: Download and Extract

```bash
# Example: Using Kaggle API
kaggle datasets download -d [dataset-name]
unzip [dataset-name].zip -d data/raw/
```

### Step 2: Organize Data

Choose one of the supported formats and organize accordingly.

### Step 3: Configure Dataset

Edit `config/dataset.yaml`:

```yaml
dataset:
  type: "folder_binary"  # or "csv_labels"

  paths:
    raw_data_dir: "data/raw"
    processed_data_dir: "data/processed"
    splits_dir: "data/splits"

  image:
    target_size: [224, 224]
    channels: 3
    extensions: [".png", ".jpg", ".jpeg", ".tif"]

  labels:
    num_classes: 2
    class_names: ["non_cancer", "cancer"]

  split:
    train_ratio: 0.7
    val_ratio: 0.15
    test_ratio: 0.15
    stratify: true
    seed: 42
```

### Step 4: Create Splits

```bash
python scripts/create_splits.py --config config/dataset.yaml
```

This generates:
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

## Working with Large Whole-Slide Images (WSI)

If your dataset contains large WSI images (e.g., 10000x10000 pixels), you need to tile them first.

### Tiling Configuration

```yaml
# In config/dataset.yaml
dataset:
  image:
    create_patches: true
    patch_size: 512
    patch_overlap: 0
    min_tissue_ratio: 0.5  # Filter out mostly background patches
```

### Run Tiling

```bash
python scripts/prepare_data.py --config config/dataset.yaml --create-tiles
```

This will:
1. Load each large image
2. Extract patches of size `patch_size x patch_size`
3. Filter patches with insufficient tissue
4. Save patches to `data/processed/`

### Tissue Detection

The tiling script automatically detects tissue regions by:
- Converting to grayscale
- Thresholding (pixels < 220 are considered tissue)
- Computing tissue ratio
- Keeping only patches with `tissue_ratio >= min_tissue_ratio`

## Data Quality Checks

### Check Class Distribution

```python
import pandas as pd

# Load splits
train_df = pd.read_csv("data/splits/train.csv")
val_df = pd.read_csv("data/splits/val.csv")
test_df = pd.read_csv("data/splits/test.csv")

# Check distribution
print("Train distribution:")
print(train_df['label'].value_counts())

print("\nVal distribution:")
print(val_df['label'].value_counts())

print("\nTest distribution:")
print(test_df['label'].value_counts())
```

### Check Image Quality

```python
from PIL import Image
import numpy as np

# Check a sample image
img_path = train_df.iloc[0]['image_path']
img = Image.open(img_path)

print(f"Image size: {img.size}")
print(f"Image mode: {img.mode}")
print(f"Image format: {img.format}")

# Check for corrupted images
img_array = np.array(img)
print(f"Array shape: {img_array.shape}")
print(f"Array dtype: {img_array.dtype}")
print(f"Value range: [{img_array.min()}, {img_array.max()}]")
```

## Common Issues and Solutions

### Issue: Images in Different Formats

**Solution**: Convert all images to same format (PNG recommended)

```python
from pathlib import Path
from PIL import Image

for img_path in Path("data/raw").rglob("*.jpg"):
    img = Image.open(img_path)
    new_path = img_path.with_suffix(".png")
    img.save(new_path)
    img_path.unlink()  # Delete original
```

### Issue: Imbalanced Classes

**Solution**: The training script automatically computes class weights.

Alternatively, manually balance:
```python
from sklearn.utils import resample

# Oversample minority class or undersample majority class
```

### Issue: Missing Files

**Solution**: The dataset validates file existence and reports missing files.

Check logs for:
```
WARNING: Found X missing image files
```

### Issue: Inconsistent Image Sizes

**Solution**: The dataloader automatically resizes images. No action needed.

For preprocessing, set target size in config:
```yaml
dataset:
  image:
    target_size: [224, 224]
```

## Example: Complete Setup

Here's a complete example starting from a Kaggle download:

```bash
# 1. Download dataset
kaggle datasets download -d paultimothymooney/breast-histopathology-images
unzip breast-histopathology-images.zip -d data/raw/

# 2. Organize into binary classes
mkdir -p data/raw/train/0 data/raw/train/1
# Move images accordingly

# 3. Create splits
python scripts/create_splits.py --config config/dataset.yaml

# 4. Verify
ls data/splits/
# Should see: train.csv  val.csv  test.csv

# 5. Check distribution
python -c "import pandas as pd; df = pd.read_csv('data/splits/train.csv'); print(df['label'].value_counts())"

# 6. Ready to train!
python scripts/train.py
```

## Metadata Support

If your CSV has additional metadata columns (patient ID, tissue type, etc.), include them:

```yaml
# In config/dataset.yaml
dataset:
  paths:
    csv_metadata_cols: ["patient_id", "tissue_type", "magnification"]
```

These will be available in the dataset's metadata dictionary during training.
