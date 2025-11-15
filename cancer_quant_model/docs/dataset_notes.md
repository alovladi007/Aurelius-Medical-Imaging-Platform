# Dataset Preparation Guide

This guide explains how to prepare your Kaggle histopathology dataset for use with the cancer quantitative model.

## Supported Dataset Formats

### Option 1: Folder Structure (folder_binary)

Organize images in class-based folders:

```
data/raw/
├── train/
│   ├── 0/  # Class 0 (e.g., non-cancer)
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   └── 1/  # Class 1 (e.g., cancer)
│       ├── image_100.png
│       ├── image_101.png
│       └── ...
├── val/  # Optional
│   ├── 0/
│   └── 1/
└── test/  # Optional
    ├── 0/
    └── 1/
```

**Configuration (`config/dataset.yaml`):**
```yaml
dataset_type: "folder_binary"
folder_structure:
  train_dir: "data/raw/train"
  val_dir: null  # Will split from train
  test_dir: null  # Will split from train
```

### Option 2: CSV Labels (csv_labels)

All images in one folder with CSV labels:

```
data/raw/
├── images/
│   ├── img_001.png
│   ├── img_002.png
│   └── ...
└── labels.csv
```

**labels.csv format:**
```csv
image_id,label
img_001.png,0
img_002.png,1
img_003.png,0
...
```

**Configuration (`config/dataset.yaml`):**
```yaml
dataset_type: "csv_labels"
csv_structure:
  image_dir: "data/raw/images"
  csv_path: "data/raw/labels.csv"
  image_id_column: "image_id"
  label_column: "label"
```

## Common Kaggle Datasets

### 1. Breast Cancer Histopathology (BreakHis)

**Download:** https://www.kaggle.com/datasets/ambarish/breakhis

**Structure:** Organized by magnification and class
- Convert to folder_binary format
- Choose one magnification level or combine

**Classes:** Benign vs Malignant (8 subtypes total)

### 2. PatchCamelyon (PCam)

**Download:** https://www.kaggle.com/c/histopathologic-cancer-detection

**Structure:** Images + CSV labels
- Use csv_labels format
- Binary classification (metastatic vs normal)

**Size:** 220,000+ patches

### 3. Colorectal Histology (NCT-CRC-HE-100K)

**Download:** https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist

**Structure:** Folder-based, 9 tissue classes
- Use folder_binary (or extend to multi-class)

## Data Preparation Workflow

### Step 1: Download Dataset

Download from Kaggle:
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API credentials
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download dataset
kaggle datasets download -d <dataset-name>
unzip <dataset-name>.zip -d data/raw/
```

### Step 2: Organize Data

Place data in appropriate structure:

- **Folder-based:** Ensure class directories are properly named
- **CSV-based:** Verify CSV has correct columns and image paths are valid

### Step 3: Configure Dataset

Edit `config/dataset.yaml`:

```yaml
# Dataset type
dataset_type: "folder_binary"  # or "csv_labels"

# Paths
data_root: "data/raw"
processed_root: "data/processed"
splits_root: "data/splits"

# Image settings
image_settings:
  target_size: 224  # Standard for ImageNet models
  extensions: [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
  
  # For large WSI images, create tiles
  create_tiles: false
  tile_size: 512
  tile_overlap: 0

# Split configuration
split:
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  stratify: true  # Maintain class balance
  seed: 42

# Classes
classes:
  num_classes: 2
  class_names: ["non_cancer", "cancer"]
```

### Step 4: Prepare Data

```bash
python scripts/prepare_data.py --config config/dataset.yaml
```

This will:
- Validate images
- Create tiles if configured
- Copy/process to `data/processed/`

### Step 5: Create Splits

```bash
python scripts/create_splits.py --config config/dataset.yaml
```

Creates:
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

## Data Quality Checks

### Before Training

1. **Check class balance:**
   ```python
   import pandas as pd
   train_df = pd.read_csv("data/splits/train.csv")
   print(train_df['label'].value_counts())
   ```

2. **Inspect sample images:**
   ```python
   import cv2
   import matplotlib.pyplot as plt
   
   img = cv2.imread(train_df.iloc[0]['image_path'])
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   plt.imshow(img)
   plt.show()
   ```

3. **Verify image sizes:**
   ```python
   from PIL import Image
   
   for path in train_df['image_path'].sample(10):
       img = Image.open(path)
       print(f"{path}: {img.size}")
   ```

## Tips for Better Performance

1. **Class Balance:**
   - Use stratified splitting
   - Consider class weights in loss function
   - Apply oversampling/undersampling if severely imbalanced

2. **Image Quality:**
   - Remove corrupted images
   - Ensure consistent color calibration
   - Filter out artifacts

3. **Data Augmentation:**
   - Adjust augmentation strength in `config/train_default.yaml`
   - Use stain normalization for histopathology

4. **Multi-Magnification:**
   - Train separate models per magnification
   - Or resize all to consistent size

## Troubleshooting

**Issue:** "No images found"
- Check paths in config
- Verify file extensions match config
- Ensure images exist in specified directories

**Issue:** "Corrupted images"
- Run quality check script
- Remove/replace corrupted files

**Issue:** "Out of memory"
- Reduce batch size
- Enable tiling for large images
- Use smaller model variant

**Issue:** "Class imbalance"
- Use `class_weights` in config
- Apply focal loss
- Use oversampling techniques

## Additional Resources

- [PyTorch Data Loading](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [Albumentations Augmentation](https://albumentations.ai/)
- [Medical Image Analysis Best Practices](https://github.com/Project-MONAI/tutorials)
