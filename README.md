# 🏥 Advanced Multimodal AI for Cancer Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

State-of-the-art production-ready AI system for multimodal cancer detection integrating medical imaging, clinical data, and genomic information.

## ✨ Features

### Core Capabilities
- **Multimodal Fusion**: Integrates medical imaging (CT, MRI, X-ray), clinical/tabular data, and genomic sequences
- **Advanced Architecture**: Vision Transformers + EfficientNet ensemble with cross-modal attention
- **Multi-Task Learning**: Simultaneous cancer type detection, staging prediction, and risk assessment
- **Production-Ready**: Complete data pipeline, training infrastructure, and deployment server
- **HIPAA Compliant**: Secure medical data handling and processing

### Supported Cancer Types
- Lung Cancer
- Breast Cancer
- Prostate Cancer
- Colorectal Cancer

### Medical Imaging Formats
- **DICOM** (.dcm) - Industry standard for medical imaging
- **NIfTI** (.nii, .nii.gz) - Neuroimaging format
- **Standard Images** (.png, .jpg, .bmp, .tiff)

## 🏗️ Architecture

```
┌─────────────────┐
│ Medical Imaging │──┐
└─────────────────┘  │
                     │    ┌──────────────────────┐
┌─────────────────┐  │    │   Image Encoder      │
│  Clinical Data  │──┼───▶│  (ViT + EfficientNet)│
└─────────────────┘  │    └──────────────────────┘
                     │             │
┌─────────────────┐  │    ┌──────────────────────┐
│  Genomic Data   │──┘    │  Clinical Encoder    │
└─────────────────┘       │  (Transformer)       │
                          └──────────────────────┘
                                   │
                          ┌──────────────────────┐
                          │  Genomic Encoder     │
                          │  (CNN + Attention)   │
                          └──────────────────────┘
                                   │
                                   ▼
                     ┌──────────────────────────┐
                     │ Cross-Modal Fusion       │
                     │ (Multi-Head Attention)   │
                     └──────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
           ┌────────▼────────┐         ┌─────────▼────────┐
           │ Cancer Type     │         │  Staging         │
           │ Classifier      │         │  Classifier      │
           └─────────────────┘         └──────────────────┘
                    │
           ┌────────▼────────┐
           │ Risk Score      │
           │ Predictor       │
           └─────────────────┘
```

## 📊 Performance Targets

| Metric | Target | Description |
|--------|--------|-------------|
| Cancer Detection AUC | >0.95 | Area under ROC curve for cancer type classification |
| Staging Accuracy | >0.85 | Accuracy for cancer stage prediction |
| Risk Assessment R² | >0.80 | R-squared for risk score regression |
| Inference Speed | <100ms | Per-patient prediction latency |

## 🎨 Frontend Dashboard

A modern, responsive web dashboard for easy interaction with the AI system:

- **Real-time Predictions**: Upload and analyze medical images instantly
- **Batch Processing**: Process multiple images simultaneously
- **Interactive Visualizations**: Rich charts showing prediction data and trends
- **Patient History**: Track and review all past predictions
- **Analytics Dashboard**: Comprehensive insights and statistics
- **Responsive Design**: Works on desktop, tablet, and mobile

### Dashboard Features

| Feature | Description |
|---------|-------------|
| Single Prediction | Upload one image with clinical data for detailed analysis |
| Batch Processing | Process multiple images at once with bulk export |
| History | Search, filter, and review all past predictions |
| Analytics | Visual charts showing trends and performance metrics |
| Settings | Configure thresholds, preferences, and view model info |

## 🚀 Quick Start

### Option 1: One-Command Startup (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/advanced-cancer-ai.git
cd advanced-cancer-ai

# Run the startup script (starts both backend and frontend)
./start.sh
```

This will:
- Install all dependencies (Python and Node.js)
- Start the backend API server on port 8000
- Start the frontend dashboard on port 5173
- Open your browser automatically

### Option 2: Manual Installation

#### Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env
```

### Prepare Your Dataset

```bash
# Create metadata template
python prepare_dataset.py --create_template --output_dir ./data

# Validate your dataset
python prepare_dataset.py --validate ./data/metadata.csv

# Split into train/val/test
python prepare_dataset.py --split ./data/metadata.csv --train_ratio 0.7 --val_ratio 0.15
```

### Training

#### Option 1: Quick Test with Synthetic Data
```bash
python train.py --test_mode --epochs 10 --batch_size 16
```

#### Option 2: Train with Real Data
```bash
python train.py \
    --config configs/default_config.yaml \
    --data_dir ./data \
    --epochs 100 \
    --batch_size 32
```

#### Option 3: Custom Configuration
```bash
# Create custom config
cp configs/default_config.yaml configs/my_config.yaml
# Edit my_config.yaml with your settings

python train.py --config configs/my_config.yaml
```

### Running the System

#### Using the Startup Script
```bash
# Start both backend and frontend
./start.sh
```

#### Manual Startup

**Terminal 1 - Backend:**
```bash
# Activate virtual environment
source venv/bin/activate

# Start inference server
python -m src.deployment.inference_server
```

**Terminal 2 - Frontend:**
```bash
# Navigate to frontend
cd frontend

# Start development server
npm run dev
```

#### Access the Application

- **Frontend Dashboard**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

#### Using Docker (Production)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Making API Predictions

```bash
# Via curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@patient_scan.dcm" \
  -F "patient_age=55" \
  -F "smoking_history=true"

# Or use the web dashboard at http://localhost:5173
```

## 📁 Project Structure

```
advanced-cancer-ai/
├── frontend/                   # Web Dashboard
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/            # Page components (Dashboard, Prediction, etc.)
│   │   ├── services/         # API integration
│   │   └── store/            # State management
│   ├── public/               # Static assets
│   ├── package.json
│   └── vite.config.js
├── src/                       # Backend Source Code
│   ├── models/               # Neural network architectures
│   │   ├── __init__.py
│   │   └── multimodal_cancer_detector.py
│   ├── training/             # Training pipeline
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── evaluation/           # Evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── data/                 # Data pipeline
│   │   ├── __init__.py
│   │   ├── loaders.py       # DICOM, NIfTI, image loaders
│   │   ├── datasets.py      # PyTorch datasets
│   │   ├── augmentation.py  # Data augmentation
│   │   ├── preprocessing.py # Preprocessing utilities
│   │   └── data_manager.py  # Data management
│   ├── deployment/          # Production deployment
│   │   ├── __init__.py
│   │   └── inference_server.py
│   └── utils/               # Utilities
│       ├── __init__.py
│       ├── config.py        # Configuration management
│       ├── logger.py        # Logging setup
│       └── visualization.py # Plotting and visualization
├── configs/                  # Configuration files
│   └── default_config.yaml  # Default training/model configuration
├── train.py                 # Main training script
├── prepare_dataset.py       # Dataset preparation utilities
├── start.sh                 # One-command startup script
├── docker-compose.yml       # Docker orchestration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🔧 Configuration

The system uses YAML configuration files for all settings. Key configuration sections:

### Model Configuration
```yaml
model:
  num_classes: 4              # Number of cancer types
  num_stages: 5               # Number of staging levels
  image_size: [224, 224]     # Input image dimensions
  vision_model: "vit_base_patch16_224"
  efficientnet_model: "efficientnet_b4"
  fusion_dim: 512             # Fusion layer dimension
```

### Data Configuration
```yaml
data:
  data_dir: "./data"
  batch_size: 32
  num_workers: 4
  multimodal: true
  augmentation:
    enabled: true
    rotation_range: 15.0
    brightness_range: [0.85, 1.15]
```

### Training Configuration
```yaml
training:
  num_epochs: 100
  learning_rate: 0.0001
  optimizer: "adamw"
  scheduler: "cosine"
  use_focal_loss: true
  early_stopping:
    patience: 15
```

See `configs/default_config.yaml` for complete configuration options.

## 📊 Data Format

### Metadata CSV Format

```csv
image_path,cancer_type,cancer_stage,risk_score,age,gender,bmi,genomic_sequence
images/patient001.dcm,0,2,0.65,55,1,25.5,ATCGATCG...
images/patient002.dcm,1,3,0.82,62,0,28.3,GCTAGCTA...
```

### Label Encodings

**Cancer Types:**
- 0: Lung Cancer
- 1: Breast Cancer
- 2: Prostate Cancer
- 3: Colorectal Cancer

**Cancer Stages:**
- 0: Stage 0 (in situ)
- 1: Stage I
- 2: Stage II
- 3: Stage III
- 4: Stage IV

**Gender:**
- 0: Female
- 1: Male

## 📈 Evaluation Metrics

The system computes comprehensive evaluation metrics:

### Classification Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC (One-vs-Rest and One-vs-One)
- Per-class metrics for each cancer type

### Medical-Specific Metrics
- Sensitivity (True Positive Rate)
- Specificity (True Negative Rate)
- Positive Predictive Value (PPV)
- Negative Predictive Value (NPV)

### Regression Metrics (Risk Assessment)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## 🔬 Advanced Features

### Data Augmentation
- Medical-specific augmentations preserving diagnostic information
- Rotation, translation, zoom, and flipping
- Brightness and contrast adjustment
- Gaussian noise and elastic deformation
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Multi-Task Learning
Simultaneous optimization of three objectives:
1. Cancer type classification
2. Cancer stage prediction
3. Risk score regression

### Focal Loss
Handles class imbalance in medical datasets by focusing on hard examples.

### Early Stopping
Prevents overfitting with patience-based monitoring of validation metrics.

### Model Checkpointing
Automatically saves best models based on validation performance.

## 🌐 Public Datasets

The system supports integration with public cancer datasets:

### Supported Datasets
- **TCGA** (The Cancer Genome Atlas)
- **LIDC-IDRI** (Lung Image Database Consortium)
- **CBIS-DDSM** (Curated Breast Imaging Subset of DDSM)
- **Custom** datasets

### Dataset Integration
```python
from src.data.datasets import PublicCancerDataset

dataset = PublicCancerDataset(
    dataset_name='lidc',
    data_dir='./datasets/lidc',
    split='train'
)
```

## 🛠️ Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Style
```bash
# Format code
black src/

# Lint code
flake8 src/
```

### Adding New Features
1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests
5. Submit a pull request

## 📝 API Reference

### Inference Server Endpoints

#### POST /predict
Predict cancer from medical image

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@scan.dcm" \
  -F "clinical_data={\"age\":55,\"bmi\":25.5}"
```

**Response:**
```json
{
  "cancer_type": "Lung Cancer",
  "cancer_type_probabilities": [0.85, 0.08, 0.04, 0.03],
  "cancer_stage": 2,
  "risk_score": 0.72,
  "confidence": 0.85,
  "recommendations": ["Recommended: Further CT imaging", ...]
}
```

#### POST /predict_batch
Batch prediction for multiple images

#### GET /health
Health check endpoint

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is for research and educational purposes only.

- NOT approved for clinical diagnosis or treatment decisions
- NOT a substitute for professional medical advice
- NOT validated for regulatory compliance (FDA, CE, etc.)
- Requires validation and approval before any clinical use
- Users assume all responsibility for any application

Always consult qualified healthcare professionals for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for guidelines.

Areas for contribution:
- Additional cancer types
- New data augmentation techniques
- Performance optimizations
- Documentation improvements
- Bug fixes

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 🙏 Acknowledgments

- Vision Transformer (ViT) architecture from Google Research
- EfficientNet from Google Brain
- Medical imaging tools: PyDICOM, NiBabel, SimpleITK
- PyTorch and Hugging Face teams

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

## 🔗 References

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Multimodal Learning Survey](https://arxiv.org/abs/2209.03430)
- [Medical Image Analysis Best Practices](https://doi.org/10.1038/s41591-019-0447-x)

---

**Built with ❤️ for advancing cancer detection research**
