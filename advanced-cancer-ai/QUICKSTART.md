# Quick Start Guide - Advanced Cancer AI System

This guide will get you up and running in 5 minutes!

## Prerequisites

Before you begin, ensure you have:

- **Python 3.8+** installed ([Download](https://www.python.org/downloads/))
- **Node.js 16+** installed ([Download](https://nodejs.org/))
- **Git** installed
- At least **4GB of free RAM**
- **2GB of free disk space**

Check your installations:
```bash
python3 --version  # Should show 3.8 or higher
node --version     # Should show 16 or higher
npm --version      # Should show 7 or higher
```

## Installation & Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/advanced-cancer-ai.git
cd advanced-cancer-ai
```

### Step 2: Run the One-Command Startup

```bash
./start.sh
```

That's it! The script will:
- ✅ Create a Python virtual environment
- ✅ Install all Python dependencies
- ✅ Install all Node.js dependencies
- ✅ Start the backend API server (port 8000)
- ✅ Start the frontend dashboard (port 5173)

### Step 3: Open the Dashboard

Once both services are running, open your browser and navigate to:

**http://localhost:5173**

You should see the Advanced Cancer AI Dashboard!

## First Prediction

### Using the Web Dashboard (Easiest)

1. Click on **"New Prediction"** in the sidebar
2. Drag and drop a medical image (or click to browse)
3. Optionally fill in clinical information
4. Click **"Analyze Image"**
5. View your prediction results!

### Using the API

```bash
# Test the health endpoint
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -F "image=@path/to/your/image.jpg" \
  -F "patient_age=55" \
  -F "smoking_history=true"
```

## Test Mode (No Real Data Required)

Want to try the system without medical images? Use test mode:

```bash
# Stop the current system (Ctrl+C in terminal)

# Start in test mode
python train.py --test_mode --epochs 5
```

This will:
- Generate synthetic data
- Train a small test model
- Allow you to test the system without real medical data

## Dashboard Features Overview

### 📊 Dashboard (Home)
- System status and health
- Recent predictions
- Quick statistics
- Visual charts

### 🔬 New Prediction
- Upload single medical image
- Enter patient clinical data
- Get AI-powered predictions
- View detailed results with visualizations

### 📦 Batch Processing
- Upload multiple images at once
- Process all simultaneously
- Export results to CSV/JSON
- View summary statistics

### 📚 History
- Browse all past predictions
- Search and filter
- Sort by various criteria
- Delete individual predictions

### 📈 Analytics
- Visual trends and insights
- Cancer type distribution
- Confidence metrics
- Performance by type

### ⚙️ Settings
- Configure preferences
- View model information
- Adjust thresholds
- Privacy settings

## File Format Support

The system accepts:

### Medical Imaging Formats
- **DICOM** (.dcm) - Medical imaging standard
- **NIfTI** (.nii, .nii.gz) - Neuroimaging format

### Standard Image Formats
- **PNG** (.png)
- **JPEG** (.jpg, .jpeg)
- **BMP** (.bmp)
- **TIFF** (.tiff, .tif)

## Stopping the System

To stop all services:

1. Press `Ctrl + C` in the terminal where `start.sh` is running
2. Both backend and frontend will shut down automatically

## Docker Alternative (Production)

For production deployment:

```bash
# Build and start
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

## Troubleshooting

### Port Already in Use

If you see "port already in use" errors:

```bash
# Check what's using port 8000
lsof -i :8000

# Check what's using port 5173
lsof -i :5173

# Kill the process if needed
kill -9 <PID>
```

### Backend Won't Start

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies again
pip install -r requirements.txt

# Try starting manually
python -m src.deployment.inference_server
```

### Frontend Won't Start

```bash
cd frontend

# Remove node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Start again
npm run dev
```

### Can't Connect to API

1. Verify backend is running: `curl http://localhost:8000/health`
2. Check frontend `.env` file has correct API URL
3. Check browser console for errors

## Next Steps

### 1. Prepare Your Dataset

```bash
# Create metadata template
python prepare_dataset.py --create_template --output_dir ./data

# Validate your dataset
python prepare_dataset.py --validate ./data/metadata.csv
```

### 2. Train Your Model

```bash
# Train with your data
python train.py \
  --config configs/default_config.yaml \
  --data_dir ./data \
  --epochs 100
```

### 3. Deploy to Production

See `docker-compose.yml` for production deployment configuration.

## Getting Help

- **Documentation**: Check the main [README.md](README.md)
- **Frontend Docs**: See [frontend/README.md](frontend/README.md)
- **Issues**: Report bugs on GitHub Issues
- **API Docs**: http://localhost:8000/docs (when server is running)

## Important Reminders

⚠️ **Medical Disclaimer**: This system is for research and educational purposes only. NOT approved for clinical use.

🔒 **Privacy**: All data is stored locally in your browser. No patient information is transmitted except to the local API endpoint.

📊 **Performance**: For best results, use high-quality medical images and include clinical data when available.

## Quick Reference

| Action | Command |
|--------|---------|
| Start System | `./start.sh` |
| Stop System | `Ctrl + C` |
| Backend URL | http://localhost:8000 |
| Frontend URL | http://localhost:5173 |
| API Docs | http://localhost:8000/docs |
| View Logs | `docker-compose logs -f` |

---

**Ready to detect cancer with AI? Start now with `./start.sh`!**
