# Advanced Cancer AI Dashboard

Modern, responsive web dashboard for the Advanced Cancer AI Detection System.

## Features

- **Real-time Predictions**: Upload medical images and get instant AI-powered cancer detection results
- **Batch Processing**: Process multiple images simultaneously
- **Interactive Visualizations**: Rich charts and graphs showing prediction data
- **Patient History**: Track and review all past predictions
- **Analytics**: Comprehensive analytics and insights
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices

## Technology Stack

- **React 18** - Modern UI library
- **Vite** - Lightning-fast build tool
- **Tailwind CSS** - Utility-first CSS framework
- **Recharts** - Composable charting library
- **Zustand** - Lightweight state management
- **React Router** - Client-side routing
- **Axios** - HTTP client
- **React Dropzone** - Drag-and-drop file uploads

## Quick Start

### Prerequisites

- Node.js 16+ and npm/yarn
- Backend API server running on port 8000

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create environment file
cp .env.example .env

# Edit .env to configure your API endpoint
# VITE_API_URL=http://localhost:8000

# Start development server
npm run dev
```

The dashboard will be available at `http://localhost:5173`

### Production Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview
```

## Project Structure

```
frontend/
├── public/              # Static assets
├── src/
│   ├── components/      # Reusable UI components
│   │   ├── Layout.jsx
│   │   └── PredictionResults.jsx
│   ├── pages/          # Page components
│   │   ├── Dashboard.jsx
│   │   ├── NewPrediction.jsx
│   │   ├── BatchProcessing.jsx
│   │   ├── History.jsx
│   │   ├── Analytics.jsx
│   │   └── Settings.jsx
│   ├── services/       # API services
│   │   └── api.js
│   ├── store/          # State management
│   │   └── useStore.js
│   ├── App.jsx         # Root component
│   ├── main.jsx        # Entry point
│   └── index.css       # Global styles
├── index.html
├── vite.config.js
├── tailwind.config.js
└── package.json
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Lint code
- `npm run format` - Format code with Prettier

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```env
VITE_API_URL=http://localhost:8000
VITE_APP_NAME=Advanced Cancer AI Dashboard
VITE_APP_VERSION=1.0.0
```

### Tailwind Configuration

Customize theme in `tailwind.config.js`:

```javascript
theme: {
  extend: {
    colors: {
      primary: {...},
      medical: {...}
    }
  }
}
```

## Features Guide

### 1. Dashboard
- Overview of system health
- Recent predictions
- Statistics and charts
- Quick action links

### 2. New Prediction
- Upload medical images (DICOM, NIfTI, PNG, JPG, etc.)
- Enter clinical data (age, gender, history)
- Get AI predictions with confidence scores
- View detailed probability breakdowns
- Export results

### 3. Batch Processing
- Upload multiple images at once
- Process all images simultaneously
- View results in a table
- Export to CSV or JSON

### 4. History
- Browse all past predictions
- Search and filter
- Sort by date, confidence, or risk
- Delete individual predictions

### 5. Analytics
- Visual analytics and trends
- Cancer type distribution
- Confidence and risk metrics
- Performance insights

### 6. Settings
- Configure application preferences
- View model information
- Adjust confidence thresholds
- Privacy and data settings

## API Integration

The dashboard communicates with the backend API:

```javascript
// Example API call
import { api } from './services/api';

// Single prediction
const result = await api.predict(imageFile, clinicalData);

// Batch prediction
const results = await api.batchPredict(imageFiles);
```

## State Management

Using Zustand for simple, efficient state management:

```javascript
import useStore from './store/useStore';

const { predictions, addPrediction, settings } = useStore();
```

## Styling

### Tailwind Utility Classes

```jsx
<button className="btn btn-primary">Click me</button>
<div className="card">Card content</div>
<span className="badge badge-success">Success</span>
```

### Custom Components

Pre-styled components available:
- `.card` - Card container
- `.btn`, `.btn-primary`, `.btn-secondary`, `.btn-danger` - Buttons
- `.input`, `.label` - Form elements
- `.badge-*` - Status badges

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Medical Disclaimer

**IMPORTANT**: This dashboard is for research and educational purposes only.
- NOT approved for clinical use
- NOT a substitute for professional medical advice
- Always consult qualified healthcare professionals

## Troubleshooting

### API Connection Issues

If you see connection errors:
1. Verify backend server is running (`http://localhost:8000/health`)
2. Check `VITE_API_URL` in `.env`
3. Ensure CORS is enabled on backend

### Build Errors

```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf node_modules/.vite
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- GitHub Issues: [Create an issue]
- Documentation: See main project README
