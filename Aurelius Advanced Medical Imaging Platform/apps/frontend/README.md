# Aurelius Medical Imaging Platform - Frontend

A modern, sophisticated frontend application for the Aurelius Medical Imaging Platform built with Next.js 14, React, and TypeScript.

## Features

### 🎨 Modern UI/UX
- Beautiful, responsive design with Tailwind CSS
- Dark mode support
- Smooth animations and transitions
- Interactive components with Radix UI

### 🏥 Medical Imaging
- DICOM viewer with interactive tools
- Study and series management
- Image manipulation (zoom, pan, rotate, window/level)
- Multi-series comparison
- 3D rendering support (planned)

### 🤖 AI/ML Integration
- Browse available AI models
- Run inference on medical images
- View confidence scores and results
- Real-time inference status
- Model performance metrics

### 👥 Multi-Tenancy
- Tenant administration dashboard
- Usage monitoring and quotas
- Billing integration
- Role-based access control

### 📊 Analytics & Monitoring
- Real-time metrics dashboard
- System health monitoring
- Usage statistics
- Activity logs

### 🔐 Authentication & Security
- OAuth2/OIDC with Keycloak
- JWT token management
- Secure API communication
- Role-based access control

## Technology Stack

- **Framework**: Next.js 14 (App Router)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: Radix UI
- **State Management**: Zustand
- **Data Fetching**: TanStack Query (React Query)
- **HTTP Client**: Axios
- **DICOM Rendering**: Cornerstone.js
- **Charts**: Recharts
- **Animations**: Framer Motion
- **Form Handling**: React Hook Form + Zod

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Access to Aurelius backend API
- Keycloak instance for authentication

### Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.local.example .env.local
   # Edit .env.local with your configuration
   ```

3. **Run development server:**
   ```bash
   npm run dev
   ```

4. **Open browser:**
   Navigate to http://localhost:3000

### Docker Deployment

```bash
# Build image
docker build -t aurelius-frontend .

# Run container
docker run -p 3000:3000 aurelius-frontend
```

### With Docker Compose

The frontend is already configured in the main `compose.yaml`:

```bash
docker compose up frontend
```

## Project Structure

```
apps/frontend/
├── src/
│   ├── app/                # Next.js App Router pages
│   │   ├── page.tsx       # Dashboard
│   │   ├── studies/       # Studies management
│   │   ├── ml/            # ML inference
│   │   ├── admin/         # Admin dashboard
│   │   └── worklists/     # Worklist management
│   ├── components/        # React components
│   │   ├── ui/            # Reusable UI components
│   │   └── DicomViewer.tsx # DICOM viewer component
│   ├── lib/               # Utilities and helpers
│   │   ├── api-client.ts  # API client
│   │   └── utils.ts       # Helper functions
│   ├── types/             # TypeScript type definitions
│   ├── hooks/             # Custom React hooks
│   └── styles/            # Global styles
├── public/                # Static assets
├── Dockerfile            # Docker configuration
├── next.config.js        # Next.js configuration
├── tailwind.config.js    # Tailwind CSS configuration
└── tsconfig.json         # TypeScript configuration
```

## Key Components

### Dashboard (/)
- System metrics and statistics
- Recent studies
- Recent activities
- Quick actions

### Studies Management (/studies)
- Browse all medical imaging studies
- Search and filter
- Upload new studies
- View DICOM images
- Run AI analysis

### ML Inference (/ml)
- Available AI models
- Model performance metrics
- Run inference on studies
- View inference results
- Confidence scores

### DICOM Viewer Component
- Interactive image viewing
- Pan, zoom, rotate tools
- Window/level adjustment
- Measurement tools
- Series navigation
- Export functionality

## API Integration

The frontend communicates with the backend through a centralized API client:

```typescript
import apiClient from '@/lib/api-client'

// Get studies
const studies = await apiClient.getStudies()

// Run ML inference
const result = await apiClient.runInference(studyId, modelId)

// Upload DICOM files
const upload = await apiClient.uploadStudy(files)
```

All API calls automatically include:
- Authentication tokens
- Error handling
- Request/response interceptors
- Type safety

## Environment Variables

### Required
- `NEXT_PUBLIC_API_URL`: Backend API URL
- `NEXT_PUBLIC_KEYCLOAK_URL`: Keycloak server URL
- `NEXT_PUBLIC_KEYCLOAK_REALM`: Keycloak realm name
- `NEXT_PUBLIC_KEYCLOAK_CLIENT_ID`: Frontend client ID

### Optional
- `NEXT_PUBLIC_WS_URL`: WebSocket URL for real-time updates
- `NEXT_PUBLIC_ENABLE_DICOM_VIEWER`: Enable DICOM viewer
- `NEXT_PUBLIC_ENABLE_ML_INFERENCE`: Enable ML features
- `NEXT_PUBLIC_ENABLE_3D_RENDERING`: Enable 3D rendering

## Development

### Available Scripts

```bash
npm run dev          # Start development server
npm run build        # Build for production
npm run start        # Start production server
npm run lint         # Run ESLint
npm run type-check   # Run TypeScript compiler
```

### Code Style

- Use TypeScript for all new files
- Follow ESLint configuration
- Use Prettier for formatting
- Write meaningful component and function names
- Add JSDoc comments for complex logic

### Adding New Pages

1. Create page in `src/app/[route]/page.tsx`
2. Add types in `src/types/index.ts`
3. Add API methods in `src/lib/api-client.ts`
4. Create components in `src/components/`
5. Update navigation in layout

### Creating UI Components

Use the shadcn/ui pattern for consistency:

```typescript
import { cn } from '@/lib/utils'

interface ComponentProps {
  className?: string
  // ... other props
}

export function Component({ className, ...props }: ComponentProps) {
  return (
    <div className={cn('base-classes', className)} {...props}>
      {/* component content */}
    </div>
  )
}
```

## Features Roadmap

### Current (v1.0)
- ✅ Dashboard with metrics
- ✅ Studies management
- ✅ ML inference interface
- ✅ DICOM viewer (basic)
- ✅ API integration
- ✅ Authentication

### Planned (v1.1)
- [ ] Advanced DICOM tools (ROI, measurements)
- [ ] 3D volume rendering
- [ ] Real-time collaboration
- [ ] Annotation tools
- [ ] Report generation
- [ ] WebSocket integration for live updates

### Future (v2.0)
- [ ] Mobile app (React Native)
- [ ] Offline mode
- [ ] Voice commands
- [ ] AR/VR visualization
- [ ] Advanced AI features

## Performance

### Optimization Techniques
- Next.js Image optimization
- Code splitting and lazy loading
- React Server Components
- Memoization for expensive operations
- Debouncing for search inputs
- Virtualized lists for large datasets

### Bundle Size
- Main bundle: ~200KB (gzipped)
- Total initial load: ~500KB (gzipped)
- Lighthouse score: 90+

## Browser Support

- Chrome/Edge (latest)
- Firefox (latest)
- Safari (latest)
- Mobile browsers (iOS Safari, Chrome Mobile)

## Troubleshooting

### Common Issues

**CORS errors:**
- Ensure backend CORS settings include frontend URL
- Check `NEXT_PUBLIC_API_URL` is correct

**Authentication failures:**
- Verify Keycloak configuration
- Check client ID and realm settings
- Ensure cookies are enabled

**DICOM viewer not loading:**
- Check browser console for errors
- Verify Cornerstone.js is properly initialized
- Ensure DICOM files are accessible

### Debug Mode

Enable debug logging:
```bash
DEBUG=* npm run dev
```

## Contributing

1. Create feature branch
2. Make changes
3. Write tests
4. Submit pull request

## License

See main project LICENSE file.

## Support

For issues and questions:
- Check documentation: [INTEGRATION_GUIDE.md](../../INTEGRATION_GUIDE.md)
- Review API contracts: [API_CONTRACTS.md](../../API_CONTRACTS.md)
- Submit issues on GitHub

---

**Built with ❤️ for the Aurelius Medical Imaging Platform**
