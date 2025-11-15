import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // In production, fetch from actual backend
    // const response = await fetch('http://localhost:8000/api/histopathology/dashboard');
    // return NextResponse.json(await response.json());

    // Mock data for development
    return NextResponse.json({
      stats: {
        totalDatasets: 3,
        totalImages: 8671,
        trainedModels: 5,
        activeExperiments: 2,
        totalFeatures: 127,
        avgAccuracy: 94.2
      },
      recentExperiments: [
        {
          id: 1,
          name: 'ResNet-50 Brain Cancer',
          status: 'completed',
          accuracy: 95.6,
          date: '2025-11-14'
        },
        {
          id: 2,
          name: 'EfficientNet-B3 Lung',
          status: 'running',
          accuracy: 92.1,
          date: '2025-11-15'
        }
      ]
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to load dashboard data' }, { status: 500 });
  }
}
