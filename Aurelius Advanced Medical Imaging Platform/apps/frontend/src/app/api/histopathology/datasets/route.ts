import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // Mock data - in production, fetch from backend
    return NextResponse.json({
      datasets: [
        {
          id: 1,
          name: 'Brain Cancer MRI',
          path: '/data/raw/brain_cancer',
          numClasses: 4,
          totalImages: 7023,
          status: 'ready',
          createdAt: '2025-11-10'
        },
        {
          id: 2,
          name: 'Lung Histopathology',
          path: '/data/raw/lung',
          numClasses: 2,
          totalImages: 1248,
          status: 'processing',
          createdAt: '2025-11-14'
        },
        {
          id: 3,
          name: 'Synthetic Test Data',
          path: '/data/raw/synthetic',
          numClasses: 2,
          totalImages: 400,
          status: 'ready',
          createdAt: '2025-11-15'
        }
      ]
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to load datasets' }, { status: 500 });
  }
}
