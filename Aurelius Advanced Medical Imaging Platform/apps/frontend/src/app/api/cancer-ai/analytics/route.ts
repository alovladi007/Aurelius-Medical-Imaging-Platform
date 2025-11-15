import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const timeRange = searchParams.get('timeRange') || '30d';

    // In production, fetch analytics from database
    // const analytics = await db.analytics.aggregate({ timeRange });

    // Mock analytics data
    return NextResponse.json({
      performance: {
        overall: {
          accuracy: 94.2,
          precision: 96.8,
          recall: 93.5,
          f1Score: 95.1,
          auroc: 0.97
        },
        byType: [
          { type: 'Lung Cancer', accuracy: 95.3, samples: 1247 },
          { type: 'Breast Cancer', accuracy: 96.1, samples: 1893 },
          { type: 'Colorectal Cancer', accuracy: 92.8, samples: 876 },
          { type: 'Prostate Cancer', accuracy: 94.5, samples: 1034 },
          { type: 'No Cancer', accuracy: 93.7, samples: 2156 }
        ],
        byModality: [
          { modality: 'CT', accuracy: 94.8, volume: 3421 },
          { modality: 'MRI', accuracy: 95.2, volume: 1876 },
          { modality: 'X-Ray', accuracy: 92.3, volume: 2134 },
          { modality: 'Mammography', accuracy: 96.5, volume: 1893 }
        ]
      },
      trends: {
        monthly: [
          { month: 'Jun', predictions: 856, accuracy: 93.2, avgConfidence: 91.5 },
          { month: 'Jul', predictions: 923, accuracy: 93.8, avgConfidence: 92.1 },
          { month: 'Aug', predictions: 1045, accuracy: 94.1, avgConfidence: 92.8 },
          { month: 'Sep', predictions: 1183, accuracy: 94.5, avgConfidence: 93.2 },
          { month: 'Oct', predictions: 1267, accuracy: 94.2, avgConfidence: 93.5 },
          { month: 'Nov', predictions: 1432, accuracy: 94.7, avgConfidence: 94.1 }
        ]
      },
      distribution: {
        cancerTypes: [
          { type: 'Lung Cancer', count: 1247, percentage: 17.2 },
          { type: 'Breast Cancer', count: 1893, percentage: 26.1 },
          { type: 'Colorectal Cancer', count: 876, percentage: 12.1 },
          { type: 'Prostate Cancer', count: 1034, percentage: 14.3 },
          { type: 'No Cancer', count: 2156, percentage: 29.8 },
          { type: 'Other', count: 100, percentage: 1.4 }
        ],
        confidence: [
          { range: '90-100%', count: 5892, percentage: 81.3 },
          { range: '80-89%', count: 987, percentage: 13.6 },
          { range: '70-79%', count: 287, percentage: 4.0 },
          { range: '< 70%', count: 80, percentage: 1.1 }
        ]
      },
      quality: {
        calibrationScore: 0.95,
        stabilityIndex: 98.2,
        avgResponseTime: 1.2
      }
    });
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch analytics data' },
      { status: 500 }
    );
  }
}
