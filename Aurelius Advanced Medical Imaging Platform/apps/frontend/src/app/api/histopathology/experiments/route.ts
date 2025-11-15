import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // Mock data - in production, fetch from MLflow backend
    return NextResponse.json({
      experiments: [
        {
          id: 'exp-001',
          name: 'ResNet-50 Brain Cancer',
          model: 'resnet50',
          status: 'completed',
          accuracy: 95.6,
          precision: 94.2,
          recall: 93.8,
          f1Score: 94.0,
          trainLoss: 0.145,
          valLoss: 0.168,
          epochs: 50,
          batchSize: 32,
          learningRate: 0.001,
          duration: '2h 15m',
          startTime: '2025-11-14 10:00',
          endTime: '2025-11-14 12:15'
        },
        {
          id: 'exp-002',
          name: 'EfficientNet-B3 Lung',
          model: 'efficientnet_b3',
          status: 'running',
          accuracy: 92.1,
          epochs: 30,
          currentEpoch: 18,
          batchSize: 16,
          learningRate: 0.0005,
          duration: '1h 30m',
          startTime: '2025-11-15 09:00'
        }
      ]
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to load experiments' }, { status: 500 });
  }
}
