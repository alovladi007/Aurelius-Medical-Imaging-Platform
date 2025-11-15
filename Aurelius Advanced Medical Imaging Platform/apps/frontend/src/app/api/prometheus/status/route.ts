import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // In production, fetch from actual Prometheus backend
    // const response = await fetch('http://localhost:8000/api/prometheus/status');
    // return NextResponse.json(await response.json());

    // Mock data for development
    return NextResponse.json({
      status: {
        overall: 'healthy',
        compute: 'healthy',
        storage: 'healthy',
        network: 'healthy',
        security: 'healthy'
      },
      metrics: {
        activePipelines: 12,
        dataIngested: 2.4e12, // 2.4TB
        modelsRunning: 5,
        graphNodes: 1.2e7, // 12M nodes
        complianceScore: 98.5,
        activeUsers: 47
      }
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to load system status' }, { status: 500 });
  }
}
