import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const config = await request.json();

    // In production, forward to Python backend
    // const response = await fetch('http://localhost:8000/api/histopathology/train', {
    //   method: 'POST',
    //   headers: { 'Content-Type': 'application/json' },
    //   body: JSON.stringify(config)
    // });

    // For now, return success
    return NextResponse.json({
      message: 'Training started',
      experimentId: 'exp-' + Date.now(),
      config
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to start training' }, { status: 500 });
  }
}
