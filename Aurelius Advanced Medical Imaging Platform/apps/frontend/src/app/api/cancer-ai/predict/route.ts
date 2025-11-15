import { NextRequest, NextResponse } from 'next/server';

const GATEWAY_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:10200';

export async function POST(request: NextRequest) {
  try {
    // Get the form data from the request
    const formData = await request.formData();

    // Forward to gateway
    const response = await fetch(`${GATEWAY_URL}/cancer-ai/predict`, {
      method: 'POST',
      body: formData,
      headers: {
        // Forward authentication headers if present
        ...(request.headers.get('authorization') && {
          authorization: request.headers.get('authorization')!
        })
      }
    });

    if (!response.ok) {
      const error = await response.text();
      return NextResponse.json(
        { error: error || 'Prediction failed' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Cancer AI prediction error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
