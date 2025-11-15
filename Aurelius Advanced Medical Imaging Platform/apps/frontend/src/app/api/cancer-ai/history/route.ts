import { NextResponse } from 'next/server';

export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = parseInt(searchParams.get('limit') || '50');
    const offset = parseInt(searchParams.get('offset') || '0');
    const filterType = searchParams.get('filterType');
    const searchTerm = searchParams.get('search');

    // In production, fetch from database with filters
    // const predictions = await db.predictions.find({ ... });

    // Mock data
    const predictions = [
      {
        id: 1,
        patientId: 'P-2024-001',
        imageName: 'chest_ct_001.dcm',
        prediction: 'Lung Cancer',
        confidence: 94.5,
        risk: 'High',
        date: '2025-11-15 10:30',
        modality: 'CT',
        clinician: 'Dr. Smith'
      },
      {
        id: 2,
        patientId: 'P-2024-002',
        imageName: 'mammo_002.dcm',
        prediction: 'Breast Cancer',
        confidence: 89.2,
        risk: 'Medium',
        date: '2025-11-15 09:15',
        modality: 'Mammography',
        clinician: 'Dr. Johnson'
      },
      {
        id: 3,
        patientId: 'P-2024-003',
        imageName: 'prostate_mri_003.dcm',
        prediction: 'No Cancer',
        confidence: 96.8,
        risk: 'Low',
        date: '2025-11-14 16:45',
        modality: 'MRI',
        clinician: 'Dr. Williams'
      },
      {
        id: 4,
        patientId: 'P-2024-004',
        imageName: 'colon_ct_004.dcm',
        prediction: 'Colorectal Cancer',
        confidence: 91.3,
        risk: 'High',
        date: '2025-11-14 14:20',
        modality: 'CT',
        clinician: 'Dr. Brown'
      },
      {
        id: 5,
        patientId: 'P-2024-005',
        imageName: 'lung_xray_005.png',
        prediction: 'No Cancer',
        confidence: 88.7,
        risk: 'Low',
        date: '2025-11-14 11:10',
        modality: 'X-Ray',
        clinician: 'Dr. Davis'
      }
    ];

    return NextResponse.json({
      predictions,
      total: predictions.length,
      limit,
      offset
    });
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to fetch prediction history' },
      { status: 500 }
    );
  }
}

export async function DELETE(request: Request) {
  try {
    const { predictionId } = await request.json();

    // In production, delete from database
    // await db.predictions.delete({ id: predictionId });

    return NextResponse.json({
      success: true,
      message: 'Prediction deleted successfully'
    });
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to delete prediction' },
      { status: 500 }
    );
  }
}
