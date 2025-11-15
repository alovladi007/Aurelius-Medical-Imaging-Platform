import { NextResponse } from 'next/server';

export async function GET() {
  try {
    return NextResponse.json({
      compute: {
        gpuNodes: 24,
        cpuUtilization: 67,
        gpuUtilization: 82,
        memoryUsed: 145,
        memoryTotal: 192,
        activePods: 156,
        storageUsed: 2.4e12,
        storageTotal: 5e12
      },
      security: {
        complianceScore: 98.5,
        auditEvents: 12483,
        policyViolations: 0,
        activeConnections: 47,
        encryptionStatus: 'full',
        lastAudit: '2 hours ago'
      }
    });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to load metrics' }, { status: 500 });
  }
}
