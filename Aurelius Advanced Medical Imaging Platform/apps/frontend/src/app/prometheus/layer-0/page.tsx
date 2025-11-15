"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Shield,
  Cpu,
  Server,
  Network,
  Lock,
  Activity,
  Database,
  Cloud,
  HardDrive,
  Zap,
  CheckCircle,
  AlertCircle,
  TrendingUp,
  BarChart3
} from 'lucide-react';

export default function Layer0Page() {
  const [computeMetrics, setComputeMetrics] = useState({
    gpuNodes: 24,
    cpuUtilization: 67,
    gpuUtilization: 82,
    memoryUsed: 145,
    memoryTotal: 192,
    activePods: 156,
    storageUsed: 2.4e12,
    storageTotal: 5e12
  });

  const [securityMetrics, setSecurityMetrics] = useState({
    complianceScore: 98.5,
    auditEvents: 12483,
    policyViolations: 0,
    activeConnections: 47,
    encryptionStatus: 'full',
    lastAudit: '2 hours ago'
  });

  useEffect(() => {
    loadMetrics();
  }, []);

  const loadMetrics = async () => {
    try {
      const response = await fetch('/api/prometheus/layer-0/metrics');
      if (response.ok) {
        const data = await response.json();
        setComputeMetrics(data.compute);
        setSecurityMetrics(data.security);
      }
    } catch (error) {
      console.error('Error loading metrics:', error);
    }
  };

  const computeResources = [
    {
      name: 'GPU Nodes',
      value: computeMetrics.gpuNodes,
      icon: Cpu,
      status: 'healthy',
      details: '24 NVIDIA A100 80GB'
    },
    {
      name: 'CPU Utilization',
      value: `${computeMetrics.cpuUtilization}%`,
      icon: Activity,
      status: 'healthy',
      details: 'Across 48 cores'
    },
    {
      name: 'GPU Utilization',
      value: `${computeMetrics.gpuUtilization}%`,
      icon: Zap,
      status: 'warning',
      details: 'High load on training cluster'
    },
    {
      name: 'Memory Usage',
      value: `${computeMetrics.memoryUsed}/${computeMetrics.memoryTotal} GB`,
      icon: HardDrive,
      status: 'healthy',
      details: `${((computeMetrics.memoryUsed / computeMetrics.memoryTotal) * 100).toFixed(1)}% used`
    }
  ];

  const storageComponents = [
    {
      name: 'Delta Lake',
      type: 'Tabular Data',
      size: '1.2 TB',
      encryption: 'AES-256',
      status: 'operational'
    },
    {
      name: 'Object Store',
      type: 'Blobs (Images, Docs)',
      size: '800 GB',
      encryption: 'AES-256',
      status: 'operational'
    },
    {
      name: 'VNA/PACS',
      type: 'DICOM Archive',
      size: '400 GB',
      encryption: 'AES-256',
      status: 'operational'
    },
    {
      name: 'HSM/KMS',
      type: 'Secrets Management',
      size: '< 1 GB',
      encryption: 'Hardware',
      status: 'operational'
    }
  ];

  const securityFeatures = [
    {
      name: 'Zero-Trust Network',
      enabled: true,
      description: 'mTLS between all services'
    },
    {
      name: 'RBAC + ABAC',
      enabled: true,
      description: 'Role & attribute-based access'
    },
    {
      name: 'OPA Policy Engine',
      enabled: true,
      description: 'Rego policy enforcement'
    },
    {
      name: 'W3C PROV Lineage',
      enabled: true,
      description: 'Full data provenance tracking'
    },
    {
      name: 'DLP Scanning',
      enabled: true,
      description: 'Ingress/egress PHI detection'
    },
    {
      name: 'Immutable Audit Logs',
      enabled: true,
      description: `${securityMetrics.auditEvents.toLocaleString()} events logged`
    }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Shield className="h-8 w-8 text-primary" />
            Layer 0: Secure Data & Compute Plane
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            PHI/PII safe infrastructure with hybrid Kubernetes
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <BarChart3 className="h-4 w-4 mr-2" />
            Detailed Metrics
          </Button>
          <Button>
            <Zap className="h-4 w-4 mr-2" />
            Scale Resources
          </Button>
        </div>
      </div>

      {/* Compliance Alert */}
      <Card className="border-green-200 bg-green-50 dark:bg-green-950">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <CheckCircle className="h-5 w-5 text-green-600" />
            <CardTitle className="text-green-900 dark:text-green-100">
              HIPAA Compliance: {securityMetrics.complianceScore}%
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent className="text-sm text-green-800 dark:text-green-200">
          All security controls operational • Zero policy violations • Encryption verified • Last audit: {securityMetrics.lastAudit}
        </CardContent>
      </Card>

      {/* Compute Resources */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Compute Infrastructure</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {computeResources.map((resource, index) => (
            <Card key={index}>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">{resource.name}</CardTitle>
                <resource.icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{resource.value}</div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">{resource.details}</p>
                <Badge variant={resource.status === 'healthy' ? 'default' : 'secondary'} className="mt-2">
                  {resource.status}
                </Badge>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Kubernetes Cluster */}
      <Card>
        <CardHeader>
          <CardTitle>Kubernetes Cluster Status</CardTitle>
          <CardDescription>Hybrid on-prem + cloud autoscaling with KEDA</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="overview">
            <TabsList>
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="nodes">Nodes</TabsTrigger>
              <TabsTrigger value="workloads">Workloads</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-4">
              <div className="grid grid-cols-3 gap-6">
                <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <Server className="h-8 w-8 text-blue-600 mx-auto mb-2" />
                  <p className="text-2xl font-bold">24</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">GPU Nodes</p>
                </div>
                <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                  <Cloud className="h-8 w-8 text-green-600 mx-auto mb-2" />
                  <p className="text-2xl font-bold">8</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Cloud Nodes</p>
                </div>
                <div className="text-center p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                  <Activity className="h-8 w-8 text-purple-600 mx-auto mb-2" />
                  <p className="text-2xl font-bold">{computeMetrics.activePods}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Active Pods</p>
                </div>
              </div>
              <div className="mt-4">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Resource Utilization</p>
                <div className="space-y-2">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>CPU</span>
                      <span>{computeMetrics.cpuUtilization}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                      <div className="bg-blue-600 h-2 rounded-full" style={{ width: `${computeMetrics.cpuUtilization}%` }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span>GPU</span>
                      <span>{computeMetrics.gpuUtilization}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                      <div className="bg-green-600 h-2 rounded-full" style={{ width: `${computeMetrics.gpuUtilization}%` }} />
                    </div>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="nodes">
              <div className="space-y-2">
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>On-Premises GPU Nodes:</strong> 24 nodes with NVIDIA A100 80GB
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>Cloud TPU Nodes:</strong> 8 nodes with Google TPU v4
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-400">
                  <strong>Autoscaling:</strong> KEDA active, scaling based on queue depth
                </div>
              </div>
            </TabsContent>

            <TabsContent value="workloads">
              <div className="space-y-2 text-sm">
                <p><strong>Ray Clusters:</strong> 3 active (batch processing)</p>
                <p><strong>Spark Jobs:</strong> 12 running (data transformation)</p>
                <p><strong>gRPC Services:</strong> 45 pods (online inference)</p>
                <p><strong>REST APIs:</strong> 28 pods (web services)</p>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Storage Systems */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Storage Infrastructure</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {storageComponents.map((storage, index) => (
            <Card key={index}>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg">{storage.name}</CardTitle>
                  <Badge variant={storage.status === 'operational' ? 'default' : 'secondary'}>
                    {storage.status}
                  </Badge>
                </div>
                <CardDescription>{storage.type}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span>Size:</span>
                    <span className="font-mono">{storage.size}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span>Encryption:</span>
                    <span className="font-mono">{storage.encryption}</span>
                  </div>
                  <div className="flex items-center gap-2 mt-3">
                    <Lock className="h-4 w-4 text-green-600" />
                    <span className="text-xs text-green-600">Encrypted at rest & in transit</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Security & Compliance */}
      <Card>
        <CardHeader>
          <CardTitle>Security & Compliance Features</CardTitle>
          <CardDescription>Zero-trust architecture with comprehensive auditing</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {securityFeatures.map((feature, index) => (
              <div key={index} className="flex items-start gap-3 p-4 border rounded-lg">
                {feature.enabled ? (
                  <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                ) : (
                  <AlertCircle className="h-5 w-5 text-gray-400 flex-shrink-0 mt-0.5" />
                )}
                <div>
                  <p className="font-semibold">{feature.name}</p>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{feature.description}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Network Security */}
      <Card>
        <CardHeader>
          <CardTitle>Network Security</CardTitle>
          <CardDescription>Private service connect with zero-trust policies</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="text-center p-4 border rounded-lg">
                <Network className="h-6 w-6 mx-auto mb-2 text-blue-600" />
                <p className="text-lg font-bold">mTLS</p>
                <p className="text-xs text-gray-600">All Services</p>
              </div>
              <div className="text-center p-4 border rounded-lg">
                <Shield className="h-6 w-6 mx-auto mb-2 text-green-600" />
                <p className="text-lg font-bold">VPC Isolated</p>
                <p className="text-xs text-gray-600">Train vs Inference</p>
              </div>
              <div className="text-center p-4 border rounded-lg">
                <Lock className="h-6 w-6 mx-auto mb-2 text-purple-600" />
                <p className="text-lg font-bold">{securityMetrics.activeConnections}</p>
                <p className="text-xs text-gray-600">Active Connections</p>
              </div>
              <div className="text-center p-4 border rounded-lg">
                <CheckCircle className="h-6 w-6 mx-auto mb-2 text-orange-600" />
                <p className="text-lg font-bold">0</p>
                <p className="text-xs text-gray-600">Policy Violations</p>
              </div>
            </div>
            <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
              <p className="text-sm font-semibold mb-2">PHI Isolation</p>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                • Separate VPCs for training and inference workloads<br />
                • Hardened de-identification services<br />
                • DLP scanning at ingress/egress<br />
                • Row-level access control with Lakehouse ACLs
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
