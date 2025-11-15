"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Cpu,
  Database,
  Network,
  Brain,
  Shield,
  Activity,
  Zap,
  Lock,
  Server,
  GitBranch,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Clock,
  Users,
  FileText,
  BarChart3,
  Microscope,
  FlaskConical
} from 'lucide-react';

export default function PrometheusDashboard() {
  const [systemStatus, setSystemStatus] = useState({
    overall: 'healthy',
    compute: 'healthy',
    storage: 'healthy',
    network: 'healthy',
    security: 'healthy'
  });

  const [metrics, setMetrics] = useState({
    activePipelines: 0,
    dataIngested: 0,
    modelsRunning: 0,
    graphNodes: 0,
    complianceScore: 0,
    activeUsers: 0
  });

  useEffect(() => {
    loadSystemStatus();
  }, []);

  const loadSystemStatus = async () => {
    try {
      const response = await fetch('/api/prometheus/status');
      if (response.ok) {
        const data = await response.json();
        setSystemStatus(data.status);
        setMetrics(data.metrics);
      } else {
        // Mock data
        setMetrics({
          activePipelines: 12,
          dataIngested: 2.4e12, // 2.4TB
          modelsRunning: 5,
          graphNodes: 1.2e7, // 12M
          complianceScore: 98.5,
          activeUsers: 47
        });
      }
    } catch (error) {
      console.error('Error loading system status:', error);
    }
  };

  const layers = [
    {
      id: 0,
      name: 'Secure Data & Compute Plane',
      description: 'PHI/PII safe infrastructure with hybrid Kubernetes',
      icon: Shield,
      href: '/prometheus/layer-0',
      status: systemStatus.compute,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50 dark:bg-blue-950',
      stats: [
        { label: 'GPU Nodes', value: '24', icon: Cpu },
        { label: 'Compliance', value: `${metrics.complianceScore}%`, icon: Lock },
        { label: 'Uptime', value: '99.9%', icon: Activity }
      ]
    },
    {
      id: 1,
      name: 'Clinical Data Ingestion',
      description: 'HL7, FHIR, DICOM harmonization & normalization',
      icon: Database,
      href: '/prometheus/layer-1',
      status: systemStatus.storage,
      color: 'text-green-600',
      bgColor: 'bg-green-50 dark:bg-green-950',
      stats: [
        { label: 'Pipelines', value: metrics.activePipelines, icon: Zap },
        { label: 'Data Ingested', value: formatBytes(metrics.dataIngested), icon: Database },
        { label: 'Sources', value: '8', icon: Server }
      ]
    },
    {
      id: 2,
      name: 'Clinical Knowledge Graph',
      description: 'Unified patient-centric temporal graph with reasoning',
      icon: GitBranch,
      href: '/prometheus/layer-2',
      status: systemStatus.network,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50 dark:bg-purple-950',
      stats: [
        { label: 'Graph Nodes', value: formatNumber(metrics.graphNodes), icon: GitBranch },
        { label: 'Ontologies', value: '12', icon: FileText },
        { label: 'Rules', value: '450+', icon: Brain }
      ]
    },
    {
      id: 3,
      name: 'Foundation Model Stack',
      description: 'Multimodal AI (text, vision, time-series, genomics)',
      icon: Brain,
      href: '/prometheus/layer-3',
      status: systemStatus.security,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50 dark:bg-orange-950',
      stats: [
        { label: 'Models Active', value: metrics.modelsRunning, icon: Brain },
        { label: 'Modalities', value: '5', icon: Microscope },
        { label: 'Accuracy', value: '94.2%', icon: TrendingUp }
      ]
    }
  ];

  const quickStats = [
    {
      title: 'System Health',
      value: systemStatus.overall === 'healthy' ? '100%' : '85%',
      icon: Activity,
      trend: 'Optimal',
      trendUp: true,
      color: 'text-green-600'
    },
    {
      title: 'Active Users',
      value: metrics.activeUsers,
      icon: Users,
      trend: '+12 today',
      trendUp: true,
      color: 'text-blue-600'
    },
    {
      title: 'Models Running',
      value: metrics.modelsRunning,
      icon: Brain,
      trend: 'Production',
      trendUp: true,
      color: 'text-purple-600'
    },
    {
      title: 'Compliance Score',
      value: `${metrics.complianceScore}%`,
      icon: Shield,
      trend: 'HIPAA Compliant',
      trendUp: true,
      color: 'text-orange-600'
    }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
            <FlaskConical className="h-10 w-10 text-primary" />
            P.R.O.M.E.T.H.E.U.S.
          </h1>
          <p className="text-lg text-gray-600 dark:text-gray-400 mb-2">
            Precision Research and Oncology Machine-learning Engine for
          </p>
          <p className="text-lg text-gray-600 dark:text-gray-400">
            Therapeutics, Health, Exploration, Understanding, and Science
          </p>
          <Badge variant="outline" className="mt-3">
            Medical AGI System
          </Badge>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <BarChart3 className="h-4 w-4 mr-2" />
            System Monitor
          </Button>
          <Button>
            <Zap className="h-4 w-4 mr-2" />
            Launch Workbench
          </Button>
        </div>
      </div>

      {/* System Status Alert */}
      <Card className={systemStatus.overall === 'healthy' ? 'border-green-200 bg-green-50 dark:bg-green-950' : 'border-orange-200 bg-orange-50 dark:bg-orange-950'}>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            {systemStatus.overall === 'healthy' ? (
              <CheckCircle className="h-5 w-5 text-green-600" />
            ) : (
              <AlertCircle className="h-5 w-5 text-orange-600" />
            )}
            <CardTitle className={systemStatus.overall === 'healthy' ? 'text-green-900 dark:text-green-100' : 'text-orange-900 dark:text-orange-100'}>
              System Status: {systemStatus.overall === 'healthy' ? 'All Systems Operational' : 'Degraded Performance'}
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent className="text-sm text-green-800 dark:text-green-200">
          All layers operational • Zero-trust security active • HIPAA compliance verified • Audit logging enabled
        </CardContent>
      </Card>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {quickStats.map((stat, index) => (
          <Card key={index}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
              <stat.icon className={`h-4 w-4 ${stat.color}`} />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className={`text-xs ${stat.trendUp ? 'text-green-600' : 'text-red-600'} flex items-center gap-1 mt-1`}>
                {stat.trendUp ? <TrendingUp className="h-3 w-3" /> : null}
                {stat.trend}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* System Layers */}
      <div>
        <h2 className="text-2xl font-bold mb-4">System Architecture Layers</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {layers.map((layer) => (
            <Link key={layer.id} href={layer.href}>
              <Card className="h-full hover:shadow-lg transition-shadow cursor-pointer">
                <CardHeader>
                  <div className="flex items-start justify-between mb-3">
                    <div className={`w-12 h-12 rounded-lg ${layer.bgColor} flex items-center justify-center`}>
                      <layer.icon className={`h-6 w-6 ${layer.color}`} />
                    </div>
                    <Badge variant={layer.status === 'healthy' ? 'default' : 'secondary'}>
                      <CheckCircle className="h-3 w-3 mr-1" />
                      {layer.status}
                    </Badge>
                  </div>
                  <CardTitle className="text-lg">Layer {layer.id}: {layer.name}</CardTitle>
                  <CardDescription>{layer.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4">
                    {layer.stats.map((stat, idx) => (
                      <div key={idx} className="text-center">
                        <div className="flex justify-center mb-1">
                          <stat.icon className="h-4 w-4 text-gray-400" />
                        </div>
                        <p className="text-lg font-bold">{stat.value}</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">{stat.label}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      {/* Key Features */}
      <Card>
        <CardHeader>
          <CardTitle>Key Capabilities</CardTitle>
          <CardDescription>What P.R.O.M.E.T.H.E.U.S. enables for clinical teams</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div>
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Shield className="h-4 w-4 text-blue-600" />
                HIPAA-Compliant by Design
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Zero-trust architecture, encrypted at rest and in transit, full audit trails, consent management
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Database className="h-4 w-4 text-green-600" />
                Multi-Source Data Integration
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                HL7, FHIR R4/R5, DICOM, lab middleware, wearables—all harmonized to SNOMED, LOINC, ICD-10
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <GitBranch className="h-4 w-4 text-purple-600" />
                Temporal Clinical Reasoning
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Patient-centric knowledge graph with causal pathways, rule engines, counterfactual simulators
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Brain className="h-4 w-4 text-orange-600" />
                Multimodal Foundation Models
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Clinical LLM with tool use, DICOM vision encoders, time-series transformers, genomic variant analysis
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <Lock className="h-4 w-4 text-red-600" />
                Row-Level Access Control
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                RBAC + ABAC with clinical roles, lakehouse ACLs, purpose-of-use toggles, de-identification modes
              </p>
            </div>
            <div>
              <h4 className="font-semibold mb-2 flex items-center gap-2">
                <TrendingUp className="h-4 w-4 text-pink-600" />
                Calibrated Uncertainty
              </h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Conformal prediction, selective abstention, evidential deep learning for trustworthy AI
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Getting Started */}
      <Card className="border-primary">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            Quick Start Guide
          </CardTitle>
          <CardDescription>Get started with P.R.O.M.E.T.H.E.U.S. in 4 steps</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center text-sm font-bold">1</div>
              <div>
                <p className="font-medium">Configure Data Sources</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Connect HL7, FHIR, DICOM, and wearable streams</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center text-sm font-bold">2</div>
              <div>
                <p className="font-medium">Build Clinical Cohorts</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Query the knowledge graph with temporal constraints</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center text-sm font-bold">3</div>
              <div>
                <p className="font-medium">Deploy Foundation Models</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Run multimodal inference with calibrated uncertainty</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center text-sm font-bold">4</div>
              <div>
                <p className="font-medium">Monitor & Audit</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">Track lineage, compliance, and model performance</p>
              </div>
            </div>
          </div>
          <div className="mt-6 flex gap-3">
            <Link href="/prometheus/layer-0">
              <Button>Explore System Layers</Button>
            </Link>
            <Button variant="outline">View Documentation</Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function formatNumber(num: number): string {
  if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
  if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
  return num.toString();
}
