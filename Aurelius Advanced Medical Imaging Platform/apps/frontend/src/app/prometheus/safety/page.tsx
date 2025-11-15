"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Shield,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Activity,
  TrendingUp,
  Users,
  Lock,
  Eye,
  FileText,
  BarChart3,
  AlertCircle
} from 'lucide-react';

export default function SafetyDashboardPage() {
  const [timeRange, setTimeRange] = useState('24h');

  const safetyMetrics = {
    totalQueries: 15847,
    safetyBlocks: 12,
    uncertaintyAbstentions: 145,
    humanOverrides: 23,
    policyViolations: 0,
    phiExposures: 0,
    biasAlerts: 8,
    calibrationScore: 0.95
  };

  const recentBlocks = [
    {
      id: 1,
      timestamp: '2025-11-15 14:23',
      agent: 'Therapy Planner',
      query: 'Narcotic dosing request',
      reason: 'Missing weight and creatinine clearance',
      severity: 'high',
      resolved: false
    },
    {
      id: 2,
      timestamp: '2025-11-15 13:45',
      agent: 'Diagnostic Copilot',
      query: 'Cancer diagnosis suggestion',
      reason: 'Uncertainty >30%, insufficient data',
      severity: 'medium',
      resolved: true
    },
    {
      id: 3,
      timestamp: '2025-11-15 12:18',
      agent: 'Research Copilot',
      query: 'Patient cohort query',
      reason: 'Purpose-of-use violation (research not authorized)',
      severity: 'high',
      resolved: true
    }
  ];

  const biasMetrics = [
    { subgroup: 'Age <40', accuracy: 94.8, parity: 0.98, sampleSize: 2341 },
    { subgroup: 'Age 40-65', accuracy: 95.2, parity: 1.00, sampleSize: 8934 },
    { subgroup: 'Age >65', accuracy: 94.1, parity: 0.97, sampleSize: 4572 },
    { subgroup: 'Male', accuracy: 94.6, parity: 0.99, sampleSize: 7823 },
    { subgroup: 'Female', accuracy: 95.1, parity: 1.00, sampleSize: 8024 },
    { subgroup: 'Race: White', accuracy: 94.9, parity: 1.00, sampleSize: 10234 },
    { subgroup: 'Race: Black', accuracy: 94.2, parity: 0.98, sampleSize: 3456 },
    { subgroup: 'Race: Hispanic', accuracy: 94.7, parity: 0.99, sampleSize: 1543 },
    { subgroup: 'Race: Asian', accuracy: 95.3, parity: 1.01, sampleSize: 614 }
  ];

  const provenanceExamples = [
    {
      claim: 'Dual antiplatelet therapy recommended for NSTEMI',
      sources: [
        { type: 'guideline', title: 'ACC/AHA NSTEMI Guidelines 2021', level: '1A', url: '#' },
        { type: 'trial', title: 'CURE Trial (Clopidogrel + ASA)', level: '1A', url: '#' },
        { type: 'ucg', title: 'Patient UCG node: Prior MI, stent 2022', level: 'Patient Data', url: '#' }
      ]
    },
    {
      claim: 'CrCl 45, reduce metformin dose to 500mg daily',
      sources: [
        { type: 'guideline', title: 'FDA Metformin Dosing Guideline', level: '1A', url: '#' },
        { type: 'calculation', title: 'Cockcroft-Gault: CrCl 45 mL/min', level: 'Computed', url: '#' },
        { type: 'ucg', title: 'Patient labs: SCr 1.8', level: 'Patient Data', url: '#' }
      ]
    }
  ];

  const neverRules = [
    { rule: 'No narcotic dosing without weight + CrCl', violations: 0, triggers: 23 },
    { rule: 'No chemotherapy dosing without BSA', violations: 0, triggers: 12 },
    { rule: 'No anticoagulation without renal function', violations: 0, triggers: 34 },
    { rule: 'No PII/PHI in research queries', violations: 0, triggers: 8 },
    { rule: 'No clinical decisions without human oversight', violations: 0, triggers: 15847 }
  ];

  const offlineEval = [
    { task: 'MedQA', score: 87.3, benchmark: '85th percentile', status: 'pass' },
    { task: 'MedMCQA', score: 84.1, benchmark: '80th percentile', status: 'pass' },
    { task: 'PubMedQA', score: 91.2, benchmark: '90th percentile', status: 'pass' },
    { task: 'CheXpert (AUROC)', score: 0.96, benchmark: '≥0.95', status: 'pass' },
    { task: 'MIMIC-IV Sepsis (AUROC)', score: 0.89, benchmark: '≥0.85', status: 'pass' },
    { task: 'Guideline Concordance', score: 94.8, benchmark: '≥90%', status: 'pass' }
  ];

  const driftMonitoring = [
    { metric: 'Diagnostic Accuracy', current: 94.8, baseline: 95.1, drift: -0.3, alert: false },
    { metric: 'Prescription Safety', current: 98.9, baseline: 98.7, drift: +0.2, alert: false },
    { metric: 'Lab Interpretation', current: 96.2, baseline: 96.5, drift: -0.3, alert: false },
    { metric: 'Radiology Reads', current: 93.8, baseline: 95.2, drift: -1.4, alert: true },
    { metric: 'Response Time (s)', current: 1.9, baseline: 1.8, drift: +0.1, alert: false }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Shield className="h-8 w-8 text-primary" />
            Safety, Governance & Evaluation
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Policy engine, provenance, calibration, bias monitoring, and continuous evaluation
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <FileText className="h-4 w-4 mr-2" />
            Incident Report
          </Button>
          <Button>
            <AlertTriangle className="h-4 w-4 mr-2" />
            Review Alerts
          </Button>
        </div>
      </div>

      {/* Safety Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Total Queries</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{safetyMetrics.totalQueries.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card className={safetyMetrics.safetyBlocks === 0 ? 'border-green-500' : 'border-yellow-500'}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Safety Blocks</CardTitle>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-bold ${safetyMetrics.safetyBlocks === 0 ? 'text-green-600' : 'text-yellow-600'}`}>
              {safetyMetrics.safetyBlocks}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Abstentions</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-orange-600">{safetyMetrics.uncertaintyAbstentions}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Overrides</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">{safetyMetrics.humanOverrides}</p>
          </CardContent>
        </Card>
        <Card className={safetyMetrics.policyViolations === 0 ? 'border-green-500' : 'border-red-500'}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Policy Violations</CardTitle>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-bold ${safetyMetrics.policyViolations === 0 ? 'text-green-600' : 'text-red-600'}`}>
              {safetyMetrics.policyViolations}
            </p>
          </CardContent>
        </Card>
        <Card className={safetyMetrics.phiExposures === 0 ? 'border-green-500' : 'border-red-500'}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">PHI Exposures</CardTitle>
          </CardHeader>
          <CardContent>
            <p className={`text-2xl font-bold ${safetyMetrics.phiExposures === 0 ? 'text-green-600' : 'text-red-600'}`}>
              {safetyMetrics.phiExposures}
            </p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Bias Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-yellow-600">{safetyMetrics.biasAlerts}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Calibration</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-purple-600">{safetyMetrics.calibrationScore}</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Safety Blocks */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Shield className="h-5 w-5" />
              Recent Safety Blocks
            </CardTitle>
            <CardDescription>Automatic safety interventions in the last {timeRange}</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {recentBlocks.map((block) => (
                <div key={block.id} className="p-4 border rounded-lg">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <Badge variant={block.severity === 'high' ? 'destructive' : 'default'}>
                          {block.severity}
                        </Badge>
                        <p className="text-xs text-gray-600">{block.timestamp}</p>
                      </div>
                      <p className="font-semibold text-sm">{block.agent}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{block.query}</p>
                    </div>
                    {block.resolved ? (
                      <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0" />
                    ) : (
                      <AlertCircle className="h-5 w-5 text-yellow-600 flex-shrink-0" />
                    )}
                  </div>
                  <div className="p-2 bg-red-50 dark:bg-red-950 rounded text-xs mt-2">
                    <p className="font-semibold text-red-900 dark:text-red-100">Reason:</p>
                    <p className="text-red-800 dark:text-red-200">{block.reason}</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* "Never" Rules Enforcement */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lock className="h-5 w-5" />
              "Never" Rules Enforcement
            </CardTitle>
            <CardDescription>Hard safety constraints - zero violations allowed</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {neverRules.map((rule, index) => (
                <div key={index} className="p-3 border rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <p className="font-semibold text-sm flex-1">{rule.rule}</p>
                    {rule.violations === 0 ? (
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    ) : (
                      <XCircle className="h-5 w-5 text-red-600" />
                    )}
                  </div>
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-600">Violations:</span>
                    <span className={`font-bold ${rule.violations === 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {rule.violations}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-xs mt-1">
                    <span className="text-gray-600">Triggers (prevented):</span>
                    <span className="font-semibold">{rule.triggers}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Provenance Examples */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Eye className="h-5 w-5" />
            Provenance Everywhere
          </CardTitle>
          <CardDescription>Every claim linked to sources + timestamps + evidence strength</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {provenanceExamples.map((example, index) => (
              <div key={index} className="p-4 border rounded-lg">
                <p className="font-semibold mb-3">{example.claim}</p>
                <div className="space-y-2">
                  <p className="text-xs font-semibold text-gray-600 dark:text-gray-400">Sources:</p>
                  {example.sources.map((source, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" className="text-xs">{source.type}</Badge>
                        <p className="text-sm">{source.title}</p>
                      </div>
                      <Badge variant="secondary">{source.level}</Badge>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Bias & Fairness Monitoring */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5" />
            Bias & Fairness Metrics
          </CardTitle>
          <CardDescription>Subgroup performance parity monitoring</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {biasMetrics.map((metric, index) => (
              <div key={index} className="flex items-center justify-between p-3 border rounded-lg">
                <div className="flex-1">
                  <p className="font-semibold text-sm">{metric.subgroup}</p>
                  <p className="text-xs text-gray-600">n = {metric.sampleSize.toLocaleString()}</p>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-right">
                    <p className="text-sm font-semibold">{metric.accuracy}%</p>
                    <p className="text-xs text-gray-600">Accuracy</p>
                  </div>
                  <div className="text-right">
                    <Badge variant={metric.parity >= 0.95 ? 'default' : 'destructive'}>
                      {metric.parity.toFixed(2)} parity
                    </Badge>
                  </div>
                  {metric.parity >= 0.95 ? (
                    <CheckCircle className="h-5 w-5 text-green-600" />
                  ) : (
                    <AlertCircle className="h-5 w-5 text-red-600" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="offline">
        <TabsList>
          <TabsTrigger value="offline">Offline Evaluation</TabsTrigger>
          <TabsTrigger value="drift">Drift Monitoring</TabsTrigger>
        </TabsList>

        {/* Offline Evaluation */}
        <TabsContent value="offline">
          <Card>
            <CardHeader>
              <CardTitle>Offline Evaluation Suites</CardTitle>
              <CardDescription>Benchmark performance on standard medical AI tasks</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {offlineEval.map((task, index) => (
                  <div key={index} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <p className="font-semibold">{task.task}</p>
                      {task.status === 'pass' ? (
                        <Badge className="bg-green-600">PASS</Badge>
                      ) : (
                        <Badge className="bg-red-600">FAIL</Badge>
                      )}
                    </div>
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-gray-600">Score:</span>
                      <span className="font-bold text-primary">{task.score}</span>
                    </div>
                    <div className="flex items-center justify-between text-sm mt-1">
                      <span className="text-gray-600">Benchmark:</span>
                      <span className="font-semibold">{task.benchmark}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Drift Monitoring */}
        <TabsContent value="drift">
          <Card>
            <CardHeader>
              <CardTitle>Model Drift Monitoring</CardTitle>
              <CardDescription>Real-time performance vs baseline - alerts on >1% degradation</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {driftMonitoring.map((metric, index) => (
                  <div key={index} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <p className="font-semibold">{metric.metric}</p>
                      {metric.alert ? (
                        <Badge className="bg-red-600">
                          <AlertTriangle className="h-3 w-3 mr-1" />
                          ALERT
                        </Badge>
                      ) : (
                        <Badge className="bg-green-600">Normal</Badge>
                      )}
                    </div>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <p className="text-gray-600">Current</p>
                        <p className="font-bold text-primary">{metric.current}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Baseline</p>
                        <p className="font-semibold">{metric.baseline}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Drift</p>
                        <p className={`font-bold ${metric.drift < 0 && Math.abs(metric.drift) > 1 ? 'text-red-600' : metric.drift > 0 ? 'text-green-600' : 'text-gray-600'}`}>
                          {metric.drift > 0 ? '+' : ''}{metric.drift}
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
