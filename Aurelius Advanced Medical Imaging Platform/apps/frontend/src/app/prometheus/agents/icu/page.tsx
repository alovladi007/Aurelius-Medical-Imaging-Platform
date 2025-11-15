"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Heart,
  Activity,
  Droplets,
  Wind,
  Thermometer,
  AlertTriangle,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  TrendingDown,
  Zap,
  Shield,
  Brain,
  BarChart3
} from 'lucide-react';

export default function ICUAgentPage() {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const stats = {
    activePatients: 89,
    criticalAlerts: 3,
    earlyWarnings: 7,
    avgResponseTime: '0.8s',
    sepsisAUROC: 0.89
  };

  const patients = [
    {
      id: 'ICU-001',
      name: 'John Doe',
      bed: 'Bed 12',
      age: 67,
      diagnosis: 'Septic shock',
      status: 'critical',
      vitals: {
        hr: 118,
        bp: '88/54',
        map: 65,
        rr: 28,
        spo2: 91,
        temp: 101.2,
        gcs: 12
      },
      trends: {
        hr: 'up',
        bp: 'down',
        rr: 'up',
        spo2: 'down'
      },
      alerts: [
        { type: 'sepsis', score: 0.92, message: 'High sepsis risk - qSOFA 2, lactate 4.2', severity: 'critical' },
        { type: 'hypotension', score: 0.85, message: 'MAP <65, may need vasopressor escalation', severity: 'warning' }
      ],
      suggestions: [
        'Consider norepinephrine dose increase (current 0.1 mcg/kg/min)',
        'Repeat lactate in 2h to assess resuscitation',
        'Check ScvO2 if central line available'
      ],
      lastUpdate: '30s ago'
    },
    {
      id: 'ICU-002',
      name: 'Jane Smith',
      bed: 'Bed 8',
      age: 54,
      diagnosis: 'ARDS post-COVID',
      status: 'stable',
      vitals: {
        hr: 92,
        bp: '118/72',
        map: 87,
        rr: 22,
        spo2: 94,
        temp: 99.1,
        gcs: 15
      },
      trends: {
        hr: 'stable',
        bp: 'stable',
        rr: 'down',
        spo2: 'up'
      },
      alerts: [
        { type: 'ventilation', score: 0.68, message: 'P/F ratio improving, wean watch', severity: 'info' }
      ],
      suggestions: [
        'Consider spontaneous breathing trial if no sedation',
        'P/F ratio 210 - improving trend',
        'Continue lung-protective ventilation'
      ],
      lastUpdate: '15s ago'
    },
    {
      id: 'ICU-003',
      name: 'Robert Johnson',
      bed: 'Bed 15',
      age: 71,
      diagnosis: 'Post-op cardiac surgery',
      status: 'warning',
      vitals: {
        hr: 132,
        bp: '145/92',
        map: 110,
        rr: 20,
        spo2: 96,
        temp: 98.8,
        gcs: 14
      },
      trends: {
        hr: 'up',
        bp: 'up',
        rr: 'stable',
        spo2: 'stable'
      },
      alerts: [
        { type: 'afib', score: 0.78, message: 'New onset A-fib with RVR', severity: 'warning' }
      ],
      suggestions: [
        'Rate control needed - consider beta-blocker or amiodarone',
        'Check TSH, electrolytes (K, Mg)',
        'Consider anticoagulation per CHA₂DS₂-VASc if persistent'
      ],
      lastUpdate: '45s ago'
    }
  ];

  const earlyWarningSystems = [
    {
      name: 'Sepsis Predictor',
      auroc: 0.89,
      ppv: 0.72,
      sensitivity: 0.91,
      lookAhead: '6 hours',
      features: ['HR', 'Temp', 'WBC', 'Lactate', 'BP', 'GCS', 'Platelets']
    },
    {
      name: 'Decompensation Risk',
      auroc: 0.87,
      ppv: 0.68,
      sensitivity: 0.88,
      lookAhead: '4 hours',
      features: ['Vital trends', 'Lab trajectory', 'Fluid balance', 'Vasopressor dose']
    },
    {
      name: 'Extubation Failure',
      auroc: 0.83,
      ppv: 0.65,
      sensitivity: 0.84,
      lookAhead: '48 hours',
      features: ['RSBI', 'GCS', 'Secretions', 'Cuff leak', 'Prior failure']
    }
  ];

  const closedLoopSuggestions = {
    title: 'Closed-Loop Suggestions (Never Auto-Execute)',
    note: 'All suggestions require explicit clinician approval',
    examples: [
      {
        condition: 'Hypotension (MAP <65)',
        suggestion: 'Increase norepinephrine by 0.05 mcg/kg/min',
        rationale: 'MAP 62 despite fluid bolus. Current dose 0.1 mcg/kg/min.',
        confidence: 0.82
      },
      {
        condition: 'Hyperglycemia (BG 245)',
        suggestion: 'Insulin infusion rate: 6 units/h (up from 4 units/h)',
        rationale: 'BG rising trend. Rate of change +15 mg/dL/h.',
        confidence: 0.76
      },
      {
        condition: 'Hypokalemia (K 3.1)',
        suggestion: 'KCl 20 mEq IV over 2h',
        rationale: 'K trending down, on diuretics. Repeat K after replacement.',
        confidence: 0.88
      }
    ]
  };

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Heart className="h-8 w-8 text-red-600 animate-pulse" />
            ICU Agent
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Streaming vitals, early warning, closed-loop suggestions (human-in-loop only)
          </p>
        </div>
        <div className="flex gap-3">
          <div className="text-right">
            <p className="text-sm text-gray-600">Current Time</p>
            <p className="font-mono text-lg font-bold">{currentTime.toLocaleTimeString()}</p>
          </div>
          <Button variant="outline">
            <BarChart3 className="h-4 w-4 mr-2" />
            Analytics
          </Button>
          <Button>
            <Shield className="h-4 w-4 mr-2" />
            Safety Monitor
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Active Patients</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-primary">{stats.activePatients}</p>
          </CardContent>
        </Card>
        <Card className="border-red-500">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Critical Alerts</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-red-600">{stats.criticalAlerts}</p>
            <p className="text-xs text-gray-600">Require immediate action</p>
          </CardContent>
        </Card>
        <Card className="border-yellow-500">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Early Warnings</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-yellow-600">{stats.earlyWarnings}</p>
            <p className="text-xs text-gray-600">Predictive alerts</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Response Time</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-green-600">{stats.avgResponseTime}</p>
            <p className="text-xs text-gray-600">Real-time inference</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Sepsis AUROC</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-purple-600">{stats.sepsisAUROC}</p>
            <p className="text-xs text-gray-600">6h look-ahead</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Patient Monitoring */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Real-Time Patient Monitoring</CardTitle>
              <CardDescription>Streaming vitals with AI-powered early warning</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {patients.map((patient) => (
                  <Card
                    key={patient.id}
                    className={`${
                      patient.status === 'critical' ? 'border-red-500 border-2' :
                      patient.status === 'warning' ? 'border-yellow-500 border-2' :
                      'border-green-500'
                    }`}
                  >
                    <CardContent className="pt-6">
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <Badge variant="outline">{patient.id}</Badge>
                            <Badge variant="secondary">{patient.bed}</Badge>
                            <Badge className={
                              patient.status === 'critical' ? 'bg-red-600' :
                              patient.status === 'warning' ? 'bg-yellow-600' :
                              'bg-green-600'
                            }>
                              {patient.status.toUpperCase()}
                            </Badge>
                            <span className="text-xs text-gray-500">{patient.lastUpdate}</span>
                          </div>
                          <p className="font-semibold text-lg">{patient.name}, {patient.age}y</p>
                          <p className="text-sm text-gray-600">{patient.diagnosis}</p>
                        </div>
                      </div>

                      {/* Vitals Grid */}
                      <div className="grid grid-cols-4 gap-3 mb-4 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                        <div className="text-center">
                          <div className="flex items-center justify-center gap-1 mb-1">
                            <Activity className="h-4 w-4 text-pink-600" />
                            {patient.trends.hr === 'up' && <TrendingUp className="h-3 w-3 text-red-600" />}
                            {patient.trends.hr === 'down' && <TrendingDown className="h-3 w-3 text-blue-600" />}
                          </div>
                          <p className="text-xs text-gray-600">HR</p>
                          <p className={`font-bold ${patient.vitals.hr > 110 || patient.vitals.hr < 60 ? 'text-red-600' : ''}`}>
                            {patient.vitals.hr}
                          </p>
                        </div>
                        <div className="text-center">
                          <div className="flex items-center justify-center gap-1 mb-1">
                            <Heart className="h-4 w-4 text-red-600" />
                            {patient.trends.bp === 'up' && <TrendingUp className="h-3 w-3 text-blue-600" />}
                            {patient.trends.bp === 'down' && <TrendingDown className="h-3 w-3 text-red-600" />}
                          </div>
                          <p className="text-xs text-gray-600">BP</p>
                          <p className={`text-sm font-bold ${patient.vitals.map < 65 ? 'text-red-600' : ''}`}>
                            {patient.vitals.bp}
                          </p>
                          <p className="text-xs text-gray-500">MAP {patient.vitals.map}</p>
                        </div>
                        <div className="text-center">
                          <div className="flex items-center justify-center gap-1 mb-1">
                            <Wind className="h-4 w-4 text-blue-600" />
                            {patient.trends.rr === 'up' && <TrendingUp className="h-3 w-3 text-red-600" />}
                            {patient.trends.rr === 'down' && <TrendingDown className="h-3 w-3 text-green-600" />}
                          </div>
                          <p className="text-xs text-gray-600">RR</p>
                          <p className={`font-bold ${patient.vitals.rr > 24 ? 'text-red-600' : ''}`}>
                            {patient.vitals.rr}
                          </p>
                        </div>
                        <div className="text-center">
                          <div className="flex items-center justify-center gap-1 mb-1">
                            <Droplets className="h-4 w-4 text-blue-600" />
                            {patient.trends.spo2 === 'up' && <TrendingUp className="h-3 w-3 text-green-600" />}
                            {patient.trends.spo2 === 'down' && <TrendingDown className="h-3 w-3 text-red-600" />}
                          </div>
                          <p className="text-xs text-gray-600">SpO2</p>
                          <p className={`font-bold ${patient.vitals.spo2 < 92 ? 'text-red-600' : ''}`}>
                            {patient.vitals.spo2}%
                          </p>
                        </div>
                      </div>

                      {/* Alerts */}
                      {patient.alerts.map((alert, idx) => (
                        <div
                          key={idx}
                          className={`mb-3 p-3 rounded-lg ${
                            alert.severity === 'critical' ? 'bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-900' :
                            alert.severity === 'warning' ? 'bg-yellow-50 dark:bg-yellow-950 border border-yellow-200 dark:border-yellow-900' :
                            'bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-900'
                          }`}
                        >
                          <div className="flex items-start gap-2">
                            {alert.severity === 'critical' && <AlertTriangle className="h-5 w-5 text-red-600 flex-shrink-0" />}
                            {alert.severity === 'warning' && <AlertCircle className="h-5 w-5 text-yellow-600 flex-shrink-0" />}
                            {alert.severity === 'info' && <Zap className="h-5 w-5 text-blue-600 flex-shrink-0" />}
                            <div className="flex-1">
                              <div className="flex items-center justify-between mb-1">
                                <p className="font-semibold text-sm capitalize">{alert.type} Alert</p>
                                <Badge variant="outline" className="text-xs">
                                  {Math.round(alert.score * 100)}% confidence
                                </Badge>
                              </div>
                              <p className="text-sm">{alert.message}</p>
                            </div>
                          </div>
                        </div>
                      ))}

                      {/* Suggestions */}
                      <div className="p-3 bg-purple-50 dark:bg-purple-950 rounded-lg border border-purple-200 dark:border-purple-900">
                        <div className="flex items-start gap-2">
                          <Brain className="h-5 w-5 text-purple-600 flex-shrink-0 mt-0.5" />
                          <div className="flex-1">
                            <p className="font-semibold text-sm mb-2">AI Suggestions:</p>
                            <ul className="space-y-1">
                              {patient.suggestions.map((suggestion, i) => (
                                <li key={i} className="text-xs flex items-start gap-2">
                                  <span className="text-purple-600">•</span>
                                  <span>{suggestion}</span>
                                </li>
                              ))}
                            </ul>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Early Warning Systems & Closed-Loop */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Early Warning Systems</CardTitle>
              <CardDescription>Predictive models with validation metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {earlyWarningSystems.map((system, idx) => (
                  <div key={idx} className="p-4 border rounded-lg">
                    <p className="font-semibold mb-3">{system.name}</p>
                    <div className="grid grid-cols-2 gap-2 text-xs mb-3">
                      <div>
                        <p className="text-gray-600">AUROC</p>
                        <p className="font-bold">{system.auroc}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">PPV</p>
                        <p className="font-bold">{system.ppv}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Sensitivity</p>
                        <p className="font-bold">{system.sensitivity}</p>
                      </div>
                      <div>
                        <p className="text-gray-600">Look-ahead</p>
                        <p className="font-bold">{system.lookAhead}</p>
                      </div>
                    </div>
                    <div>
                      <p className="text-xs text-gray-600 mb-1">Key Features:</p>
                      <div className="flex flex-wrap gap-1">
                        {system.features.map((feature, i) => (
                          <Badge key={i} variant="secondary" className="text-xs">
                            {feature}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="border-purple-200 dark:border-purple-900">
            <CardHeader>
              <CardTitle className="text-sm">{closedLoopSuggestions.title}</CardTitle>
              <CardDescription className="flex items-start gap-2">
                <Shield className="h-4 w-4 text-orange-600 flex-shrink-0 mt-0.5" />
                <span className="text-xs">{closedLoopSuggestions.note}</span>
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {closedLoopSuggestions.examples.map((example, idx) => (
                  <div key={idx} className="p-3 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <p className="font-semibold text-sm">{example.condition}</p>
                      <Badge variant="outline" className="text-xs">
                        {Math.round(example.confidence * 100)}%
                      </Badge>
                    </div>
                    <div className="p-2 bg-purple-50 dark:bg-purple-950 rounded mb-2">
                      <p className="text-xs font-semibold">Suggestion:</p>
                      <p className="text-xs">{example.suggestion}</p>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">{example.rationale}</p>
                    <Button size="sm" variant="outline" className="w-full mt-2 text-xs">
                      Review & Approve
                    </Button>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="border-red-200 dark:border-red-900">
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Shield className="h-4 w-4 text-red-600" />
                Safety Constraints
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-xs">
                <div className="p-2 bg-red-50 dark:bg-red-950 rounded">
                  <p className="font-semibold">Never Auto-Execute</p>
                  <p className="text-gray-600 dark:text-gray-400">All suggestions require explicit human approval</p>
                </div>
                <div className="p-2 bg-yellow-50 dark:bg-yellow-950 rounded">
                  <p className="font-semibold">Guardrails Active</p>
                  <p className="text-gray-600 dark:text-gray-400">Dose limits, contraindication checks, allergy screening</p>
                </div>
                <div className="p-2 bg-blue-50 dark:bg-blue-950 rounded">
                  <p className="font-semibold">Audit Trail</p>
                  <p className="text-gray-600 dark:text-gray-400">Every suggestion logged with timestamp and rationale</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
