"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Activity,
  AlertCircle,
  CheckCircle,
  Clock,
  Heart,
  Thermometer,
  Wind,
  Droplets,
  User,
  FileText,
  TrendingUp,
  Brain,
  Zap
} from 'lucide-react';

export default function TriageAgentPage() {
  const [activePatient, setActivePatient] = useState<any>(null);

  const stats = {
    patientsToday: 342,
    avgTriageTime: '2.3s',
    accuracy: 94.8,
    criticalCaught: 28,
    avgESI: 2.8
  };

  const recentTriage = [
    {
      id: 'P-ED-001',
      chiefComplaint: 'Chest pain',
      vitals: { bp: '145/92', hr: 88, rr: 18, temp: 98.6, spo2: 96 },
      esi: 2,
      differential: ['ACS', 'GERD', 'Anxiety'],
      urgency: 'High',
      orders: ['ECG', 'Troponin', 'CXR'],
      time: '5 min ago',
      status: 'completed'
    },
    {
      id: 'P-ED-002',
      chiefComplaint: 'Abdominal pain',
      vitals: { bp: '110/70', hr: 95, rr: 16, temp: 100.2, spo2: 98 },
      esi: 3,
      differential: ['Appendicitis', 'Gastroenteritis', 'Cholecystitis'],
      urgency: 'Medium',
      orders: ['CBC', 'CMP', 'Lipase', 'Abdominal CT'],
      time: '8 min ago',
      status: 'completed'
    },
    {
      id: 'P-ED-003',
      chiefComplaint: 'Shortness of breath',
      vitals: { bp: '178/95', hr: 112, rr: 26, temp: 98.8, spo2: 89 },
      esi: 1,
      differential: ['PE', 'CHF exacerbation', 'Pneumonia'],
      urgency: 'Critical',
      orders: ['Stat CXR', 'BNP', 'D-dimer', 'ABG', 'CTPA'],
      time: '12 min ago',
      status: 'alert'
    },
    {
      id: 'P-ED-004',
      chiefComplaint: 'Headache',
      vitals: { bp: '132/78', hr: 72, rr: 14, temp: 98.4, spo2: 99 },
      esi: 4,
      differential: ['Migraine', 'Tension headache', 'Sinusitis'],
      urgency: 'Low',
      orders: ['None initially'],
      time: '15 min ago',
      status: 'completed'
    }
  ];

  const triageProtocols = [
    {
      name: 'Chest Pain Protocol',
      triggers: ['chest pain', 'pressure', 'radiating arm pain'],
      esi: 2,
      mandatoryOrders: ['ECG within 10min', 'Troponin', 'Aspirin if no contraindications'],
      redFlags: ['ST elevation', 'Hemodynamic instability', 'Syncope']
    },
    {
      name: 'Sepsis Screening',
      triggers: ['fever + hypotension', 'altered mental status + infection'],
      esi: 1,
      mandatoryOrders: ['Lactate', 'Blood cultures', 'Broad spectrum antibiotics within 1h'],
      redFlags: ['SBP <90', 'Lactate >4', 'AMS']
    },
    {
      name: 'Stroke Alert',
      triggers: ['weakness', 'speech difficulty', 'facial droop'],
      esi: 1,
      mandatoryOrders: ['Stat CT head', 'Glucose', 'Stroke team activation'],
      redFlags: ['Last known well <4.5h', 'Rapidly progressive symptoms']
    }
  ];

  const performanceMetrics = [
    { metric: 'ESI Agreement', value: '96.2%', benchmark: 'vs expert triage nurses', color: 'text-green-600' },
    { metric: 'Critical Miss Rate', value: '0.3%', benchmark: '<1% threshold', color: 'text-green-600' },
    { metric: 'Over-triage', value: '12.1%', benchmark: '<15% acceptable', color: 'text-green-600' },
    { metric: 'Under-triage', value: '2.8%', benchmark: '<5% critical', color: 'text-yellow-600' }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Activity className="h-8 w-8 text-red-600" />
            Triage/Intake Agent
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            ED/Urgent Care symptom parsing → differential → initial orders
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <TrendingUp className="h-4 w-4 mr-2" />
            Performance
          </Button>
          <Button>
            <FileText className="h-4 w-4 mr-2" />
            Protocols
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Patients Today</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-primary">{stats.patientsToday}</p>
            <p className="text-xs text-gray-600 mt-1">Avg ESI: {stats.avgESI}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Triage Time</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-blue-600">{stats.avgTriageTime}</p>
            <p className="text-xs text-green-600">-0.4s vs baseline</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">ESI Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-green-600">{stats.accuracy}%</p>
            <p className="text-xs text-gray-600">vs expert nurses</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Critical Caught</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-orange-600">{stats.criticalCaught}</p>
            <p className="text-xs text-gray-600">ESI 1-2 today</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Status</CardTitle>
          </CardHeader>
          <CardContent>
            <Badge className="bg-green-600">Active</Badge>
            <p className="text-xs text-gray-600 mt-2">All systems operational</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Triage Stream */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Live Triage Stream</CardTitle>
              <CardDescription>Real-time patient triage with AI-assisted decision support</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentTriage.map((patient, idx) => (
                  <Card
                    key={patient.id}
                    className={`cursor-pointer transition-all ${
                      patient.status === 'alert' ? 'border-red-500 border-2' : 'hover:shadow-lg'
                    }`}
                    onClick={() => setActivePatient(patient)}
                  >
                    <CardContent className="pt-6">
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <Badge variant="outline">{patient.id}</Badge>
                            <Badge className={
                              patient.esi === 1 ? 'bg-red-600' :
                              patient.esi === 2 ? 'bg-orange-600' :
                              patient.esi === 3 ? 'bg-yellow-600' :
                              'bg-green-600'
                            }>
                              ESI {patient.esi}
                            </Badge>
                            {patient.status === 'alert' && (
                              <Badge className="bg-red-600 animate-pulse">
                                <AlertCircle className="h-3 w-3 mr-1" />
                                CRITICAL
                              </Badge>
                            )}
                            <span className="text-xs text-gray-500">{patient.time}</span>
                          </div>
                          <p className="font-semibold text-lg">{patient.chiefComplaint}</p>
                        </div>
                        {patient.status === 'completed' ? (
                          <CheckCircle className="h-5 w-5 text-green-600" />
                        ) : (
                          <Clock className="h-5 w-5 text-orange-600 animate-spin" />
                        )}
                      </div>

                      {/* Vitals */}
                      <div className="grid grid-cols-5 gap-3 mb-4 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                        <div className="text-center">
                          <Heart className="h-4 w-4 mx-auto mb-1 text-red-600" />
                          <p className="text-xs text-gray-600">BP</p>
                          <p className="font-semibold text-sm">{patient.vitals.bp}</p>
                        </div>
                        <div className="text-center">
                          <Activity className="h-4 w-4 mx-auto mb-1 text-pink-600" />
                          <p className="text-xs text-gray-600">HR</p>
                          <p className="font-semibold text-sm">{patient.vitals.hr}</p>
                        </div>
                        <div className="text-center">
                          <Wind className="h-4 w-4 mx-auto mb-1 text-blue-600" />
                          <p className="text-xs text-gray-600">RR</p>
                          <p className="font-semibold text-sm">{patient.vitals.rr}</p>
                        </div>
                        <div className="text-center">
                          <Thermometer className="h-4 w-4 mx-auto mb-1 text-orange-600" />
                          <p className="text-xs text-gray-600">Temp</p>
                          <p className="font-semibold text-sm">{patient.vitals.temp}</p>
                        </div>
                        <div className="text-center">
                          <Droplets className="h-4 w-4 mx-auto mb-1 text-blue-600" />
                          <p className="text-xs text-gray-600">SpO2</p>
                          <p className={`font-semibold text-sm ${patient.vitals.spo2 < 92 ? 'text-red-600' : ''}`}>
                            {patient.vitals.spo2}%
                          </p>
                        </div>
                      </div>

                      {/* Differential */}
                      <div className="mb-3">
                        <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-2">
                          <Brain className="h-3 w-3 inline mr-1" />
                          AI Differential:
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {patient.differential.map((dx, i) => (
                            <Badge key={i} variant={i === 0 ? 'default' : 'secondary'} className="text-xs">
                              {dx}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      {/* Orders */}
                      <div>
                        <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-2">
                          <Zap className="h-3 w-3 inline mr-1" />
                          Initial Orders:
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {patient.orders.map((order, i) => (
                            <Badge key={i} variant="outline" className="text-xs">
                              {order}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Performance Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Performance Metrics</CardTitle>
              <CardDescription>Validation against expert triage nurses</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {performanceMetrics.map((metric, idx) => (
                  <div key={idx} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <p className="font-semibold text-sm">{metric.metric}</p>
                      <p className={`text-2xl font-bold ${metric.color}`}>{metric.value}</p>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">{metric.benchmark}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Triage Protocols */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Triage Protocols</CardTitle>
              <CardDescription>Evidence-based triage algorithms</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {triageProtocols.map((protocol, idx) => (
                  <div key={idx} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <p className="font-semibold">{protocol.name}</p>
                      <Badge className={
                        protocol.esi === 1 ? 'bg-red-600' :
                        protocol.esi === 2 ? 'bg-orange-600' :
                        'bg-yellow-600'
                      }>
                        ESI {protocol.esi}
                      </Badge>
                    </div>

                    <div className="space-y-3 text-xs">
                      <div>
                        <p className="font-semibold text-gray-600 mb-1">Triggers:</p>
                        <div className="flex flex-wrap gap-1">
                          {protocol.triggers.map((t, i) => (
                            <Badge key={i} variant="secondary" className="text-xs">{t}</Badge>
                          ))}
                        </div>
                      </div>

                      <div>
                        <p className="font-semibold text-gray-600 mb-1">Mandatory Orders:</p>
                        <ul className="list-disc ml-4 space-y-1">
                          {protocol.mandatoryOrders.map((order, i) => (
                            <li key={i}>{order}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="p-2 bg-red-50 dark:bg-red-950 rounded">
                        <p className="font-semibold text-red-900 dark:text-red-100 mb-1">Red Flags:</p>
                        <ul className="list-disc ml-4 space-y-1 text-red-800 dark:text-red-200">
                          {protocol.redFlags.map((flag, i) => (
                            <li key={i}>{flag}</li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="border-blue-200 dark:border-blue-900">
            <CardHeader>
              <CardTitle className="text-sm">Agent Capabilities</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 text-xs">
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold">NLP Symptom Parsing</p>
                    <p className="text-gray-600 dark:text-gray-400">Extracts chief complaint, HPI, ROS from free text</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold">Vital Sign Analysis</p>
                    <p className="text-gray-600 dark:text-gray-400">Detects abnormal vitals, calculates NEWS/MEWS</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold">ESI Assignment</p>
                    <p className="text-gray-600 dark:text-gray-400">Evidence-based 5-level emergency severity index</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold">Differential Generation</p>
                    <p className="text-gray-600 dark:text-gray-400">Top 3-5 diagnoses ranked by probability</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold">Protocol Matching</p>
                    <p className="text-gray-600 dark:text-gray-400">Auto-activates chest pain, stroke, sepsis protocols</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
