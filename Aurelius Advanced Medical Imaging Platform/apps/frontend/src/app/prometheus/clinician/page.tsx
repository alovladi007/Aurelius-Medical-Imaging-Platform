"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  User,
  Activity,
  FileText,
  AlertCircle,
  CheckCircle,
  Clock,
  Pill,
  TestTube,
  Heart,
  Brain,
  BookOpen,
  Info,
  TrendingUp,
  Calendar,
  Stethoscope,
  Shield
} from 'lucide-react';

export default function ClinicianUIPage() {
  const [selectedPatient, setSelectedPatient] = useState('P-2024-001');

  const patient = {
    id: 'P-2024-001',
    name: 'John Doe',
    age: 67,
    gender: 'Male',
    mrn: 'MRN-789456',
    location: 'ED Bay 3',
    chiefComplaint: 'Chest pain',
    arrivalTime: '2025-11-15 10:30',
    triageLevel: 2
  };

  const timeline = [
    { time: '10:30', event: 'Patient arrival', type: 'arrival', user: 'Triage RN' },
    { time: '10:35', event: 'Vital signs recorded', type: 'vitals', details: 'BP 145/92, HR 88, RR 18, SpO2 96%' },
    { time: '10:40', event: 'ECG performed', type: 'test', details: 'Sinus rhythm, no acute ST changes' },
    { time: '10:42', event: 'Troponin ordered', type: 'order', user: 'Dr. Smith' },
    { time: '10:45', event: 'HEART score calculated', type: 'calculator', details: 'Score: 4 (Moderate risk)' },
    { time: '10:48', event: 'ACS guideline consulted', type: 'guideline', details: 'Serial troponins recommended' },
    { time: '10:50', event: 'Aspirin 325mg administered', type: 'medication', user: 'RN Johnson' }
  ];

  const problems = [
    { name: 'Acute Coronary Syndrome', status: 'suspected', confidence: 78, evidence: ['Chest pain', 'Risk factors', 'HEART score 4'] },
    { name: 'Hypertension', status: 'chronic', confidence: 100, evidence: ['Known HTN', 'Current BP 145/92'] },
    { name: 'Type 2 Diabetes', status: 'chronic', confidence: 100, evidence: ['A1C 7.8%', 'On metformin'] },
    { name: 'Hyperlipidemia', status: 'chronic', confidence: 100, evidence: ['LDL 145', 'On statin'] }
  ];

  const labs = [
    { test: 'Troponin I', value: '0.04', units: 'ng/mL', flag: 'normal', ref: '<0.05', time: '10:55' },
    { test: 'Glucose', value: '156', units: 'mg/dL', flag: 'high', ref: '70-100', time: '10:55' },
    { test: 'Creatinine', value: '1.1', units: 'mg/dL', flag: 'normal', ref: '0.7-1.3', time: '10:55' },
    { test: 'BNP', value: '85', units: 'pg/mL', flag: 'normal', ref: '<100', time: '10:55' }
  ];

  const medications = [
    { name: 'Aspirin', dose: '81mg', route: 'PO', frequency: 'Daily', status: 'active' },
    { name: 'Metoprolol', dose: '50mg', route: 'PO', frequency: 'BID', status: 'active' },
    { name: 'Metformin', dose: '1000mg', route: 'PO', frequency: 'BID', status: 'active' },
    { name: 'Atorvastatin', dose: '40mg', route: 'PO', frequency: 'QHS', status: 'active' }
  ];

  const guidelineCards = [
    {
      title: 'ACS Management (ESC 2023)',
      recommendation: 'Serial troponins at 0h and 3h for HEART score 4-6',
      strength: 'Class I',
      evidence: 'Level A',
      source: 'ESC Guidelines 2023'
    },
    {
      title: 'Antiplatelet Therapy',
      recommendation: 'Continue aspirin. Consider dual antiplatelet if ACS confirmed',
      strength: 'Class I',
      evidence: 'Level A',
      source: 'ACC/AHA 2021'
    }
  ];

  const draftOrders = [
    { order: 'Troponin I', timing: 'Now and in 3 hours', justification: 'HEART score 4 - serial troponins per ACS pathway', status: 'pending' },
    { order: 'Continue telemetry monitoring', timing: 'Continuous', justification: 'Suspected ACS, chest pain', status: 'pending' },
    { order: 'Cardiology consult', timing: 'If troponin elevated', justification: 'Conditional on positive troponin', status: 'pending' }
  ];

  const uncertaintyBadges = {
    acs: { level: 'moderate', confidence: 78 },
    pe: { level: 'low', confidence: 12 },
    gerd: { level: 'low', confidence: 15 }
  };

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Stethoscope className="h-8 w-8 text-primary" />
            Clinician Workspace
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Patient timeline, uncertainty badges, rationale + citations
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Calendar className="h-4 w-4 mr-2" />
            Schedule
          </Button>
          <Button>
            <User className="h-4 w-4 mr-2" />
            Dr. Smith
          </Button>
        </div>
      </div>

      {/* Patient Banner */}
      <Card className="border-blue-500 border-2">
        <CardContent className="pt-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="w-16 h-16 bg-blue-100 dark:bg-blue-950 rounded-full flex items-center justify-center">
                <User className="h-8 w-8 text-blue-600" />
              </div>
              <div>
                <div className="flex items-center gap-3">
                  <h2 className="text-2xl font-bold">{patient.name}</h2>
                  <Badge variant="outline">{patient.age}y {patient.gender}</Badge>
                  <Badge className="bg-red-600">ESI Level {patient.triageLevel}</Badge>
                </div>
                <p className="text-gray-600 dark:text-gray-400">MRN: {patient.mrn} • Location: {patient.location}</p>
                <p className="font-semibold mt-1">Chief Complaint: {patient.chiefComplaint}</p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-600">Arrival Time</p>
              <p className="font-semibold">{patient.arrivalTime}</p>
              <p className="text-sm text-gray-600 mt-2">Time in ED</p>
              <p className="font-semibold text-orange-600">2h 30min</p>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Panel */}
        <div className="lg:col-span-2 space-y-6">
          {/* Problem List with Uncertainty */}
          <Card>
            <CardHeader>
              <CardTitle>Problems & Differential</CardTitle>
              <CardDescription>AI-assisted problem list with uncertainty quantification</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {problems.map((problem, index) => (
                  <div key={index} className="p-4 border rounded-lg">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <p className="font-semibold">{problem.name}</p>
                          <Badge variant={problem.status === 'suspected' ? 'outline' : 'secondary'}>
                            {problem.status}
                          </Badge>
                          {problem.confidence < 100 && (
                            <Badge variant="outline" className="bg-yellow-50 text-yellow-800 dark:bg-yellow-950">
                              <AlertCircle className="h-3 w-3 mr-1" />
                              {problem.confidence}% confidence
                            </Badge>
                          )}
                        </div>
                        <div className="flex flex-wrap gap-1 mt-2">
                          {problem.evidence.map((ev, idx) => (
                            <Badge key={idx} variant="secondary" className="text-xs">
                              {ev}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      {problem.status === 'suspected' && (
                        <Button variant="outline" size="sm">
                          <Info className="h-4 w-4 mr-1" />
                          Evidence
                        </Button>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Guideline Cards */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <BookOpen className="h-5 w-5" />
                Guideline Recommendations
              </CardTitle>
              <CardDescription>Context-specific evidence-based recommendations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {guidelineCards.map((card, index) => (
                  <div key={index} className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-900">
                    <div className="flex items-start justify-between mb-2">
                      <p className="font-semibold text-sm">{card.title}</p>
                      <div className="flex gap-2">
                        <Badge variant="outline" className="text-xs">{card.strength}</Badge>
                        <Badge variant="outline" className="text-xs">{card.evidence}</Badge>
                      </div>
                    </div>
                    <p className="text-sm mb-2">{card.recommendation}</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      <BookOpen className="h-3 w-3 inline mr-1" />
                      {card.source}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Draft Orders */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Draft Orders
              </CardTitle>
              <CardDescription>Context-aware orders with justifications</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {draftOrders.map((order, index) => (
                  <div key={index} className="p-4 border rounded-lg">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex-1">
                        <p className="font-semibold">{order.order}</p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">{order.timing}</p>
                      </div>
                      <div className="flex gap-2">
                        <Button size="sm" variant="outline">Edit</Button>
                        <Button size="sm">Sign</Button>
                      </div>
                    </div>
                    <div className="p-2 bg-gray-50 dark:bg-gray-900 rounded text-xs">
                      <p className="text-gray-600 dark:text-gray-400 mb-1">Justification:</p>
                      <p>{order.justification}</p>
                    </div>
                  </div>
                ))}
                <Button variant="outline" className="w-full">
                  <FileText className="h-4 w-4 mr-2" />
                  Add Custom Order
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Patient Timeline */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-5 w-5" />
                Patient Timeline
              </CardTitle>
              <CardDescription>Chronological view of all events and interventions</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="relative">
                <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-200 dark:bg-gray-800" />
                <div className="space-y-4">
                  {timeline.map((item, index) => (
                    <div key={index} className="flex gap-4">
                      <div className="relative">
                        <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center z-10">
                          {item.type === 'arrival' && <User className="h-4 w-4 text-white" />}
                          {item.type === 'vitals' && <Activity className="h-4 w-4 text-white" />}
                          {item.type === 'test' && <TestTube className="h-4 w-4 text-white" />}
                          {item.type === 'order' && <FileText className="h-4 w-4 text-white" />}
                          {item.type === 'calculator' && <Brain className="h-4 w-4 text-white" />}
                          {item.type === 'guideline' && <BookOpen className="h-4 w-4 text-white" />}
                          {item.type === 'medication' && <Pill className="h-4 w-4 text-white" />}
                        </div>
                      </div>
                      <div className="flex-1 pb-4">
                        <div className="flex items-center gap-2 mb-1">
                          <p className="text-sm font-semibold">{item.time}</p>
                          <Badge variant="secondary" className="text-xs">{item.type}</Badge>
                        </div>
                        <p className="text-sm">{item.event}</p>
                        {item.details && (
                          <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">{item.details}</p>
                        )}
                        {item.user && (
                          <p className="text-xs text-gray-500 mt-1">By: {item.user}</p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Sidebar */}
        <div className="space-y-6">
          {/* Labs */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <TestTube className="h-5 w-5" />
                Recent Labs
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {labs.map((lab, index) => (
                  <div key={index} className="p-3 border rounded-lg">
                    <div className="flex items-center justify-between mb-1">
                      <p className="font-semibold text-sm">{lab.test}</p>
                      {lab.flag === 'high' ? (
                        <AlertCircle className="h-4 w-4 text-red-600" />
                      ) : (
                        <CheckCircle className="h-4 w-4 text-green-600" />
                      )}
                    </div>
                    <div className="flex items-baseline gap-2">
                      <p className={`text-lg font-bold ${lab.flag === 'high' ? 'text-red-600' : 'text-green-600'}`}>
                        {lab.value}
                      </p>
                      <p className="text-xs text-gray-600">{lab.units}</p>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">Ref: {lab.ref} • {lab.time}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Active Medications */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Pill className="h-5 w-5" />
                Active Medications
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {medications.map((med, index) => (
                  <div key={index} className="p-3 border rounded-lg">
                    <p className="font-semibold text-sm">{med.name}</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">
                      {med.dose} {med.route} {med.frequency}
                    </p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* AI Assistance */}
          <Card className="border-purple-200 dark:border-purple-900">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Brain className="h-5 w-5 text-purple-600" />
                AI Insights
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="p-3 bg-purple-50 dark:bg-purple-950 rounded-lg">
                <p className="text-sm font-semibold mb-1">Diagnostic Support</p>
                <p className="text-xs">HEART score suggests moderate risk. Serial troponins critical for risk stratification.</p>
              </div>
              <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                <p className="text-sm font-semibold mb-1">Why not PE?</p>
                <p className="text-xs">Low pretest probability. No dyspnea, no tachycardia. Wells score for PE would be low.</p>
              </div>
              <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                <p className="text-sm font-semibold mb-1">Medication Safety</p>
                <p className="text-xs">All current medications have appropriate renal dosing for CrCl 75.</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
