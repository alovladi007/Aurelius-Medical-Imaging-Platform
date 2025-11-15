"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  BookOpen,
  Search,
  CheckCircle,
  AlertCircle,
  Info,
  FileText,
  TrendingUp,
  Heart,
  Activity,
  Pill,
  Shield
} from 'lucide-react';

export default function GuidelinesEnginePage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedGuideline, setSelectedGuideline] = useState<any>(null);
  const [patientContext, setPatientContext] = useState({
    age: 67,
    gender: 'male',
    conditions: ['ACS', 'HTN', 'DM2'],
    meds: ['Aspirin', 'Metoprolol', 'Metformin']
  });

  const guidelines = [
    {
      id: 'acs-2023',
      title: 'Acute Coronary Syndrome Management',
      organization: 'ESC',
      year: 2023,
      specialty: 'Cardiology',
      applicable: true,
      recommendations: [
        {
          condition: 'NSTEMI with HEART score 4-6',
          recommendation: 'Serial troponins at 0h and 3h',
          class: 'I',
          loe: 'A',
          rationale: 'High sensitivity troponin protocols reduce time to diagnosis',
          contraindications: []
        },
        {
          condition: 'Confirmed ACS',
          recommendation: 'Dual antiplatelet therapy (Aspirin + P2Y12 inhibitor)',
          class: 'I',
          loe: 'A',
          rationale: 'CURE trial showed 20% reduction in CV events',
          contraindications: ['Active bleeding', 'Severe thrombocytopenia', 'Recent ICH']
        },
        {
          condition: 'NSTEMI high risk',
          recommendation: 'Invasive strategy within 24h',
          class: 'I',
          loe: 'A',
          rationale: 'Early intervention improves outcomes in high-risk patients',
          contraindications: []
        }
      ]
    },
    {
      id: 'sepsis-ssc-2021',
      title: 'Surviving Sepsis Campaign Bundle',
      organization: 'SSC',
      year: 2021,
      specialty: 'Critical Care',
      applicable: false,
      recommendations: [
        {
          condition: 'Sepsis suspected',
          recommendation: 'Measure lactate within 1 hour',
          class: 'I',
          loe: 'B',
          rationale: 'Lactate >2 mmol/L indicates tissue hypoperfusion',
          contraindications: []
        },
        {
          condition: 'Septic shock',
          recommendation: 'Broad-spectrum antibiotics within 1 hour',
          class: 'I',
          loe: 'A',
          rationale: 'Every hour delay increases mortality by 7.6%',
          contraindications: []
        },
        {
          condition: 'Septic shock',
          recommendation: '30 mL/kg crystalloid bolus',
          class: 'I',
          loe: 'B',
          rationale: 'Early aggressive fluid resuscitation',
          contraindications: ['Pulmonary edema', 'Severe heart failure']
        }
      ]
    },
    {
      id: 'dm-ada-2024',
      title: 'Diabetes Management Standards',
      organization: 'ADA',
      year: 2024,
      specialty: 'Endocrinology',
      applicable: true,
      recommendations: [
        {
          condition: 'T2DM + CVD',
          recommendation: 'GLP-1 RA or SGLT2i with proven CV benefit',
          class: 'I',
          loe: 'A',
          rationale: 'CVOT trials show CV mortality reduction',
          contraindications: ['eGFR <30 for SGLT2i']
        },
        {
          condition: 'T2DM',
          recommendation: 'A1C target <7% for most adults',
          class: 'I',
          loe: 'A',
          rationale: 'Microvascular risk reduction',
          contraindications: []
        }
      ]
    },
    {
      id: 'htn-aha-2023',
      title: 'Hypertension Management',
      organization: 'ACC/AHA',
      year: 2023,
      specialty: 'Cardiology',
      applicable: true,
      recommendations: [
        {
          condition: 'Stage 2 HTN (BP ≥140/90)',
          recommendation: 'Initiate 2-drug combination therapy',
          class: 'I',
          loe: 'A',
          rationale: 'Faster BP control with combination therapy',
          contraindications: []
        },
        {
          condition: 'HTN + CAD',
          recommendation: 'Beta-blocker + ACE-I/ARB preferred',
          class: 'I',
          loe: 'A',
          rationale: 'Dual cardioprotection',
          contraindications: []
        }
      ]
    },
    {
      id: 'dvt-accp-2021',
      title: 'VTE Prophylaxis Guidelines',
      organization: 'ACCP',
      year: 2021,
      specialty: 'Hematology',
      applicable: false,
      recommendations: [
        {
          condition: 'Hospitalized medical patients',
          recommendation: 'Pharmacologic VTE prophylaxis if low bleeding risk',
          class: 'I',
          loe: 'A',
          rationale: '50% reduction in VTE events',
          contraindications: ['Active bleeding', 'PLT <50k', 'Recent neurosurgery']
        }
      ]
    }
  ];

  const executeCQL = () => {
    // Simulate CQL execution to generate patient-specific recommendations
    const applicable = guidelines.filter(g => g.applicable);
    setSelectedGuideline(applicable[0]);
  };

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <BookOpen className="h-8 w-8 text-primary" />
            Guideline Engine
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            CQL executor with patient-specific recommendations + strength/grade of evidence
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <FileText className="h-4 w-4 mr-2" />
            All Guidelines (230)
          </Button>
          <Button onClick={executeCQL}>
            <Activity className="h-4 w-4 mr-2" />
            Execute CQL
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Guideline Library */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Guideline Library</CardTitle>
              <CardDescription>Evidence-based clinical pathways</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3 mb-4">
                <Input
                  placeholder="Search guidelines..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full"
                />
              </div>

              <div className="space-y-2">
                {guidelines.map((guideline) => (
                  <div
                    key={guideline.id}
                    className={`p-3 border-2 rounded-lg cursor-pointer transition-all ${
                      selectedGuideline?.id === guideline.id
                        ? 'border-primary bg-primary/5'
                        : guideline.applicable
                        ? 'border-green-200 dark:border-green-900 hover:border-primary/50'
                        : 'border-gray-200 dark:border-gray-800 hover:border-primary/50'
                    }`}
                    onClick={() => setSelectedGuideline(guideline)}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <p className="font-semibold text-sm flex-1">{guideline.title}</p>
                      {guideline.applicable && (
                        <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 ml-2" />
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">{guideline.organization}</Badge>
                      <Badge variant="secondary" className="text-xs">{guideline.year}</Badge>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">{guideline.specialty}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Patient Context */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Patient Context</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-xs">
              <div className="flex justify-between">
                <span className="text-gray-600">Age:</span>
                <span className="font-semibold">{patientContext.age}y</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Gender:</span>
                <span className="font-semibold capitalize">{patientContext.gender}</span>
              </div>
              <div>
                <p className="text-gray-600 mb-1">Conditions:</p>
                <div className="flex flex-wrap gap-1">
                  {patientContext.conditions.map((c, i) => (
                    <Badge key={i} variant="outline" className="text-xs">{c}</Badge>
                  ))}
                </div>
              </div>
              <div>
                <p className="text-gray-600 mb-1">Medications:</p>
                <div className="flex flex-wrap gap-1">
                  {patientContext.meds.map((m, i) => (
                    <Badge key={i} variant="secondary" className="text-xs">{m}</Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Guideline Details */}
        <div className="lg:col-span-2 space-y-6">
          {selectedGuideline ? (
            <>
              <Card>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle>{selectedGuideline.title}</CardTitle>
                      <CardDescription className="mt-2">
                        {selectedGuideline.organization} • {selectedGuideline.year} • {selectedGuideline.specialty}
                      </CardDescription>
                    </div>
                    {selectedGuideline.applicable ? (
                      <Badge className="bg-green-600">
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Applicable
                      </Badge>
                    ) : (
                      <Badge variant="outline">Not Applicable</Badge>
                    )}
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {selectedGuideline.recommendations.map((rec: any, index: number) => (
                      <div key={index} className="p-4 border rounded-lg">
                        <div className="flex items-start justify-between mb-3">
                          <div className="flex-1">
                            <Badge variant="outline" className="mb-2">{rec.condition}</Badge>
                            <p className="font-semibold">{rec.recommendation}</p>
                          </div>
                          <div className="flex gap-2 flex-shrink-0 ml-4">
                            <Badge className={
                              rec.class === 'I' ? 'bg-green-600' :
                              rec.class === 'II' ? 'bg-yellow-600' :
                              'bg-gray-600'
                            }>
                              Class {rec.class}
                            </Badge>
                            <Badge variant="outline">LoE {rec.loe}</Badge>
                          </div>
                        </div>

                        <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg mb-3">
                          <p className="text-xs font-semibold mb-1">Rationale:</p>
                          <p className="text-xs">{rec.rationale}</p>
                        </div>

                        {rec.contraindications.length > 0 && (
                          <div className="p-3 bg-red-50 dark:bg-red-950 rounded-lg">
                            <div className="flex items-start gap-2">
                              <AlertCircle className="h-4 w-4 text-red-600 flex-shrink-0 mt-0.5" />
                              <div>
                                <p className="text-xs font-semibold text-red-900 dark:text-red-100 mb-1">
                                  Contraindications:
                                </p>
                                <ul className="text-xs text-red-800 dark:text-red-200 space-y-1">
                                  {rec.contraindications.map((contra: string, idx: number) => (
                                    <li key={idx}>• {contra}</li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          </div>
                        )}

                        <div className="flex gap-2 mt-3">
                          <Button size="sm" variant="outline" className="flex-1">
                            <Info className="h-3 w-3 mr-1" />
                            View Evidence
                          </Button>
                          <Button size="sm" className="flex-1">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Add to Plan
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              {/* Evidence Grading Legend */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Evidence Grading System</CardTitle>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="class">
                    <TabsList className="w-full">
                      <TabsTrigger value="class" className="flex-1">Recommendation Class</TabsTrigger>
                      <TabsTrigger value="loe" className="flex-1">Level of Evidence</TabsTrigger>
                    </TabsList>

                    <TabsContent value="class" className="space-y-2 mt-4">
                      <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                        <p className="font-semibold text-sm">Class I</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          Benefit >>> Risk. Should be performed/administered.
                        </p>
                      </div>
                      <div className="p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg">
                        <p className="font-semibold text-sm">Class IIa</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          Benefit >> Risk. Reasonable to perform/administer.
                        </p>
                      </div>
                      <div className="p-3 bg-orange-50 dark:bg-orange-950 rounded-lg">
                        <p className="font-semibold text-sm">Class IIb</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          Benefit ≥ Risk. May be considered.
                        </p>
                      </div>
                      <div className="p-3 bg-red-50 dark:bg-red-950 rounded-lg">
                        <p className="font-semibold text-sm">Class III</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          No benefit or harm. Should not be performed/administered.
                        </p>
                      </div>
                    </TabsContent>

                    <TabsContent value="loe" className="space-y-2 mt-4">
                      <div className="p-3 border rounded-lg">
                        <p className="font-semibold text-sm">Level A</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          High-quality evidence from multiple RCTs or meta-analyses
                        </p>
                      </div>
                      <div className="p-3 border rounded-lg">
                        <p className="font-semibold text-sm">Level B</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          Moderate-quality evidence from single RCT or non-randomized studies
                        </p>
                      </div>
                      <div className="p-3 border rounded-lg">
                        <p className="font-semibold text-sm">Level C</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          Limited evidence from consensus opinion, case studies, or standard of care
                        </p>
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </>
          ) : (
            <Card>
              <CardContent className="py-12">
                <div className="text-center text-gray-500">
                  <BookOpen className="h-12 w-12 mx-auto mb-4 opacity-50" />
                  <p>Select a guideline to view recommendations</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
