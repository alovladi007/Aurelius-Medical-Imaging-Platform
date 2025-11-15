"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Stethoscope,
  Brain,
  Target,
  AlertCircle,
  CheckCircle,
  TrendingDown,
  TrendingUp,
  FileQuestion,
  Lightbulb,
  MessageSquare,
  BarChart3,
  Activity
} from 'lucide-react';

export default function DiagnosticCopilotPage() {
  const [selectedCase, setSelectedCase] = useState<any>(null);

  const stats = {
    patientsToday: 567,
    avgDxTime: '4.1s',
    accuracy: 96.2,
    uncertaintyAbstentions: 34,
    avgDifferentialSize: 3.2
  };

  const activeCases = [
    {
      id: 'P-2024-156',
      age: 58,
      gender: 'F',
      chiefComplaint: 'Progressive dyspnea × 3 weeks',
      initialDifferential: ['CHF', 'COPD exacerbation', 'Pneumonia', 'PE', 'ILD'],
      currentDifferential: ['CHF', 'PE'],
      probability: [72, 18],
      uncertaintyScore: 0.28,
      dataSufficiency: 0.75,
      missingData: ['BNP', 'CTPA', 'Echo'],
      nextSteps: ['Order BNP', 'Consider CTPA if D-dimer elevated', 'Echo for EF'],
      reasoning: 'CHF most likely given orthopnea, PND, lower extremity edema. PE remains in differential due to acute component. Need BNP to differentiate.',
      keyFindings: [
        'JVD present',
        'Bilateral crackles',
        'Lower extremity edema',
        'Recent long flight (PE risk)'
      ],
      status: 'in_progress'
    },
    {
      id: 'P-2024-157',
      age: 42,
      gender: 'M',
      chiefComplaint: 'Fever, headache, photophobia × 2 days',
      initialDifferential: ['Meningitis', 'Migraine', 'Sinusitis', 'Viral syndrome'],
      currentDifferential: ['Bacterial meningitis'],
      probability: [85],
      uncertaintyScore: 0.15,
      dataSufficiency: 0.90,
      missingData: [],
      nextSteps: ['Stat LP', 'Blood cultures', 'Empiric antibiotics NOW'],
      reasoning: 'Classic meningitis triad (fever, headache, nuchal rigidity) + photophobia. High suspicion warrants immediate empiric treatment before LP.',
      keyFindings: [
        'Nuchal rigidity',
        'Kernig sign positive',
        'Temp 102.8°F',
        'WBC 18,000'
      ],
      status: 'alert'
    },
    {
      id: 'P-2024-158',
      age: 71,
      gender: 'M',
      chiefComplaint: 'Confusion, falls × 1 week',
      initialDifferential: ['Delirium (infectious)', 'Dementia', 'Stroke', 'Medication effect', 'Metabolic'],
      currentDifferential: ['UTI-induced delirium', 'Medication toxicity', 'Subdural hematoma'],
      probability: [45, 30, 15],
      uncertaintyScore: 0.55,
      dataSufficiency: 0.40,
      missingData: ['UA/UCx', 'Medication levels', 'Head CT', 'TSH', 'B12'],
      nextSteps: ['UA/UCx', 'Medication reconciliation', 'Consider head CT given falls', 'Metabolic panel'],
      reasoning: 'Broad differential given nonspecific presentation. Need more data to narrow. UTI common cause in elderly. Must rule out subdural given falls.',
      keyFindings: [
        'AMS × 1 week',
        'Multiple falls',
        'On warfarin',
        'Cloudy urine noted'
      ],
      status: 'needs_data'
    }
  ];

  const diagnosticFrameworks = [
    {
      name: 'VINDICATE',
      categories: ['Vascular', 'Infectious', 'Neoplastic', 'Degenerative', 'Intoxication', 'Congenital', 'Autoimmune', 'Trauma', 'Endocrine'],
      useCase: 'Comprehensive systematic approach'
    },
    {
      name: 'Bayesian Reasoning',
      steps: ['Prior probability', 'Likelihood ratio', 'Posterior probability', 'Test threshold'],
      useCase: 'Quantitative diagnostic reasoning'
    },
    {
      name: 'Dual Process',
      modes: ['System 1: Pattern recognition', 'System 2: Analytical reasoning'],
      useCase: 'Metacognitive error reduction'
    }
  ];

  const performanceMetrics = [
    { metric: 'Diagnostic Accuracy', value: '96.2%', change: '+1.1%', period: 'vs last month', color: 'text-green-600' },
    { metric: 'Differential Completeness', value: '94.8%', change: '+0.5%', period: 'includes correct dx', color: 'text-green-600' },
    { metric: 'Data Utilization', value: '91.3%', change: '-0.3%', period: 'relevant data used', color: 'text-yellow-600' },
    { metric: 'Time to Diagnosis', value: '4.1s', change: '-0.7s', period: 'avg processing time', color: 'text-green-600' }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Stethoscope className="h-8 w-8 text-blue-600" />
            Diagnostic Copilot
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Synthesizes problems, narrows differential via targeted questions and orders
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <BarChart3 className="h-4 w-4 mr-2" />
            Analytics
          </Button>
          <Button>
            <Brain className="h-4 w-4 mr-2" />
            Reasoning Log
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
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Dx Time</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-blue-600">{stats.avgDxTime}</p>
            <p className="text-xs text-green-600">-0.7s improvement</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-green-600">{stats.accuracy}%</p>
            <p className="text-xs text-gray-600">vs expert internists</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Abstentions</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-orange-600">{stats.uncertaintyAbstentions}</p>
            <p className="text-xs text-gray-600">High uncertainty</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Diff Size</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-purple-600">{stats.avgDifferentialSize}</p>
            <p className="text-xs text-gray-600">diagnoses</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Active Cases */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Active Diagnostic Cases</CardTitle>
              <CardDescription>Ongoing diagnostic reasoning with uncertainty quantification</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {activeCases.map((case_) => (
                  <Card
                    key={case_.id}
                    className={`cursor-pointer transition-all ${
                      case_.status === 'alert' ? 'border-red-500 border-2' :
                      case_.status === 'needs_data' ? 'border-yellow-500 border-2' :
                      'hover:shadow-lg'
                    }`}
                    onClick={() => setSelectedCase(case_)}
                  >
                    <CardContent className="pt-6">
                      <div className="flex items-start justify-between mb-4">
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-2">
                            <Badge variant="outline">{case_.id}</Badge>
                            <Badge variant="secondary">{case_.age}y {case_.gender}</Badge>
                            {case_.status === 'alert' && (
                              <Badge className="bg-red-600">
                                <AlertCircle className="h-3 w-3 mr-1" />
                                URGENT
                              </Badge>
                            )}
                            {case_.status === 'needs_data' && (
                              <Badge className="bg-yellow-600">
                                <FileQuestion className="h-3 w-3 mr-1" />
                                NEEDS DATA
                              </Badge>
                            )}
                          </div>
                          <p className="font-semibold">{case_.chiefComplaint}</p>
                        </div>
                        <div className="text-right ml-4">
                          <div className="w-16 h-16 rounded-full border-4 border-blue-500 flex items-center justify-center">
                            <p className="text-xs font-bold text-blue-600">
                              {Math.round((1 - case_.uncertaintyScore) * 100)}%
                            </p>
                          </div>
                          <p className="text-xs text-gray-600 mt-1">Confidence</p>
                        </div>
                      </div>

                      {/* Current Differential */}
                      <div className="mb-4 p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
                        <p className="text-xs font-semibold mb-2 flex items-center gap-1">
                          <Target className="h-3 w-3" />
                          Current Differential (narrowed from {case_.initialDifferential.length}):
                        </p>
                        <div className="space-y-2">
                          {case_.currentDifferential.map((dx, idx) => (
                            <div key={idx} className="flex items-center gap-2">
                              <div className="flex-1">
                                <div className="flex items-center justify-between mb-1">
                                  <span className="font-semibold text-sm">{dx}</span>
                                  <span className="text-sm font-bold text-blue-600">
                                    {case_.probability[idx]}%
                                  </span>
                                </div>
                                <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                                  <div
                                    className="bg-blue-600 h-2 rounded-full"
                                    style={{ width: `${case_.probability[idx]}%` }}
                                  />
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Key Findings */}
                      <div className="mb-4">
                        <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-2">
                          <Lightbulb className="h-3 w-3 inline mr-1" />
                          Key Findings:
                        </p>
                        <div className="flex flex-wrap gap-1">
                          {case_.keyFindings.map((finding, i) => (
                            <Badge key={i} variant="outline" className="text-xs">
                              {finding}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      {/* Reasoning */}
                      <div className="mb-4 p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                        <p className="text-xs font-semibold mb-1 flex items-center gap-1">
                          <Brain className="h-3 w-3" />
                          Reasoning:
                        </p>
                        <p className="text-xs">{case_.reasoning}</p>
                      </div>

                      {/* Missing Data */}
                      {case_.missingData.length > 0 && (
                        <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg border border-yellow-200 dark:border-yellow-900">
                          <p className="text-xs font-semibold text-yellow-900 dark:text-yellow-100 mb-2">
                            <AlertCircle className="h-3 w-3 inline mr-1" />
                            Missing Data (Data Sufficiency: {Math.round(case_.dataSufficiency * 100)}%):
                          </p>
                          <div className="flex flex-wrap gap-1">
                            {case_.missingData.map((data, i) => (
                              <Badge key={i} variant="outline" className="text-xs bg-yellow-100 dark:bg-yellow-950">
                                {data}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Next Steps */}
                      <div>
                        <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-2">
                          <MessageSquare className="h-3 w-3 inline mr-1" />
                          Recommended Next Steps:
                        </p>
                        <ul className="list-disc ml-5 space-y-1">
                          {case_.nextSteps.map((step, i) => (
                            <li key={i} className="text-xs">{step}</li>
                          ))}
                        </ul>
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
              <CardDescription>Diagnostic accuracy and efficiency</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {performanceMetrics.map((metric, idx) => (
                  <div key={idx} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <p className="font-semibold text-sm">{metric.metric}</p>
                      <p className={`text-2xl font-bold ${metric.color}`}>{metric.value}</p>
                    </div>
                    <div className="flex items-center gap-1 text-xs">
                      {metric.change.startsWith('+') ? (
                        <TrendingUp className="h-3 w-3 text-green-600" />
                      ) : metric.change.startsWith('-') && metric.metric !== 'Time to Diagnosis' ? (
                        <TrendingDown className="h-3 w-3 text-red-600" />
                      ) : (
                        <TrendingDown className="h-3 w-3 text-green-600" />
                      )}
                      <span>{metric.change} {metric.period}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Diagnostic Frameworks */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Diagnostic Frameworks</CardTitle>
              <CardDescription>Evidence-based reasoning approaches</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {diagnosticFrameworks.map((framework, idx) => (
                  <div key={idx} className="p-4 border rounded-lg">
                    <p className="font-semibold mb-2">{framework.name}</p>
                    {'categories' in framework && (
                      <div className="space-y-1 text-xs mb-2">
                        {framework.categories.map((cat, i) => (
                          <Badge key={i} variant="secondary" className="text-xs mr-1">
                            {cat}
                          </Badge>
                        ))}
                      </div>
                    )}
                    {'steps' in framework && (
                      <ul className="list-disc ml-5 space-y-1 text-xs mb-2">
                        {framework.steps.map((step, i) => (
                          <li key={i}>{step}</li>
                        ))}
                      </ul>
                    )}
                    {'modes' in framework && (
                      <ul className="list-disc ml-5 space-y-1 text-xs mb-2">
                        {framework.modes.map((mode, i) => (
                          <li key={i}>{mode}</li>
                        ))}
                      </ul>
                    )}
                    <p className="text-xs text-gray-600 dark:text-gray-400 italic">{framework.useCase}</p>
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
                    <p className="font-semibold">Problem Synthesis</p>
                    <p className="text-gray-600 dark:text-gray-400">Combines multi-source data into unified problem list</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold">Differential Narrowing</p>
                    <p className="text-gray-600 dark:text-gray-400">Bayesian updating with each new data point</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold">Targeted Questioning</p>
                    <p className="text-gray-600 dark:text-gray-400">Asks specific questions to disambiguate diagnoses</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold">Test Selection</p>
                    <p className="text-gray-600 dark:text-gray-400">Recommends highest yield tests based on LR+/LR-</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold">Uncertainty Gating</p>
                    <p className="text-gray-600 dark:text-gray-400">Abstains when confidence <70%, requests more data</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold">Explainable Reasoning</p>
                    <p className="text-gray-600 dark:text-gray-400">Shows step-by-step diagnostic logic with citations</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="border-yellow-200 dark:border-yellow-900">
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <AlertCircle className="h-4 w-4 text-yellow-600" />
                Safety Features
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2 text-xs">
                <div className="p-2 bg-yellow-50 dark:bg-yellow-950 rounded">
                  <p className="font-semibold">Life-Threatening First</p>
                  <p className="text-gray-600 dark:text-gray-400">Always considers dangerous diagnoses (PE, MI, meningitis, etc.)</p>
                </div>
                <div className="p-2 bg-red-50 dark:bg-red-950 rounded">
                  <p className="font-semibold">Cannot-Miss Alerts</p>
                  <p className="text-gray-600 dark:text-gray-400">Flags conditions requiring immediate action</p>
                </div>
                <div className="p-2 bg-blue-50 dark:bg-blue-950 rounded">
                  <p className="font-semibold">Human Override</p>
                  <p className="text-gray-600 dark:text-gray-400">All recommendations subject to clinician review</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
