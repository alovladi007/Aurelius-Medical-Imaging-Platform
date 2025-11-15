"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  FileText,
  Search,
  CheckCircle,
  AlertCircle,
  Plus,
  Trash2,
  Edit,
  Shield,
  Brain,
  Clock,
  TrendingUp,
  Pill,
  Activity,
  Zap,
  AlertTriangle
} from 'lucide-react';

export default function OrderSetsPage() {
  const [selectedOrderSet, setSelectedOrderSet] = useState<any>(null);
  const [customizing, setCustomizing] = useState(false);

  const stats = {
    orderSets: 47,
    avgOrders: 12,
    safetyChecks: '100%',
    avgTime: '45s',
    adoptionRate: '89%'
  };

  const patientContext = {
    id: 'P-2024-0156',
    name: 'Sarah Johnson',
    age: 68,
    weight: 72,
    height: 165,
    creatinine: 1.2,
    egfr: 54,
    allergies: ['Penicillin', 'Sulfa'],
    activeOrders: [
      'Metformin 1000mg PO BID',
      'Lisinopril 10mg PO daily',
      'Atorvastatin 40mg PO qHS'
    ],
    diagnosis: 'Community-acquired pneumonia',
    severityScore: 'CURB-65: 2 (moderate risk)'
  };

  const orderSets = [
    {
      id: 'OS-001',
      name: 'Community-Acquired Pneumonia (CAP)',
      category: 'Infectious Disease',
      severity: 'Moderate (CURB-65: 2)',
      indication: 'Hospitalized CAP, no ICU criteria',
      evidenceBased: 'IDSA/ATS 2019 Guidelines',
      orders: [
        {
          type: 'Antibiotic',
          order: 'Ceftriaxone 1g IV q24h',
          justification: 'Empiric coverage for S. pneumoniae, H. influenzae. Renally dosed for CrCl 54.',
          priority: 'STAT',
          safety: { alerts: [], interactions: [], contraindications: [] }
        },
        {
          type: 'Antibiotic',
          order: 'Azithromycin 500mg PO/IV daily',
          justification: 'Atypical coverage (Mycoplasma, Legionella, Chlamydophila). Alternative to doxycycline given sulfa allergy.',
          priority: 'STAT',
          safety: {
            alerts: ['QT prolongation - check baseline EKG'],
            interactions: ['Monitor with atorvastatin (rhabdomyolysis risk)'],
            contraindications: []
          }
        },
        {
          type: 'Labs',
          order: 'CBC with diff, CMP, lactate now',
          justification: 'Baseline labs for sepsis monitoring, renal function, electrolytes',
          priority: 'STAT',
          safety: { alerts: [], interactions: [], contraindications: [] }
        },
        {
          type: 'Labs',
          order: 'Blood cultures x2 (before antibiotics)',
          justification: 'Identify causative organism for antibiotic tailoring',
          priority: 'STAT',
          safety: { alerts: [], interactions: [], contraindications: [] }
        },
        {
          type: 'Imaging',
          order: 'Chest X-ray PA and lateral',
          justification: 'Confirm infiltrate, assess severity, rule out complications',
          priority: 'STAT',
          safety: { alerts: [], interactions: [], contraindications: [] }
        },
        {
          type: 'Fluids',
          order: 'NS 1L IV bolus over 1h',
          justification: 'Volume resuscitation if hypotensive or tachycardic',
          priority: 'Routine',
          safety: {
            alerts: ['Monitor for fluid overload (CHF risk)'],
            interactions: [],
            contraindications: []
          }
        },
        {
          type: 'O2',
          order: 'Oxygen therapy to maintain SpO2 >92%',
          justification: 'Target oxygen saturation per guidelines',
          priority: 'STAT',
          safety: { alerts: [], interactions: [], contraindications: [] }
        },
        {
          type: 'VTE Prophylaxis',
          order: 'Enoxaparin 40mg SC daily',
          justification: 'VTE prophylaxis per CHEST guidelines, dose adjusted for CrCl 54',
          priority: 'Routine',
          safety: {
            alerts: ['Monitor platelets (HIT risk)', 'Adjust for CrCl <30'],
            interactions: [],
            contraindications: []
          }
        },
        {
          type: 'Monitoring',
          order: 'Vital signs q4h, I/O',
          justification: 'Close monitoring for clinical deterioration',
          priority: 'Routine',
          safety: { alerts: [], interactions: [], contraindications: [] }
        }
      ],
      duration: '5-7 days',
      reviewCriteria: [
        'Clinical stability x24h (afebrile, hemodynamically stable)',
        'Oral tolerance achieved',
        'WBC trending down',
        'Consider procalcitonin-guided duration'
      ],
      alternatives: {
        allergyAdjustments: [
          'Beta-lactam allergy: Switch ceftriaxone → Levofloxacin 750mg daily',
          'Macrolide intolerance: Switch azithromycin → Doxycycline 100mg BID'
        ],
        severityAdjustments: [
          'Severe CAP (ICU): Add vancomycin for MRSA coverage',
          'Pseudomonas risk: Switch to piperacillin-tazobactam + azithromycin'
        ]
      }
    },
    {
      id: 'OS-002',
      name: 'Sepsis/Septic Shock',
      category: 'Critical Care',
      severity: 'Severe',
      indication: 'Sepsis-3 criteria met, lactate >2',
      evidenceBased: 'Surviving Sepsis Campaign 2021',
      orders: [
        {
          type: 'Fluids',
          order: 'Lactated Ringers 30mL/kg IV (2160mL) over 3 hours',
          justification: 'Initial resuscitation per SSC guidelines, calculated for 72kg',
          priority: 'STAT',
          safety: {
            alerts: ['Reassess after bolus - avoid fluid overload'],
            interactions: [],
            contraindications: []
          }
        },
        {
          type: 'Antibiotics',
          order: 'Vancomycin 1500mg IV loading dose, then 1g q12h',
          justification: 'Empiric MRSA coverage, dose adjusted for weight and renal function',
          priority: 'STAT (within 1 hour)',
          safety: {
            alerts: ['Check vancomycin trough before 4th dose'],
            interactions: [],
            contraindications: []
          }
        },
        {
          type: 'Antibiotics',
          order: 'Cefepime 2g IV q8h',
          justification: 'Broad gram-negative coverage including Pseudomonas',
          priority: 'STAT (within 1 hour)',
          safety: {
            alerts: ['Renally dosed - adjust for CrCl changes'],
            interactions: [],
            contraindications: []
          }
        },
        {
          type: 'Labs',
          order: 'Lactate, CBC, CMP, coags, blood cultures x2, UA/UCx',
          justification: 'Identify source, assess organ dysfunction, guide resuscitation',
          priority: 'STAT',
          safety: { alerts: [], interactions: [], contraindications: [] }
        },
        {
          type: 'Vasopressor',
          order: 'Norepinephrine drip - target MAP ≥65',
          justification: 'First-line vasopressor if hypotensive after fluid bolus',
          priority: 'STAT',
          safety: {
            alerts: ['Requires central line', 'Titrate to MAP, avoid excessive doses'],
            interactions: [],
            contraindications: []
          }
        },
        {
          type: 'Source Control',
          order: 'CT chest/abdomen/pelvis with IV contrast',
          justification: 'Identify infection source for source control',
          priority: 'Urgent',
          safety: {
            alerts: ['Check CrCl for contrast (CrCl 54 - acceptable with hydration)'],
            interactions: ['Hold metformin x48h post-contrast'],
            contraindications: []
          }
        },
        {
          type: 'Monitoring',
          order: 'ICU admission, continuous vitals, central line, arterial line',
          justification: 'Close monitoring and access for vasopressors',
          priority: 'STAT',
          safety: { alerts: [], interactions: [], contraindications: [] }
        }
      ],
      duration: '7-14 days (source-dependent)',
      reviewCriteria: [
        'Repeat lactate q2-4h until <2',
        'Reassess volume status after initial bolus',
        'De-escalate antibiotics based on cultures and clinical response',
        'Daily sedation vacation and SBT assessment if intubated'
      ]
    },
    {
      id: 'OS-003',
      name: 'Acute Decompensated Heart Failure',
      category: 'Cardiology',
      severity: 'Moderate',
      indication: 'Volume overload, orthopnea, BNP elevated',
      evidenceBased: 'ACC/AHA 2022 Guidelines',
      orders: [
        {
          type: 'Diuretic',
          order: 'Furosemide 40mg IV x1, then 20mg IV BID',
          justification: 'Volume removal for pulmonary edema, adjusted for home dose equivalent',
          priority: 'STAT',
          safety: {
            alerts: ['Monitor K, Mg, Cr daily', 'Strict I/O'],
            interactions: [],
            contraindications: []
          }
        },
        {
          type: 'Labs',
          order: 'BNP, troponin, CBC, CMP, Mg',
          justification: 'Assess severity, rule out ACS, baseline electrolytes',
          priority: 'STAT',
          safety: { alerts: [], interactions: [], contraindications: [] }
        },
        {
          type: 'Imaging',
          order: 'Chest X-ray portable',
          justification: 'Assess pulmonary edema severity',
          priority: 'STAT',
          safety: { alerts: [], interactions: [], contraindications: [] }
        },
        {
          type: 'Cardiac',
          order: 'Echocardiogram (if none in past 3 months)',
          justification: 'Assess EF, valves, wall motion abnormalities',
          priority: 'Routine',
          safety: { alerts: [], interactions: [], contraindications: [] }
        },
        {
          type: 'Monitoring',
          order: 'Telemetry monitoring, daily weights, strict I/O',
          justification: 'Monitor arrhythmias, volume status',
          priority: 'STAT',
          safety: { alerts: [], interactions: [], contraindications: [] }
        },
        {
          type: 'Diet',
          order: '2g sodium restriction, 1.5L fluid restriction',
          justification: 'Volume management',
          priority: 'Routine',
          safety: { alerts: [], interactions: [], contraindications: [] }
        }
      ],
      duration: '3-5 days',
      reviewCriteria: [
        'Net negative 1-2L daily',
        'Resolution of orthopnea/dyspnea',
        'Optimize guideline-directed medical therapy before discharge',
        'Transition to PO diuretics when stable'
      ]
    }
  ];

  const safetyChecks = {
    allergies: {
      status: 'Warning',
      alerts: [
        'Penicillin allergy: Avoid ceftriaxone if anaphylaxis history (consider aztreonam)',
        'Sulfa allergy: Avoid Bactrim, furosemide (may be tolerated)'
      ]
    },
    interactions: {
      status: 'Caution',
      alerts: [
        'Azithromycin + Atorvastatin: Increased rhabdomyolysis risk',
        'Metformin + IV contrast: Hold x48h post-contrast (CrCl <60)'
      ]
    },
    renal: {
      status: 'Adjusted',
      alerts: [
        'CrCl 54: Dose adjustments applied to enoxaparin, antibiotics',
        'Avoid NSAIDs, nephrotoxic agents when possible'
      ]
    },
    contraindications: {
      status: 'None',
      alerts: []
    }
  };

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <FileText className="h-8 w-8 text-primary" />
            Order Sets
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Context-aware draft orders with justifications and safety checks
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Search className="h-4 w-4 mr-2" />
            Search Order Sets
          </Button>
          <Button>
            <Plus className="h-4 w-4 mr-2" />
            Custom Order Set
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Available Sets</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-primary">{stats.orderSets}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Orders/Set</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-blue-600">{stats.avgOrders}</p>
            <p className="text-xs text-gray-600">Auto-customized</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Safety Checks</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-green-600">{stats.safetyChecks}</p>
            <p className="text-xs text-gray-600">Coverage rate</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Time Saved</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-purple-600">{stats.avgTime}</p>
            <p className="text-xs text-gray-600">Per order set</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Adoption Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-orange-600">{stats.adoptionRate}</p>
            <p className="text-xs text-gray-600">Clinician usage</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Patient Context */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Patient Context</CardTitle>
              <CardDescription>Orders auto-adjusted for patient</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <p className="text-sm font-semibold mb-2">Demographics</p>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-600">ID:</span>
                    <span className="font-semibold">{patientContext.id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Age:</span>
                    <span className="font-semibold">{patientContext.age}y</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Weight:</span>
                    <span className="font-semibold">{patientContext.weight}kg</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">eGFR:</span>
                    <span className="font-semibold text-yellow-600">{patientContext.egfr}</span>
                  </div>
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">Allergies</p>
                <div className="flex flex-wrap gap-1">
                  {patientContext.allergies.map((allergy, i) => (
                    <Badge key={i} variant="destructive" className="text-xs">
                      {allergy}
                    </Badge>
                  ))}
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">Active Medications</p>
                <div className="space-y-1">
                  {patientContext.activeOrders.map((order, i) => (
                    <div key={i} className="text-xs p-2 bg-gray-50 dark:bg-gray-900 rounded">
                      {order}
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">Current Diagnosis</p>
                <Badge className="mb-2">{patientContext.diagnosis}</Badge>
                <p className="text-xs text-gray-600">{patientContext.severityScore}</p>
              </div>
            </CardContent>
          </Card>

          {/* Safety Checks */}
          <Card className="border-orange-200 dark:border-orange-900">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Shield className="h-5 w-5 text-orange-600" />
                Safety Checks
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {Object.entries(safetyChecks).map(([key, value]) => (
                <div key={key}>
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-sm font-semibold capitalize">{key}</p>
                    <Badge
                      variant="outline"
                      className={
                        value.status === 'Warning' ? 'bg-red-50 text-red-800 dark:bg-red-950' :
                        value.status === 'Caution' ? 'bg-yellow-50 text-yellow-800 dark:bg-yellow-950' :
                        value.status === 'Adjusted' ? 'bg-blue-50 text-blue-800 dark:bg-blue-950' :
                        'bg-green-50 text-green-800 dark:bg-green-950'
                      }
                    >
                      {value.status}
                    </Badge>
                  </div>
                  {value.alerts.length > 0 && (
                    <div className="space-y-1">
                      {value.alerts.map((alert, idx) => (
                        <div key={idx} className="text-xs p-2 bg-orange-50 dark:bg-orange-950 rounded border border-orange-200 dark:border-orange-900">
                          <AlertTriangle className="h-3 w-3 inline mr-1 text-orange-600" />
                          {alert}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </CardContent>
          </Card>
        </div>

        {/* Order Sets List */}
        <div className="lg:col-span-2 space-y-4">
          {orderSets.map((orderSet) => (
            <Card
              key={orderSet.id}
              className={`cursor-pointer transition-all ${
                selectedOrderSet?.id === orderSet.id ? 'border-primary border-2' : 'hover:shadow-lg'
              }`}
              onClick={() => setSelectedOrderSet(orderSet)}
            >
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge variant="outline">{orderSet.id}</Badge>
                      <Badge>{orderSet.category}</Badge>
                      <Badge
                        className={
                          orderSet.severity === 'Severe' ? 'bg-red-600' :
                          orderSet.severity.includes('Moderate') ? 'bg-yellow-600' :
                          'bg-green-600'
                        }
                      >
                        {orderSet.severity}
                      </Badge>
                    </div>
                    <CardTitle className="text-lg">{orderSet.name}</CardTitle>
                    <CardDescription className="mt-2">
                      <span className="font-semibold">Indication:</span> {orderSet.indication}
                    </CardDescription>
                    <div className="mt-2 flex items-center gap-2">
                      <Badge variant="secondary" className="text-xs">
                        <Brain className="h-3 w-3 mr-1" />
                        {orderSet.evidenceBased}
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {orderSet.orders.length} orders
                      </Badge>
                    </div>
                  </div>
                </div>
              </CardHeader>

              {selectedOrderSet?.id === orderSet.id && (
                <CardContent>
                  <Tabs defaultValue="orders" className="w-full">
                    <TabsList className="grid w-full grid-cols-3">
                      <TabsTrigger value="orders">Orders</TabsTrigger>
                      <TabsTrigger value="review">Review Criteria</TabsTrigger>
                      <TabsTrigger value="alternatives">Alternatives</TabsTrigger>
                    </TabsList>

                    <TabsContent value="orders" className="space-y-3 mt-4">
                      {orderSet.orders.map((order, idx) => (
                        <div key={idx} className="p-4 border rounded-lg">
                          <div className="flex items-start justify-between mb-2">
                            <div className="flex items-center gap-2">
                              <Badge variant="outline" className="text-xs">{order.type}</Badge>
                              <Badge
                                className={
                                  order.priority === 'STAT' ? 'bg-red-600' :
                                  order.priority === 'Urgent' ? 'bg-orange-600' :
                                  'bg-blue-600'
                                }
                              >
                                {order.priority}
                              </Badge>
                            </div>
                            <div className="flex gap-2">
                              <Button size="sm" variant="ghost">
                                <Edit className="h-3 w-3" />
                              </Button>
                              <Button size="sm" variant="ghost">
                                <Trash2 className="h-3 w-3" />
                              </Button>
                            </div>
                          </div>
                          <p className="font-semibold mb-2">{order.order}</p>
                          <div className="p-2 bg-blue-50 dark:bg-blue-950 rounded mb-2">
                            <p className="text-xs font-semibold mb-1">
                              <Brain className="h-3 w-3 inline mr-1" />
                              Justification:
                            </p>
                            <p className="text-xs">{order.justification}</p>
                          </div>

                          {/* Safety Alerts */}
                          {(order.safety.alerts.length > 0 ||
                            order.safety.interactions.length > 0 ||
                            order.safety.contraindications.length > 0) && (
                            <div className="space-y-1">
                              {order.safety.alerts.map((alert, i) => (
                                <div key={i} className="text-xs p-2 bg-yellow-50 dark:bg-yellow-950 rounded border border-yellow-200 dark:border-yellow-900">
                                  <AlertCircle className="h-3 w-3 inline mr-1 text-yellow-600" />
                                  {alert}
                                </div>
                              ))}
                              {order.safety.interactions.map((interaction, i) => (
                                <div key={i} className="text-xs p-2 bg-orange-50 dark:bg-orange-950 rounded border border-orange-200 dark:border-orange-900">
                                  <Zap className="h-3 w-3 inline mr-1 text-orange-600" />
                                  Drug Interaction: {interaction}
                                </div>
                              ))}
                              {order.safety.contraindications.map((contra, i) => (
                                <div key={i} className="text-xs p-2 bg-red-50 dark:bg-red-950 rounded border border-red-200 dark:border-red-900">
                                  <AlertTriangle className="h-3 w-3 inline mr-1 text-red-600" />
                                  Contraindication: {contra}
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ))}

                      <div className="flex gap-2 mt-4">
                        <Button className="flex-1">
                          <CheckCircle className="h-4 w-4 mr-2" />
                          Sign All Orders
                        </Button>
                        <Button variant="outline" className="flex-1">
                          <Edit className="h-4 w-4 mr-2" />
                          Customize Set
                        </Button>
                      </div>
                    </TabsContent>

                    <TabsContent value="review" className="space-y-3 mt-4">
                      <div className="p-4 border rounded-lg">
                        <p className="font-semibold mb-3 flex items-center gap-2">
                          <Clock className="h-4 w-4 text-blue-600" />
                          Duration: {orderSet.duration}
                        </p>
                        <p className="text-sm font-semibold mb-2">Review Criteria:</p>
                        <ul className="space-y-2">
                          {orderSet.reviewCriteria.map((criterion, idx) => (
                            <li key={idx} className="text-sm flex items-start gap-2">
                              <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                              <span>{criterion}</span>
                            </li>
                          ))}
                        </ul>
                      </div>
                    </TabsContent>

                    <TabsContent value="alternatives" className="space-y-3 mt-4">
                      {orderSet.alternatives && (
                        <>
                          {orderSet.alternatives.allergyAdjustments && (
                            <div className="p-4 border rounded-lg">
                              <p className="font-semibold mb-3 flex items-center gap-2">
                                <AlertTriangle className="h-4 w-4 text-red-600" />
                                Allergy Adjustments
                              </p>
                              <ul className="space-y-2">
                                {orderSet.alternatives.allergyAdjustments.map((adj, idx) => (
                                  <li key={idx} className="text-sm p-2 bg-red-50 dark:bg-red-950 rounded">
                                    {adj}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          {orderSet.alternatives.severityAdjustments && (
                            <div className="p-4 border rounded-lg">
                              <p className="font-semibold mb-3 flex items-center gap-2">
                                <TrendingUp className="h-4 w-4 text-orange-600" />
                                Severity Adjustments
                              </p>
                              <ul className="space-y-2">
                                {orderSet.alternatives.severityAdjustments.map((adj, idx) => (
                                  <li key={idx} className="text-sm p-2 bg-orange-50 dark:bg-orange-950 rounded">
                                    {adj}
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </>
                      )}
                    </TabsContent>
                  </Tabs>
                </CardContent>
              )}
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
