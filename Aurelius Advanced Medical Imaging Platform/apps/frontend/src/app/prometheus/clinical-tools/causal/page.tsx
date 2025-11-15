"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  GitBranch,
  TrendingUp,
  TrendingDown,
  AlertCircle,
  CheckCircle,
  ArrowRight,
  Zap,
  Brain,
  BarChart3,
  LineChart,
  Shuffle,
  Target,
  Scale,
  Activity
} from 'lucide-react';

export default function CausalWhatIfPage() {
  const [selectedScenario, setSelectedScenario] = useState<any>(null);

  const stats = {
    scenarios: 156,
    confounders: 23,
    avgCertainty: '87%',
    psmMatches: '94%',
    ate: '+12.3%'
  };

  const patientContext = {
    id: 'P-2024-0198',
    age: 72,
    diagnosis: 'Type 2 Diabetes, uncontrolled',
    currentTreatment: 'Metformin 1000mg BID',
    a1c: 8.9,
    egfr: 62,
    weight: 95,
    comorbidities: ['Hypertension', 'Hyperlipidemia', 'CKD Stage 3a']
  };

  const whatIfScenarios = [
    {
      id: 'SCENARIO-001',
      intervention: 'Add GLP-1 Agonist (Semaglutide 0.5mg weekly)',
      baseline: 'Metformin 1000mg BID alone',
      outcome: 'HbA1c reduction at 6 months',
      causalEffect: {
        ate: -1.4,
        ci95: [-1.7, -1.1],
        nnt: 3.2,
        certainty: 'High (RCT evidence)'
      },
      predictedOutcome: {
        current: 8.9,
        predicted: 7.5,
        probability: 0.78,
        timeframe: '6 months'
      },
      confounders: [
        { name: 'Age', controlled: true, method: 'Stratification' },
        { name: 'Baseline A1c', controlled: true, method: 'Regression adjustment' },
        { name: 'BMI', controlled: true, method: 'Propensity score' },
        { name: 'Medication adherence', controlled: false, method: 'Unmeasured' }
      ],
      evidenceQuality: 'High',
      evidenceSource: 'SUSTAIN-6 trial + institutional data (n=1,247)',
      benefits: [
        'A1c reduction: -1.4% (95% CI: -1.7 to -1.1)',
        'Weight loss: -4.2 kg (95% CI: -5.1 to -3.3)',
        'CV risk reduction: HR 0.74 (0.58-0.95) for MACE',
        'Renal protection: Slower eGFR decline'
      ],
      risks: [
        'Nausea: 15-20% (usually transient)',
        'Pancreatitis: <1% (causal link uncertain)',
        'Injection site reactions: 5-8%',
        'Cost: $900-1200/month'
      ],
      heterogeneity: {
        subgroups: [
          { group: 'Age >70', effect: -1.2, ci: [-1.6, -0.8] },
          { group: 'Baseline A1c >9', effect: -1.8, ci: [-2.3, -1.3] },
          { group: 'eGFR 45-60', effect: -1.3, ci: [-1.7, -0.9] }
        ],
        mostBenefit: 'Baseline A1c >9 (patient qualifies)'
      }
    },
    {
      id: 'SCENARIO-002',
      intervention: 'Add SGLT2 Inhibitor (Empagliflozin 10mg daily)',
      baseline: 'Metformin 1000mg BID alone',
      outcome: 'Composite: A1c reduction + renal protection',
      causalEffect: {
        ate: -0.9,
        ci95: [-1.2, -0.6],
        nnt: 4.8,
        certainty: 'High (RCT evidence)'
      },
      predictedOutcome: {
        current: 8.9,
        predicted: 8.0,
        probability: 0.71,
        timeframe: '6 months'
      },
      confounders: [
        { name: 'eGFR', controlled: true, method: 'Stratification (eGFR >20 required)' },
        { name: 'CHF status', controlled: true, method: 'Propensity score' },
        { name: 'Diuretic use', controlled: true, method: 'Regression adjustment' }
      ],
      evidenceQuality: 'High',
      evidenceSource: 'EMPA-REG OUTCOME + institutional data (n=892)',
      benefits: [
        'A1c reduction: -0.9% (95% CI: -1.2 to -0.6)',
        'eGFR preservation: Slower decline by 2.1 mL/min/1.73m²/year',
        'CV mortality: HR 0.62 (0.49-0.77)',
        'Heart failure hospitalization: HR 0.65 (0.50-0.85)',
        'Weight loss: -2.8 kg'
      ],
      risks: [
        'Genital mycotic infections: 8-12%',
        'Volume depletion: 2-4% (esp. with diuretics)',
        'DKA (euglycemic): <0.1% (rare but serious)',
        'UTI: slightly increased'
      ],
      heterogeneity: {
        subgroups: [
          { group: 'eGFR 45-60', effect: -0.8, ci: [-1.2, -0.4] },
          { group: 'With CHF', effect: -1.1, ci: [-1.6, -0.6] },
          { group: 'Age >70', effect: -0.7, ci: [-1.1, -0.3] }
        ],
        mostBenefit: 'Patients with CHF or CKD (patient has CKD)'
      }
    },
    {
      id: 'SCENARIO-003',
      intervention: 'Intensify Insulin (Basal insulin glargine 10 units qHS)',
      baseline: 'Metformin 1000mg BID alone',
      outcome: 'HbA1c reduction at 6 months',
      causalEffect: {
        ate: -1.6,
        ci95: [-2.0, -1.2],
        nnt: 2.8,
        certainty: 'Moderate (observational + limited RCT)'
      },
      predictedOutcome: {
        current: 8.9,
        predicted: 7.3,
        probability: 0.82,
        timeframe: '6 months'
      },
      confounders: [
        { name: 'Disease duration', controlled: true, method: 'Regression adjustment' },
        { name: 'C-peptide level', controlled: false, method: 'Unmeasured' },
        { name: 'Beta-cell reserve', controlled: false, method: 'Unmeasured' }
      ],
      evidenceQuality: 'Moderate',
      evidenceSource: 'Institutional data (n=2,341) + meta-analysis',
      benefits: [
        'A1c reduction: -1.6% (95% CI: -2.0 to -1.2)',
        'Highly effective for glycemic control',
        'Low cost ($25-40/month)',
        'Well-established therapy'
      ],
      risks: [
        'Hypoglycemia: 15-25% (major risk)',
        'Weight gain: +3.2 kg on average',
        'Injection burden (daily)',
        'Requires glucose monitoring and titration'
      ],
      heterogeneity: {
        subgroups: [
          { group: 'Baseline A1c >9', effect: -2.1, ci: [-2.6, -1.6] },
          { group: 'Age >70', effect: -1.3, ci: [-1.8, -0.8] },
          { group: 'Hypoglycemia history', effect: -1.4, ci: [-2.0, -0.8], note: 'Higher risk' }
        ],
        mostBenefit: 'Baseline A1c >9 (patient qualifies), but hypoglycemia concern in elderly'
      }
    }
  ];

  const causalGraph = {
    nodes: [
      { id: 'intervention', label: 'GLP-1 Agonist', type: 'treatment' },
      { id: 'outcome', label: 'HbA1c Reduction', type: 'outcome' },
      { id: 'age', label: 'Age', type: 'confounder' },
      { id: 'baseline_a1c', label: 'Baseline A1c', type: 'confounder' },
      { id: 'bmi', label: 'BMI', type: 'confounder' },
      { id: 'adherence', label: 'Adherence', type: 'mediator' },
      { id: 'weight_loss', label: 'Weight Loss', type: 'mediator' }
    ],
    edges: [
      { from: 'age', to: 'intervention', type: 'confounding' },
      { from: 'age', to: 'outcome', type: 'confounding' },
      { from: 'baseline_a1c', to: 'intervention', type: 'confounding' },
      { from: 'baseline_a1c', to: 'outcome', type: 'confounding' },
      { from: 'bmi', to: 'intervention', type: 'confounding' },
      { from: 'bmi', to: 'outcome', type: 'confounding' },
      { from: 'intervention', to: 'adherence', type: 'causal' },
      { from: 'intervention', to: 'weight_loss', type: 'causal' },
      { from: 'adherence', to: 'outcome', type: 'causal' },
      { from: 'weight_loss', to: 'outcome', type: 'causal' },
      { from: 'intervention', to: 'outcome', type: 'causal-direct' }
    ]
  };

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <GitBranch className="h-8 w-8 text-primary" />
            Causal What-If Analyzer
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Counterfactual reasoning and treatment effect estimation
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <BarChart3 className="h-4 w-4 mr-2" />
            View All Estimates
          </Button>
          <Button>
            <Brain className="h-4 w-4 mr-2" />
            New Scenario
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Scenarios Analyzed</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-primary">{stats.scenarios}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Confounders</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-blue-600">{stats.confounders}</p>
            <p className="text-xs text-gray-600">Adjusted for</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Certainty</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-green-600">{stats.avgCertainty}</p>
            <p className="text-xs text-gray-600">Causal estimates</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">PSM Match Rate</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-purple-600">{stats.psmMatches}</p>
            <p className="text-xs text-gray-600">Propensity score</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Treatment Effect</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-orange-600">{stats.ate}</p>
            <p className="text-xs text-gray-600">Across scenarios</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Patient Context */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Patient Context</CardTitle>
              <CardDescription>Baseline for causal inference</CardDescription>
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
                <p className="text-sm font-semibold mb-2">Current Treatment</p>
                <Badge variant="secondary">{patientContext.currentTreatment}</Badge>
                <p className="text-xs text-gray-600 mt-2">HbA1c: <span className="font-semibold text-red-600">{patientContext.a1c}%</span> (target: &lt;7%)</p>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">Comorbidities</p>
                <div className="flex flex-wrap gap-1">
                  {patientContext.comorbidities.map((condition, i) => (
                    <Badge key={i} variant="outline" className="text-xs">
                      {condition}
                    </Badge>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Causal Graph */}
          <Card className="border-purple-200 dark:border-purple-900">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <GitBranch className="h-5 w-5 text-purple-600" />
                Causal DAG
              </CardTitle>
              <CardDescription>Directed Acyclic Graph</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                <div className="space-y-3 text-xs">
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-blue-600"></div>
                    <span>Treatment (Intervention)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-green-600"></div>
                    <span>Outcome (HbA1c)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-yellow-600"></div>
                    <span>Confounders (Age, BMI, Baseline A1c)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full bg-purple-600"></div>
                    <span>Mediators (Weight loss, Adherence)</span>
                  </div>
                </div>
                <p className="text-xs text-gray-600 mt-3 italic">
                  Simplified representation. Full DAG available in analytics view.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* What-If Scenarios */}
        <div className="lg:col-span-2 space-y-4">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold">Treatment Scenarios</h2>
            <Badge variant="outline">{whatIfScenarios.length} scenarios compared</Badge>
          </div>

          {whatIfScenarios.map((scenario) => (
            <Card
              key={scenario.id}
              className={`cursor-pointer transition-all ${
                selectedScenario?.id === scenario.id ? 'border-primary border-2' : 'hover:shadow-lg'
              }`}
              onClick={() => setSelectedScenario(selectedScenario?.id === scenario.id ? null : scenario)}
            >
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge variant="outline">{scenario.id}</Badge>
                      <Badge
                        className={
                          scenario.evidenceQuality === 'High' ? 'bg-green-600' :
                          scenario.evidenceQuality === 'Moderate' ? 'bg-yellow-600' :
                          'bg-orange-600'
                        }
                      >
                        {scenario.evidenceQuality} Evidence
                      </Badge>
                      <Badge variant="secondary" className="text-xs">
                        {scenario.causalEffect.certainty}
                      </Badge>
                    </div>
                    <CardTitle className="text-lg mb-2">{scenario.intervention}</CardTitle>
                    <CardDescription>
                      <span className="font-semibold">Outcome:</span> {scenario.outcome}
                    </CardDescription>
                  </div>
                </div>
              </CardHeader>

              <CardContent>
                {/* Predicted Effect */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="p-4 bg-red-50 dark:bg-red-950 rounded-lg border border-red-200 dark:border-red-900">
                    <p className="text-xs text-gray-600 mb-1">Current HbA1c</p>
                    <p className="text-3xl font-bold text-red-600">{scenario.predictedOutcome.current}%</p>
                  </div>
                  <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg border border-green-200 dark:border-green-900">
                    <p className="text-xs text-gray-600 mb-1">Predicted HbA1c</p>
                    <p className="text-3xl font-bold text-green-600">{scenario.predictedOutcome.predicted}%</p>
                    <p className="text-xs text-gray-600 mt-1">
                      {scenario.predictedOutcome.timeframe} • {Math.round(scenario.predictedOutcome.probability * 100)}% confidence
                    </p>
                  </div>
                </div>

                {/* Causal Effect */}
                <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-900 mb-4">
                  <div className="flex items-center gap-2 mb-2">
                    <Target className="h-4 w-4 text-blue-600" />
                    <p className="font-semibold text-sm">Causal Treatment Effect</p>
                  </div>
                  <div className="grid grid-cols-3 gap-4 text-sm">
                    <div>
                      <p className="text-xs text-gray-600">ATE (Average)</p>
                      <p className="font-bold text-blue-600">{scenario.causalEffect.ate}%</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-600">95% CI</p>
                      <p className="font-bold text-sm">[{scenario.causalEffect.ci95[0]}, {scenario.causalEffect.ci95[1]}]</p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-600">NNT</p>
                      <p className="font-bold text-sm">{scenario.causalEffect.nnt}</p>
                    </div>
                  </div>
                  <p className="text-xs text-gray-600 mt-2">
                    <span className="font-semibold">Source:</span> {scenario.evidenceSource}
                  </p>
                </div>

                {selectedScenario?.id === scenario.id && (
                  <div className="space-y-4">
                    <Tabs defaultValue="confounders" className="w-full">
                      <TabsList className="grid w-full grid-cols-4">
                        <TabsTrigger value="confounders">Confounders</TabsTrigger>
                        <TabsTrigger value="benefits">Benefits</TabsTrigger>
                        <TabsTrigger value="risks">Risks</TabsTrigger>
                        <TabsTrigger value="heterogeneity">HTE</TabsTrigger>
                      </TabsList>

                      <TabsContent value="confounders" className="space-y-2 mt-4">
                        <p className="text-sm font-semibold mb-2">Confounding Control:</p>
                        {scenario.confounders.map((confounder, idx) => (
                          <div key={idx} className="p-3 border rounded-lg">
                            <div className="flex items-center justify-between mb-1">
                              <span className="font-semibold text-sm">{confounder.name}</span>
                              {confounder.controlled ? (
                                <Badge variant="outline" className="bg-green-50 text-green-800 dark:bg-green-950 text-xs">
                                  <CheckCircle className="h-3 w-3 mr-1" />
                                  Controlled
                                </Badge>
                              ) : (
                                <Badge variant="outline" className="bg-red-50 text-red-800 dark:bg-red-950 text-xs">
                                  <AlertCircle className="h-3 w-3 mr-1" />
                                  Unmeasured
                                </Badge>
                              )}
                            </div>
                            <p className="text-xs text-gray-600">Method: {confounder.method}</p>
                          </div>
                        ))}
                      </TabsContent>

                      <TabsContent value="benefits" className="space-y-2 mt-4">
                        <p className="text-sm font-semibold mb-2">Expected Benefits:</p>
                        <ul className="space-y-2">
                          {scenario.benefits.map((benefit, idx) => (
                            <li key={idx} className="text-sm flex items-start gap-2 p-2 bg-green-50 dark:bg-green-950 rounded">
                              <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0 mt-0.5" />
                              <span>{benefit}</span>
                            </li>
                          ))}
                        </ul>
                      </TabsContent>

                      <TabsContent value="risks" className="space-y-2 mt-4">
                        <p className="text-sm font-semibold mb-2">Potential Risks:</p>
                        <ul className="space-y-2">
                          {scenario.risks.map((risk, idx) => (
                            <li key={idx} className="text-sm flex items-start gap-2 p-2 bg-red-50 dark:bg-red-950 rounded">
                              <AlertCircle className="h-4 w-4 text-red-600 flex-shrink-0 mt-0.5" />
                              <span>{risk}</span>
                            </li>
                          ))}
                        </ul>
                      </TabsContent>

                      <TabsContent value="heterogeneity" className="space-y-3 mt-4">
                        <p className="text-sm font-semibold mb-2">Heterogeneous Treatment Effects:</p>
                        {scenario.heterogeneity.subgroups.map((subgroup, idx) => (
                          <div key={idx} className="p-3 border rounded-lg">
                            <div className="flex items-center justify-between mb-2">
                              <span className="font-semibold text-sm">{subgroup.group}</span>
                              <Badge variant="secondary" className="text-xs">
                                Effect: {subgroup.effect}%
                              </Badge>
                            </div>
                            <div className="flex items-center gap-2 text-xs">
                              <span className="text-gray-600">95% CI:</span>
                              <span className="font-semibold">[{subgroup.ci[0]}, {subgroup.ci[1]}]</span>
                            </div>
                            {subgroup.note && (
                              <p className="text-xs text-orange-600 mt-1">⚠ {subgroup.note}</p>
                            )}
                          </div>
                        ))}
                        <div className="p-3 bg-purple-50 dark:bg-purple-950 rounded-lg border border-purple-200 dark:border-purple-900 mt-3">
                          <p className="text-xs font-semibold mb-1">
                            <Brain className="h-3 w-3 inline mr-1" />
                            Patient-Specific Prediction:
                          </p>
                          <p className="text-xs">{scenario.heterogeneity.mostBenefit}</p>
                        </div>
                      </TabsContent>
                    </Tabs>

                    <div className="flex gap-2 mt-4">
                      <Button className="flex-1">
                        <Shuffle className="h-4 w-4 mr-2" />
                        Apply This Intervention
                      </Button>
                      <Button variant="outline" className="flex-1">
                        <LineChart className="h-4 w-4 mr-2" />
                        View Full Analysis
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
