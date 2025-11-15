"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Calculator,
  Heart,
  Droplets,
  Pill,
  AlertTriangle,
  CheckCircle,
  Info,
  BookOpen,
  TrendingUp,
  Activity
} from 'lucide-react';

export default function CalculatorsPage() {
  const [selectedCalculator, setSelectedCalculator] = useState('chadsvasc');
  const [calculatorInputs, setCalculatorInputs] = useState<Record<string, any>>({});
  const [result, setResult] = useState<any>(null);

  const calculators = [
    {
      id: 'chadsvasc',
      name: 'CHA₂DS₂-VASc Score',
      category: 'Cardiology',
      description: 'Stroke risk in atrial fibrillation',
      icon: Heart,
      color: 'text-red-600',
      inputs: [
        { id: 'chf', label: 'Congestive Heart Failure', type: 'checkbox', points: 1 },
        { id: 'htn', label: 'Hypertension', type: 'checkbox', points: 1 },
        { id: 'age75', label: 'Age ≥75', type: 'checkbox', points: 2 },
        { id: 'diabetes', label: 'Diabetes', type: 'checkbox', points: 1 },
        { id: 'stroke', label: 'Prior Stroke/TIA/TE', type: 'checkbox', points: 2 },
        { id: 'vascular', label: 'Vascular Disease', type: 'checkbox', points: 1 },
        { id: 'age65', label: 'Age 65-74', type: 'checkbox', points: 1 },
        { id: 'female', label: 'Female', type: 'checkbox', points: 1 }
      ],
      interpretation: [
        { score: 0, risk: 'Low', recommendation: 'No antithrombotic therapy', annualStroke: 0 },
        { score: 1, risk: 'Low-Moderate', recommendation: 'Consider anticoagulation', annualStroke: 1.3 },
        { score: 2, risk: 'Moderate', recommendation: 'Recommend anticoagulation', annualStroke: 2.2 },
        { score: '≥3', risk: 'High', recommendation: 'Strong recommendation for anticoagulation', annualStroke: 3.2 }
      ],
      reference: 'Lip GY et al. Chest. 2010;137(2):263-272.'
    },
    {
      id: 'wells-dvt',
      name: 'Wells Score (DVT)',
      category: 'Vascular',
      description: 'Deep vein thrombosis probability',
      icon: Activity,
      color: 'text-blue-600',
      inputs: [
        { id: 'cancer', label: 'Active cancer', type: 'checkbox', points: 1 },
        { id: 'paralysis', label: 'Paralysis/paresis/immobilization', type: 'checkbox', points: 1 },
        { id: 'bedridden', label: 'Bedridden >3 days or surgery <12 weeks', type: 'checkbox', points: 1 },
        { id: 'tenderness', label: 'Localized tenderness', type: 'checkbox', points: 1 },
        { id: 'swelling', label: 'Entire leg swollen', type: 'checkbox', points: 1 },
        { id: 'calf', label: 'Calf swelling >3cm', type: 'checkbox', points: 1 },
        { id: 'pitting', label: 'Pitting edema', type: 'checkbox', points: 1 },
        { id: 'veins', label: 'Collateral superficial veins', type: 'checkbox', points: 1 },
        { id: 'alternative', label: 'Alternative diagnosis more likely', type: 'checkbox', points: -2 }
      ],
      interpretation: [
        { score: '≤0', risk: 'Low', recommendation: 'DVT unlikely. Consider D-dimer.', probability: 5 },
        { score: '1-2', risk: 'Moderate', recommendation: 'Moderate risk. D-dimer or ultrasound.', probability: 17 },
        { score: '≥3', risk: 'High', recommendation: 'DVT likely. Ultrasound recommended.', probability: 53 }
      ],
      reference: 'Wells PS et al. Lancet. 1997;350(9094):1795-1798.'
    },
    {
      id: 'heart',
      name: 'HEART Score',
      category: 'Emergency Medicine',
      description: 'Major adverse cardiac events risk',
      icon: Heart,
      color: 'text-pink-600',
      inputs: [
        { id: 'history', label: 'History', type: 'select', options: ['Slightly suspicious (0)', 'Moderately suspicious (1)', 'Highly suspicious (2)'], points: [0, 1, 2] },
        { id: 'ecg', label: 'ECG', type: 'select', options: ['Normal (0)', 'Non-specific changes (1)', 'Significant ST changes (2)'], points: [0, 1, 2] },
        { id: 'age', label: 'Age', type: 'select', options: ['<45 (0)', '45-64 (1)', '≥65 (2)'], points: [0, 1, 2] },
        { id: 'risk', label: 'Risk Factors', type: 'select', options: ['No risk factors (0)', '1-2 risk factors (1)', '≥3 or history CAD (2)'], points: [0, 1, 2] },
        { id: 'troponin', label: 'Troponin', type: 'select', options: ['Normal (0)', '1-3× normal (1)', '>3× normal (2)'], points: [0, 1, 2] }
      ],
      interpretation: [
        { score: '0-3', risk: 'Low', recommendation: 'Discharge. MACE risk 0.9-1.7%', probability: 1.7 },
        { score: '4-6', risk: 'Moderate', recommendation: 'Admit for observation. MACE risk 12-17%', probability: 17 },
        { score: '7-10', risk: 'High', recommendation: 'Urgent intervention. MACE risk 50-65%', probability: 65 }
      ],
      reference: 'Six AJ et al. Neth Heart J. 2008;16(6):191-196.'
    },
    {
      id: 'meld',
      name: 'MELD Score',
      category: 'Hepatology',
      description: 'Liver disease severity',
      icon: Droplets,
      color: 'text-yellow-600',
      inputs: [
        { id: 'creatinine', label: 'Serum Creatinine (mg/dL)', type: 'number', min: 0.5, max: 10 },
        { id: 'bilirubin', label: 'Total Bilirubin (mg/dL)', type: 'number', min: 0.5, max: 50 },
        { id: 'inr', label: 'INR', type: 'number', min: 1.0, max: 5.0 },
        { id: 'dialysis', label: 'Dialysis ≥2 times in last week', type: 'checkbox' }
      ],
      interpretation: [
        { score: '<10', risk: 'Low', recommendation: '3-month mortality ~1.9%', mortality: 1.9 },
        { score: '10-19', risk: 'Moderate', recommendation: '3-month mortality ~6%', mortality: 6.0 },
        { score: '20-29', risk: 'High', recommendation: '3-month mortality ~20%', mortality: 19.6 },
        { score: '30-39', risk: 'Very High', recommendation: '3-month mortality ~50%', mortality: 52.6 },
        { score: '≥40', risk: 'Critical', recommendation: '3-month mortality ~70%', mortality: 71.3 }
      ],
      reference: 'Kamath PS et al. Hepatology. 2001;33(2):464-470.'
    }
  ];

  const dosingTools = [
    {
      id: 'creatinine-clearance',
      name: 'Creatinine Clearance (Cockcroft-Gault)',
      description: 'Renal function for drug dosing',
      inputs: ['age', 'weight', 'serum_cr', 'sex']
    },
    {
      id: 'child-pugh',
      name: 'Child-Pugh Score',
      description: 'Hepatic function for drug dosing',
      inputs: ['bilirubin', 'albumin', 'inr', 'ascites', 'encephalopathy']
    },
    {
      id: 'bsa',
      name: 'Body Surface Area (BSA)',
      description: 'Chemotherapy dosing',
      inputs: ['height', 'weight']
    }
  ];

  const calculate = () => {
    const calc = calculators.find(c => c.id === selectedCalculator);
    if (!calc) return;

    let score = 0;
    calc.inputs.forEach(input => {
      if (input.type === 'checkbox' && calculatorInputs[input.id]) {
        score += input.points;
      } else if (input.type === 'select' && calculatorInputs[input.id] !== undefined) {
        score += input.points[calculatorInputs[input.id]];
      }
    });

    // For MELD score (special calculation)
    if (selectedCalculator === 'meld') {
      const cr = Math.max(1, calculatorInputs.creatinine || 1);
      const bili = Math.max(1, calculatorInputs.bilirubin || 1);
      const inr = Math.max(1, calculatorInputs.inr || 1);
      score = Math.round(3.78 * Math.log(bili) + 11.2 * Math.log(inr) + 9.57 * Math.log(cr) + 6.43);
      score = Math.max(6, Math.min(40, score)); // Clamp between 6-40
    }

    const interpretation = calc.interpretation.find(interp => {
      if (typeof interp.score === 'string') {
        if (interp.score.includes('≥')) {
          const threshold = parseInt(interp.score.replace('≥', ''));
          return score >= threshold;
        } else if (interp.score.includes('≤')) {
          const threshold = parseInt(interp.score.replace('≤', ''));
          return score <= threshold;
        } else if (interp.score.includes('<')) {
          const threshold = parseInt(interp.score.replace('<', ''));
          return score < threshold;
        } else if (interp.score.includes('-')) {
          const [min, max] = interp.score.split('-').map(s => parseInt(s));
          return score >= min && score <= max;
        }
      }
      return interp.score === score;
    }) || calc.interpretation[calc.interpretation.length - 1];

    setResult({
      score,
      interpretation,
      calculator: calc
    });
  };

  const currentCalc = calculators.find(c => c.id === selectedCalculator);

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Calculator className="h-8 w-8 text-primary" />
            Clinical Calculators
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Risk scores, dosing, renal/hepatic adjustments with citations
          </p>
        </div>
        <Button variant="outline">
          <BookOpen className="h-4 w-4 mr-2" />
          View All (45)
          </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Calculator Selection */}
        <div className="lg:col-span-1">
          <Card>
            <CardHeader>
              <CardTitle>Select Calculator</CardTitle>
              <CardDescription>Choose a clinical calculator or dosing tool</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="risk">
                <TabsList className="w-full">
                  <TabsTrigger value="risk" className="flex-1">Risk Scores</TabsTrigger>
                  <TabsTrigger value="dosing" className="flex-1">Dosing</TabsTrigger>
                </TabsList>

                <TabsContent value="risk" className="space-y-2 mt-4">
                  {calculators.map((calc) => (
                    <div
                      key={calc.id}
                      className={`p-3 border-2 rounded-lg cursor-pointer transition-all ${
                        selectedCalculator === calc.id
                          ? 'border-primary bg-primary/5'
                          : 'border-gray-200 dark:border-gray-800 hover:border-primary/50'
                      }`}
                      onClick={() => {
                        setSelectedCalculator(calc.id);
                        setCalculatorInputs({});
                        setResult(null);
                      }}
                    >
                      <div className="flex items-center gap-3">
                        <calc.icon className={`h-5 w-5 ${calc.color}`} />
                        <div className="flex-1">
                          <p className="font-semibold text-sm">{calc.name}</p>
                          <p className="text-xs text-gray-600 dark:text-gray-400">{calc.category}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </TabsContent>

                <TabsContent value="dosing" className="space-y-2 mt-4">
                  {dosingTools.map((tool) => (
                    <div
                      key={tool.id}
                      className="p-3 border rounded-lg hover:border-primary cursor-pointer transition-all"
                    >
                      <p className="font-semibold text-sm">{tool.name}</p>
                      <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">{tool.description}</p>
                    </div>
                  ))}
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>
        </div>

        {/* Calculator Input */}
        <div className="lg:col-span-2 space-y-6">
          {currentCalc && (
            <>
              <Card>
                <CardHeader>
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <currentCalc.icon className={`h-6 w-6 ${currentCalc.color}`} />
                        {currentCalc.name}
                      </CardTitle>
                      <CardDescription className="mt-2">
                        {currentCalc.description}
                      </CardDescription>
                    </div>
                    <Badge variant="outline">{currentCalc.category}</Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4">
                  {currentCalc.inputs.map((input) => (
                    <div key={input.id} className="space-y-2">
                      {input.type === 'checkbox' && (
                        <div className="flex items-center justify-between p-3 border rounded-lg">
                          <div className="flex items-center gap-3">
                            <input
                              type="checkbox"
                              id={input.id}
                              checked={calculatorInputs[input.id] || false}
                              onChange={(e) => setCalculatorInputs({ ...calculatorInputs, [input.id]: e.target.checked })}
                              className="w-4 h-4"
                            />
                            <Label htmlFor={input.id} className="text-sm">
                              {input.label}
                            </Label>
                          </div>
                          <Badge variant="secondary">{input.points} pt{input.points !== 1 ? 's' : ''}</Badge>
                        </div>
                      )}
                      {input.type === 'number' && (
                        <div>
                          <Label htmlFor={input.id}>{input.label}</Label>
                          <Input
                            id={input.id}
                            type="number"
                            min={input.min}
                            max={input.max}
                            step="0.1"
                            value={calculatorInputs[input.id] || ''}
                            onChange={(e) => setCalculatorInputs({ ...calculatorInputs, [input.id]: parseFloat(e.target.value) })}
                          />
                        </div>
                      )}
                      {input.type === 'select' && (
                        <div>
                          <Label htmlFor={input.id}>{input.label}</Label>
                          <select
                            id={input.id}
                            className="w-full p-2 border rounded"
                            value={calculatorInputs[input.id] || ''}
                            onChange={(e) => setCalculatorInputs({ ...calculatorInputs, [input.id]: parseInt(e.target.value) })}
                          >
                            <option value="">Select...</option>
                            {input.options && input.options.map((option, idx) => (
                              <option key={idx} value={idx}>{option}</option>
                            ))}
                          </select>
                        </div>
                      )}
                    </div>
                  ))}

                  <Button onClick={calculate} className="w-full">
                    <Calculator className="h-4 w-4 mr-2" />
                    Calculate Score
                  </Button>
                </CardContent>
              </Card>

              {/* Result */}
              {result && (
                <Card className="border-primary">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      {result.interpretation.risk.includes('Low') && <CheckCircle className="h-5 w-5 text-green-600" />}
                      {result.interpretation.risk.includes('Moderate') && <AlertTriangle className="h-5 w-5 text-yellow-600" />}
                      {result.interpretation.risk.includes('High') && <AlertTriangle className="h-5 w-5 text-red-600" />}
                      Result: Score {result.score}
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <p className="font-semibold">Risk Level</p>
                        <Badge
                          className={
                            result.interpretation.risk.includes('Low') ? 'bg-green-600' :
                            result.interpretation.risk.includes('Moderate') ? 'bg-yellow-600' :
                            'bg-red-600'
                          }
                        >
                          {result.interpretation.risk}
                        </Badge>
                      </div>
                      <p className="text-sm font-bold mt-3">{result.interpretation.recommendation}</p>
                      {result.interpretation.annualStroke !== undefined && (
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                          Annual stroke risk: {result.interpretation.annualStroke}%
                        </p>
                      )}
                      {result.interpretation.probability !== undefined && (
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                          Probability: {result.interpretation.probability}%
                        </p>
                      )}
                      {result.interpretation.mortality !== undefined && (
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                          3-month mortality: ~{result.interpretation.mortality}%
                        </p>
                      )}
                    </div>

                    <div className="p-3 border rounded-lg flex items-start gap-2">
                      <Info className="h-4 w-4 text-blue-600 flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="text-xs font-semibold">Reference</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                          {currentCalc.reference}
                        </p>
                      </div>
                    </div>

                    <div className="flex gap-2">
                      <Button className="flex-1">
                        Add to Chart
                      </Button>
                      <Button variant="outline" className="flex-1">
                        Print Report
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Interpretation Guide */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-lg">Interpretation Guide</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {currentCalc.interpretation.map((interp, idx) => (
                      <div key={idx} className="p-3 border rounded-lg">
                        <div className="flex items-center justify-between mb-1">
                          <p className="font-semibold text-sm">Score: {interp.score}</p>
                          <Badge variant="outline">{interp.risk}</Badge>
                        </div>
                        <p className="text-xs text-gray-600 dark:text-gray-400">{interp.recommendation}</p>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
