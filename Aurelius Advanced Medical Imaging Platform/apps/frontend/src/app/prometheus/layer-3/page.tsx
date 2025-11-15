"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Brain,
  Image,
  Activity,
  Dna,
  FileText,
  Zap,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Cpu,
  Eye,
  Heart
} from 'lucide-react';

export default function Layer3Page() {
  const [models, setModels] = useState([
    { name: 'Clinical LLM', modality: 'Text', status: 'running', requests: 2483, latency: 145, accuracy: 94.2 },
    { name: 'DICOM Vision Encoder', modality: 'Vision', status: 'running', requests: 892, latency: 230, accuracy: 96.5 },
    { name: 'ICU Waveform Transformer', modality: 'Time-Series', status: 'running', requests: 4521, latency: 45, accuracy: 91.8 },
    { name: 'Variant Effect Predictor', modality: 'Genomics', status: 'running', requests: 156, latency: 380, accuracy: 89.3 },
    { name: 'Multimodal Fusion', modality: 'Fusion', status: 'running', requests: 724, latency: 520, accuracy: 95.7 }
  ]);

  const modalities = [
    {
      name: 'Text & Code',
      icon: FileText,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50 dark:bg-blue-950',
      capabilities: ['Long-context clinical LLM', 'Tool-use (calculators, order sets)', 'Code interpreter (CQL authoring)'],
      models: ['GPT-4 Medical', 'Clinical BERT', 'MedPaLM 2']
    },
    {
      name: 'Vision',
      icon: Eye,
      color: 'text-green-600',
      bgColor: 'bg-green-50 dark:bg-green-950',
      capabilities: ['DICOM-native (CT/MR/X-ray/US)', 'Pathology WSI encoders', 'Uncertainty quantification'],
      models: ['Med-ViT', 'RadImageNet', 'PathCLIP']
    },
    {
      name: 'Time-Series',
      icon: Activity,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50 dark:bg-purple-950',
      capabilities: ['ICU waveforms (ECG/SpO₂/ABP)', 'Conformal risk sets', 'Alarm prediction'],
      models: ['Temporal Fusion Transformer', 'TimesNet', 'WaveNet']
    },
    {
      name: 'Genomics',
      icon: Dna,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50 dark:bg-orange-950',
      capabilities: ['Variant effect transformers', 'Gene-set enrichment', 'PGx rules'],
      models: ['AlphaMissense', 'ESM-2', 'Enformer']
    }
  ];

  const calibrationMethods = [
    { name: 'Conformal Prediction', description: 'Validity-guaranteed prediction sets', status: 'active' },
    { name: 'Selective Abstention', description: 'Refuse low-confidence predictions', status: 'active' },
    { name: 'Evidential Deep Learning', description: 'Dirichlet-based uncertainty', status: 'active' },
    { name: 'MC-Dropout', description: 'Monte Carlo uncertainty estimation', status: 'active' },
    { name: 'Deep Ensembles', description: 'Multiple model predictions', status: 'active' }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Brain className="h-8 w-8 text-primary" />
            Layer 3: Foundation Model Stack
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Multimodal AI with calibrated uncertainty
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">Model Registry</Button>
          <Button><Zap className="h-4 w-4 mr-2" />Deploy Model</Button>
        </div>
      </div>

      {/* Active Models */}
      <Card>
        <CardHeader>
          <CardTitle>Active Foundation Models</CardTitle>
          <CardDescription>Production inference endpoints</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {models.map((model, index) => (
              <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-primary/10 rounded-full flex items-center justify-center">
                    <Brain className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="font-semibold">{model.name}</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{model.modality}</p>
                  </div>
                </div>
                <div className="flex items-center gap-6">
                  <div className="text-center">
                    <p className="text-lg font-bold">{model.requests}</p>
                    <p className="text-xs text-gray-600">Requests</p>
                  </div>
                  <div className="text-center">
                    <p className="text-lg font-bold">{model.latency}ms</p>
                    <p className="text-xs text-gray-600">P95 Latency</p>
                  </div>
                  <div className="text-center">
                    <p className="text-lg font-bold text-primary">{model.accuracy}%</p>
                    <p className="text-xs text-gray-600">Accuracy</p>
                  </div>
                  <Badge variant="default" className="bg-green-600">
                    <CheckCircle className="h-3 w-3 mr-1" />
                    {model.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Modalities */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Supported Modalities</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {modalities.map((modality, index) => (
            <Card key={index}>
              <CardHeader>
                <div className="flex items-center gap-3 mb-3">
                  <div className={`w-12 h-12 rounded-lg ${modality.bgColor} flex items-center justify-center`}>
                    <modality.icon className={`h-6 w-6 ${modality.color}`} />
                  </div>
                  <CardTitle>{modality.name}</CardTitle>
                </div>
              </CardHeader>
              <CardContent className="space-y-3">
                <div>
                  <p className="text-sm font-semibold mb-2">Capabilities:</p>
                  <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
                    {modality.capabilities.map((cap, idx) => (
                      <li key={idx}>• {cap}</li>
                    ))}
                  </ul>
                </div>
                <div>
                  <p className="text-sm font-semibold mb-2">Models:</p>
                  <div className="flex flex-wrap gap-2">
                    {modality.models.map((model, idx) => (
                      <Badge key={idx} variant="outline">{model}</Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Multimodal Fusion */}
      <Card className="border-primary">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5 text-primary" />
            Multimodal Fusion Architecture
          </CardTitle>
          <CardDescription>Cross-attention over text, images, time-series, and graph hops</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
              <h4 className="font-semibold mb-2">Per-Modality Encoders</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Each modality (text, vision, time-series, genomics) has specialized encoders producing embeddings
              </p>
            </div>
            <div className="text-center py-2">
              <TrendingUp className="h-6 w-6 mx-auto text-gray-400" />
            </div>
            <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg">
              <h4 className="font-semibold mb-2">Fusion Transformer</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Multi-head cross-attention learns interactions between modalities (e.g., imaging findings ↔ lab results)
              </p>
            </div>
            <div className="text-center py-2">
              <TrendingUp className="h-6 w-6 mx-auto text-gray-400" />
            </div>
            <div className="p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
              <h4 className="font-semibold mb-2">Task-Specific Heads</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Classification, regression, sequence generation with calibrated uncertainty estimates
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Calibration & Uncertainty */}
      <Card>
        <CardHeader>
          <CardTitle>Calibrated Uncertainty Quantification</CardTitle>
          <CardDescription>Trustworthy AI with valid prediction sets</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {calibrationMethods.map((method, index) => (
              <div key={index} className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <p className="font-semibold text-sm">{method.name}</p>
                  <Badge variant="default">{method.status}</Badge>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400">{method.description}</p>
              </div>
            ))}
            <div className="p-4 border rounded-lg col-span-full bg-orange-50 dark:bg-orange-950">
              <div className="flex items-start gap-2">
                <AlertCircle className="h-5 w-5 text-orange-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-sm mb-1">Selective Abstention Policy</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    Models abstain from predictions when uncertainty exceeds configured thresholds, reducing false positives and maintaining clinical safety
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Model Performance */}
      <Card>
        <CardHeader>
          <CardTitle>Model Performance Metrics</CardTitle>
          <CardDescription>Aggregated across all modalities</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="accuracy">
            <TabsList>
              <TabsTrigger value="accuracy">Accuracy</TabsTrigger>
              <TabsTrigger value="calibration">Calibration</TabsTrigger>
              <TabsTrigger value="latency">Latency</TabsTrigger>
            </TabsList>

            <TabsContent value="accuracy" className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                  <p className="text-3xl font-bold text-blue-600">94.2%</p>
                  <p className="text-sm text-gray-600">Overall Accuracy</p>
                </div>
                <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                  <p className="text-3xl font-bold text-green-600">96.8%</p>
                  <p className="text-sm text-gray-600">Precision</p>
                </div>
                <div className="text-center p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                  <p className="text-3xl font-bold text-purple-600">93.5%</p>
                  <p className="text-sm text-gray-600">Recall</p>
                </div>
                <div className="text-center p-4 bg-orange-50 dark:bg-orange-950 rounded-lg">
                  <p className="text-3xl font-bold text-orange-600">0.97</p>
                  <p className="text-sm text-gray-600">AUROC</p>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="calibration">
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-950 rounded">
                  <span>Expected Calibration Error (ECE)</span>
                  <Badge className="bg-green-600">0.03</Badge>
                </div>
                <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-950 rounded">
                  <span>Conformal Coverage</span>
                  <Badge className="bg-green-600">95% ± 1%</Badge>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-3">
                  Models are well-calibrated with low ECE and valid conformal prediction coverage
                </p>
              </div>
            </TabsContent>

            <TabsContent value="latency">
              <div className="space-y-3">
                <div className="flex justify-between text-sm mb-1">
                  <span>Text Inference</span>
                  <span className="font-mono">145ms (P95)</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                  <div className="bg-blue-600 h-2 rounded-full" style={{ width: '29%' }} />
                </div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Vision Inference</span>
                  <span className="font-mono">230ms (P95)</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                  <div className="bg-green-600 h-2 rounded-full" style={{ width: '46%' }} />
                </div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Multimodal Fusion</span>
                  <span className="font-mono">520ms (P95)</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                  <div className="bg-purple-600 h-2 rounded-full" style={{ width: '100%' }} />
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Tool Use */}
      <Card>
        <CardHeader>
          <CardTitle>Clinical LLM Tool Use</CardTitle>
          <CardDescription>Function calling to calculators, order sets, and retrieval</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="p-3 border rounded">
              <strong>Clinical Calculators:</strong> APACHE, SOFA, CHA₂DS₂-VASc, Wells, PECARN
            </div>
            <div className="p-3 border rounded">
              <strong>Order Sets:</strong> Sepsis bundle, Stroke protocol, MI pathway
            </div>
            <div className="p-3 border rounded">
              <strong>Knowledge Retrieval:</strong> UpToDate, Guidelines, Drug interactions
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
