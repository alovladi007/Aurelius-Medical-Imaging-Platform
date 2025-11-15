"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  BarChart3,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Target,
  Activity,
  Download,
  Filter,
  Scale
} from 'lucide-react';

export default function CalibrationPage() {
  const [selectedModel, setSelectedModel] = useState('cxr_model');

  const stats = {
    modelsTracked: 12,
    avgECE: 0.042,
    avgReliability: 96.3,
    avgCoverage: 89.2,
    calibrationScore: 'Excellent'
  };

  const models = [
    {
      id: 'cxr_model',
      name: 'CXR Classification',
      version: 'v2.1.0',
      type: 'Vision',
      metrics: {
        ece: 0.042,
        reliability: 96.3,
        coverage: 89.2,
        brier: 0.089
      },
      calibrationCurve: [
        { confidence: 0.1, accuracy: 0.12, count: 234 },
        { confidence: 0.2, accuracy: 0.22, count: 456 },
        { confidence: 0.3, accuracy: 0.31, count: 678 },
        { confidence: 0.4, accuracy: 0.42, count: 892 },
        { confidence: 0.5, accuracy: 0.51, count: 1245 },
        { confidence: 0.6, accuracy: 0.61, count: 1678 },
        { confidence: 0.7, accuracy: 0.72, count: 2134 },
        { confidence: 0.8, accuracy: 0.81, count: 2567 },
        { confidence: 0.9, accuracy: 0.91, count: 3012 },
        { confidence: 1.0, accuracy: 0.98, count: 1234 }
      ],
      status: 'Excellent'
    },
    {
      id: 'ekg_model',
      name: 'EKG Classification',
      version: 'v1.3.0',
      type: 'Signal',
      metrics: {
        ece: 0.038,
        reliability: 97.1,
        coverage: 92.4,
        brier: 0.072
      },
      calibrationCurve: [
        { confidence: 0.1, accuracy: 0.11, count: 189 },
        { confidence: 0.2, accuracy: 0.21, count: 342 },
        { confidence: 0.3, accuracy: 0.30, count: 567 },
        { confidence: 0.4, accuracy: 0.41, count: 789 },
        { confidence: 0.5, accuracy: 0.50, count: 1123 },
        { confidence: 0.6, accuracy: 0.60, count: 1456 },
        { confidence: 0.7, accuracy: 0.71, count: 1892 },
        { confidence: 0.8, accuracy: 0.80, count: 2234 },
        { confidence: 0.9, accuracy: 0.90, count: 2678 },
        { confidence: 1.0, accuracy: 0.97, count: 1089 }
      ],
      status: 'Excellent'
    },
    {
      id: 'sepsis_predictor',
      name: 'Sepsis Prediction',
      version: 'v3.0.1',
      type: 'Risk',
      metrics: {
        ece: 0.056,
        reliability: 94.8,
        coverage: 86.7,
        brier: 0.098
      },
      calibrationCurve: [
        { confidence: 0.1, accuracy: 0.14, count: 345 },
        { confidence: 0.2, accuracy: 0.24, count: 567 },
        { confidence: 0.3, accuracy: 0.34, count: 789 },
        { confidence: 0.4, accuracy: 0.44, count: 1012 },
        { confidence: 0.5, accuracy: 0.53, count: 1345 },
        { confidence: 0.6, accuracy: 0.63, count: 1678 },
        { confidence: 0.7, accuracy: 0.73, count: 2012 },
        { confidence: 0.8, accuracy: 0.82, count: 2345 },
        { confidence: 0.9, accuracy: 0.89, count: 2789 },
        { confidence: 1.0, accuracy: 0.95, count: 1456 }
      ],
      status: 'Good'
    }
  ];

  const selectedModelData = models.find(m => m.id === selectedModel) || models[0];

  const uncertaintyBands = [
    {
      band: 'Low (<70%)',
      action: 'Abstain - Request More Data',
      color: 'red',
      count: 145,
      percentage: 0.9
    },
    {
      band: 'Moderate (70-85%)',
      action: 'Flag for Human Review',
      color: 'yellow',
      count: 892,
      percentage: 5.6
    },
    {
      band: 'High (>85%)',
      action: 'Proceed with Recommendation',
      color: 'green',
      count: 14810,
      percentage: 93.5
    }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Target className="h-8 w-8 text-primary" />
            Calibration Dashboard
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Model uncertainty and reliability monitoring
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Filter className="h-4 w-4 mr-2" />
            Filter Models
          </Button>
          <Button>
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Models Tracked</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-primary">{stats.modelsTracked}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg ECE</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-blue-600">{stats.avgECE.toFixed(3)}</p>
            <p className="text-xs text-gray-600">Target: &lt;0.05</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Reliability</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-green-600">{stats.avgReliability.toFixed(1)}%</p>
            <p className="text-xs text-gray-600">Target: &gt;90%</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Coverage</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-purple-600">{stats.avgCoverage.toFixed(1)}%</p>
            <p className="text-xs text-gray-600">Target: &gt;85%</p>
          </CardContent>
        </Card>
        <Card className="border-green-500">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">System Status</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">{stats.calibrationScore}</p>
            <p className="text-xs text-gray-600">All models</p>
          </CardContent>
        </Card>
      </div>

      {/* Model Selection */}
      <Card>
        <CardHeader>
          <CardTitle>Select Model</CardTitle>
          <CardDescription>Choose a model to view calibration metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4">
            {models.map((model) => (
              <Card
                key={model.id}
                className={`cursor-pointer transition-all ${
                  selectedModel === model.id ? 'border-primary border-2' : 'hover:shadow-md'
                }`}
                onClick={() => setSelectedModel(model.id)}
              >
                <CardContent className="pt-4">
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <Badge variant="outline">{model.type}</Badge>
                      <Badge
                        variant="secondary"
                        className={`ml-1 ${
                          model.status === 'Excellent' ? 'bg-green-100 text-green-800 dark:bg-green-950' :
                          model.status === 'Good' ? 'bg-blue-100 text-blue-800 dark:bg-blue-950' :
                          'bg-yellow-100 text-yellow-800 dark:bg-yellow-950'
                        }`}
                      >
                        {model.status}
                      </Badge>
                    </div>
                  </div>
                  <p className="font-semibold mb-1">{model.name}</p>
                  <p className="text-xs text-gray-600">{model.version}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Detailed Metrics */}
      <Tabs defaultValue="calibration" className="w-full">
        <TabsList className="grid w-full grid-cols-4">
          <TabsTrigger value="calibration">Calibration Curve</TabsTrigger>
          <TabsTrigger value="metrics">Metrics</TabsTrigger>
          <TabsTrigger value="uncertainty">Uncertainty Bands</TabsTrigger>
          <TabsTrigger value="coverage">Coverage Analysis</TabsTrigger>
        </TabsList>

        <TabsContent value="calibration" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>{selectedModelData.name} - Calibration Curve</CardTitle>
              <CardDescription>
                Expected Calibration Error: {selectedModelData.metrics.ece.toFixed(3)}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Calibration Plot */}
              <div className="relative h-96 bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                <svg className="w-full h-full" viewBox="0 0 400 400">
                  {/* Grid */}
                  {[0, 0.2, 0.4, 0.6, 0.8, 1.0].map((val) => (
                    <g key={val}>
                      <line
                        x1="50"
                        y1={350 - val * 300}
                        x2="350"
                        y2={350 - val * 300}
                        stroke="currentColor"
                        strokeWidth="0.5"
                        className="text-gray-300 dark:text-gray-700"
                        strokeDasharray="4"
                      />
                      <text
                        x="35"
                        y={355 - val * 300}
                        className="text-xs fill-current text-gray-600 dark:text-gray-400"
                        textAnchor="end"
                      >
                        {val.toFixed(1)}
                      </text>
                      <line
                        x1={50 + val * 300}
                        y1="50"
                        x2={50 + val * 300}
                        y2="350"
                        stroke="currentColor"
                        strokeWidth="0.5"
                        className="text-gray-300 dark:text-gray-700"
                        strokeDasharray="4"
                      />
                      <text
                        x={50 + val * 300}
                        y="370"
                        className="text-xs fill-current text-gray-600 dark:text-gray-400"
                        textAnchor="middle"
                      >
                        {val.toFixed(1)}
                      </text>
                    </g>
                  ))}

                  {/* Perfect calibration line */}
                  <line
                    x1="50"
                    y1="350"
                    x2="350"
                    y2="50"
                    stroke="currentColor"
                    strokeWidth="2"
                    className="text-gray-400 dark:text-gray-600"
                    strokeDasharray="8"
                  />

                  {/* Actual calibration curve */}
                  <polyline
                    points={selectedModelData.calibrationCurve
                      .map((point) => {
                        const x = 50 + point.confidence * 300;
                        const y = 350 - point.accuracy * 300;
                        return `${x},${y}`;
                      })
                      .join(' ')}
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="3"
                    className="text-blue-600"
                  />

                  {/* Data points */}
                  {selectedModelData.calibrationCurve.map((point, idx) => {
                    const x = 50 + point.confidence * 300;
                    const y = 350 - point.accuracy * 300;
                    return (
                      <circle
                        key={idx}
                        cx={x}
                        cy={y}
                        r="5"
                        className="fill-current text-blue-600"
                      />
                    );
                  })}

                  {/* Labels */}
                  <text
                    x="200"
                    y="390"
                    className="text-sm font-semibold fill-current text-gray-700 dark:text-gray-300"
                    textAnchor="middle"
                  >
                    Predicted Confidence
                  </text>
                  <text
                    x="15"
                    y="200"
                    className="text-sm font-semibold fill-current text-gray-700 dark:text-gray-300"
                    textAnchor="middle"
                    transform="rotate(-90, 15, 200)"
                  >
                    Observed Accuracy
                  </text>
                </svg>
              </div>

              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                <p className="text-sm font-semibold mb-2">Interpretation:</p>
                <ul className="text-xs space-y-1 text-gray-700 dark:text-gray-300">
                  <li>• Perfect calibration: points fall on diagonal line</li>
                  <li>• ECE &lt;0.05: Excellent calibration (model confidence matches accuracy)</li>
                  <li>• Reliability {selectedModelData.metrics.reliability.toFixed(1)}%: High agreement between confidence and actual performance</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="metrics" className="mt-6">
          <div className="grid grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Calibration Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold">Expected Calibration Error (ECE)</span>
                      <Badge
                        className={
                          selectedModelData.metrics.ece < 0.05 ? 'bg-green-600' :
                          selectedModelData.metrics.ece < 0.10 ? 'bg-yellow-600' :
                          'bg-red-600'
                        }
                      >
                        {selectedModelData.metrics.ece < 0.05 ? 'Excellent' :
                         selectedModelData.metrics.ece < 0.10 ? 'Good' : 'Needs Improvement'}
                      </Badge>
                    </div>
                    <p className="text-3xl font-bold text-primary mb-1">
                      {selectedModelData.metrics.ece.toFixed(3)}
                    </p>
                    <p className="text-xs text-gray-600">Target: &lt;0.05</p>
                  </div>

                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold">Reliability</span>
                      <CheckCircle className="h-5 w-5 text-green-600" />
                    </div>
                    <p className="text-3xl font-bold text-green-600 mb-1">
                      {selectedModelData.metrics.reliability.toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-600">Target: &gt;90%</p>
                  </div>

                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold">Brier Score</span>
                    </div>
                    <p className="text-3xl font-bold text-blue-600 mb-1">
                      {selectedModelData.metrics.brier.toFixed(3)}
                    </p>
                    <p className="text-xs text-gray-600">Lower is better</p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Coverage Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-semibold">Coverage</span>
                      <CheckCircle className="h-5 w-5 text-purple-600" />
                    </div>
                    <p className="text-3xl font-bold text-purple-600 mb-1">
                      {selectedModelData.metrics.coverage.toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-600">Target: &gt;85%</p>
                  </div>

                  <div className="p-4 border rounded-lg">
                    <p className="text-sm font-semibold mb-3">Coverage by Confidence</p>
                    <div className="space-y-2">
                      {selectedModelData.calibrationCurve.slice(-3).map((point, idx) => (
                        <div key={idx}>
                          <div className="flex justify-between text-xs mb-1">
                            <span>{(point.confidence * 100).toFixed(0)}% confidence</span>
                            <span className="font-semibold">{point.count} samples</span>
                          </div>
                          <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                            <div
                              className="bg-purple-600 h-2 rounded-full"
                              style={{
                                width: `${(point.count / 3000) * 100}%`
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="uncertainty" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Uncertainty-Based Action Bands</CardTitle>
              <CardDescription>
                Decision thresholds based on confidence levels
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {uncertaintyBands.map((band, idx) => (
                  <div key={idx} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <div className={`w-4 h-4 rounded-full bg-${band.color}-600`}></div>
                        <span className="font-semibold">{band.band}</span>
                      </div>
                      <div className="flex items-center gap-4">
                        <Badge variant="outline">{band.count.toLocaleString()} cases</Badge>
                        <Badge variant="secondary">{band.percentage.toFixed(1)}%</Badge>
                      </div>
                    </div>
                    <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded">
                      <p className="text-sm">
                        <span className="font-semibold">Action:</span> {band.action}
                      </p>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-3 mt-3">
                      <div
                        className={`bg-${band.color}-600 h-3 rounded-full`}
                        style={{ width: `${band.percentage}%` }}
                      />
                    </div>
                  </div>
                ))}

                <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-900">
                  <p className="font-semibold text-sm mb-2 flex items-center gap-2">
                    <Scale className="h-4 w-4 text-blue-600" />
                    Abstention Strategy
                  </p>
                  <p className="text-xs text-gray-700 dark:text-gray-300">
                    When confidence is low (&lt;70%), the system abstains and requests additional data rather than
                    making unreliable recommendations. This ensures high reliability on predictions that are made.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="coverage" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Coverage vs Risk Analysis</CardTitle>
              <CardDescription>
                Trade-off between coverage and prediction reliability
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold mb-4">Current Operating Point</h3>
                  <div className="space-y-3">
                    <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg border border-green-200 dark:border-green-900">
                      <p className="text-sm font-semibold mb-1">Coverage</p>
                      <p className="text-2xl font-bold text-green-600">
                        {selectedModelData.metrics.coverage.toFixed(1)}%
                      </p>
                      <p className="text-xs text-gray-600 mt-1">
                        Of all cases receive predictions
                      </p>
                    </div>
                    <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-900">
                      <p className="text-sm font-semibold mb-1">Reliability</p>
                      <p className="text-2xl font-bold text-blue-600">
                        {selectedModelData.metrics.reliability.toFixed(1)}%
                      </p>
                      <p className="text-xs text-gray-600 mt-1">
                        Of predictions are accurate
                      </p>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold mb-4">Alternative Thresholds</h3>
                  <div className="space-y-2 text-sm">
                    <div className="p-3 border rounded-lg">
                      <div className="flex justify-between mb-2">
                        <span className="font-semibold">Threshold: 60%</span>
                        <Badge variant="outline">Max Coverage</Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>Coverage: <span className="font-semibold">97.2%</span></div>
                        <div>Reliability: <span className="font-semibold">89.4%</span></div>
                      </div>
                    </div>
                    <div className="p-3 border rounded-lg bg-green-50 dark:bg-green-950">
                      <div className="flex justify-between mb-2">
                        <span className="font-semibold">Threshold: 70%</span>
                        <Badge className="bg-green-600">Current</Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>Coverage: <span className="font-semibold">{selectedModelData.metrics.coverage.toFixed(1)}%</span></div>
                        <div>Reliability: <span className="font-semibold">{selectedModelData.metrics.reliability.toFixed(1)}%</span></div>
                      </div>
                    </div>
                    <div className="p-3 border rounded-lg">
                      <div className="flex justify-between mb-2">
                        <span className="font-semibold">Threshold: 85%</span>
                        <Badge variant="outline">Max Reliability</Badge>
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        <div>Coverage: <span className="font-semibold">76.3%</span></div>
                        <div>Reliability: <span className="font-semibold">98.9%</span></div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
