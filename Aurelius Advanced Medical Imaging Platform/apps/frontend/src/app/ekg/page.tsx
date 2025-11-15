"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Activity,
  Upload,
  Play,
  Pause,
  ZoomIn,
  ZoomOut,
  Download,
  AlertCircle,
  CheckCircle,
  TrendingUp,
  FileText,
  Brain
} from 'lucide-react';

export default function EKGInferencePage() {
  const [selectedEKG, setSelectedEKG] = useState<any>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [inferenceResults, setInferenceResults] = useState<any>(null);
  const [zoom, setZoom] = useState(1.0);

  const stats = {
    totalEKGs: 125000,
    processedToday: 342,
    avgInferenceTime: '0.8s',
    accuracy: '96.4%',
    abnormalDetected: 23
  };

  const ekgRecordings = [
    {
      id: 'EKG-001',
      patientId: 'P-2024-0198',
      patientName: 'John Doe',
      recordingDate: '2024-11-14T10:30:00Z',
      duration: 10,
      leads: 12,
      samplingRate: 500,
      device: 'GE MAC 5500',
      indication: 'Chest pain',
      status: 'Completed'
    },
    {
      id: 'EKG-002',
      patientId: 'P-2024-0199',
      patientName: 'Jane Smith',
      recordingDate: '2024-11-14T11:15:00Z',
      duration: 10,
      leads: 12,
      samplingRate: 500,
      device: 'Philips PageWriter',
      indication: 'Pre-op screening',
      status: 'Completed'
    },
    {
      id: 'EKG-003',
      patientId: 'P-2024-0200',
      patientName: 'Robert Brown',
      recordingDate: '2024-11-14T12:00:00Z',
      duration: 10,
      leads: 12,
      samplingRate: 500,
      device: 'GE MAC 5500',
      indication: 'Palpitations',
      status: 'Completed'
    }
  ];

  const mockInference = {
    modelName: 'ekg_model',
    modelVersion: 'v1.3.0',
    timestamp: '2024-11-15T10:30:00Z',
    inferenceTime: 0.82,
    classifications: [
      {
        label: 'Normal Sinus Rhythm',
        probability: 0.12,
        uncertainty: 'Low',
        description: 'Regular rhythm, normal rate (60-100 bpm), normal P-QRS-T morphology'
      },
      {
        label: 'Atrial Fibrillation',
        probability: 0.89,
        uncertainty: 'Low',
        description: 'Irregularly irregular rhythm, absent P waves, variable R-R intervals'
      },
      {
        label: 'Atrial Flutter',
        probability: 0.08,
        uncertainty: 'Moderate',
        description: 'Regular atrial rate ~300 bpm with sawtooth pattern, variable AV block'
      },
      {
        label: 'Ventricular Tachycardia',
        probability: 0.03,
        uncertainty: 'High',
        description: 'Wide QRS tachycardia >100 bpm, often AV dissociation'
      }
    ],
    measurements: {
      heartRate: 92,
      prInterval: null,
      qrsDuration: 108,
      qtInterval: 380,
      qtcInterval: 420,
      pAxis: null,
      qrsAxis: -15,
      tAxis: 25
    },
    calibration: {
      expectedCalibrationError: 0.038,
      reliability: 0.97,
      coverage: 0.92
    },
    recommendations: [
      'Consider anticoagulation per CHA₂DS₂-VASc score',
      'Rate control vs rhythm control strategy',
      'Echocardiogram to assess LA size and function',
      'Assess for reversible causes (alcohol, thyroid, electrolytes)'
    ]
  };

  const handleInference = () => {
    setInferenceResults(mockInference);
  };

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Activity className="h-8 w-8 text-red-600" />
            EKG Waveform & AI Inference
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            12-lead ECG analysis with 4-label classification and uncertainty quantification
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Upload className="h-4 w-4 mr-2" />
            Upload EKG
          </Button>
          <Button>
            <FileText className="h-4 w-4 mr-2" />
            Reports
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Total EKGs</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-primary">{stats.totalEKGs.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Processed Today</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-blue-600">{stats.processedToday}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Inference</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-green-600">{stats.avgInferenceTime}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Model Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-purple-600">{stats.accuracy}</p>
            <p className="text-xs text-gray-600">On test set</p>
          </CardContent>
        </Card>
        <Card className="border-orange-500">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Abnormal Today</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-orange-600">{stats.abnormalDetected}</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* EKG List */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>EKG Recordings</CardTitle>
              <CardDescription>Select a recording to analyze</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {ekgRecordings.map((ekg) => (
                  <Card
                    key={ekg.id}
                    className={`cursor-pointer transition-all ${
                      selectedEKG?.id === ekg.id ? 'border-primary border-2' : 'hover:shadow-md'
                    }`}
                    onClick={() => {
                      setSelectedEKG(ekg);
                      setInferenceResults(null);
                    }}
                  >
                    <CardContent className="pt-4">
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <Badge variant="outline">{ekg.id}</Badge>
                          <Badge variant="secondary" className="ml-1">{ekg.status}</Badge>
                        </div>
                      </div>
                      <p className="font-semibold text-sm">{ekg.indication}</p>
                      <div className="mt-2 space-y-1 text-xs text-gray-600">
                        <div className="flex justify-between">
                          <span>Patient:</span>
                          <span className="font-semibold">{ekg.patientName}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Date:</span>
                          <span>{new Date(ekg.recordingDate).toLocaleDateString()}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Leads:</span>
                          <span>{ekg.leads}-lead</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Duration:</span>
                          <span>{ekg.duration}s</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Waveform Viewer & Results */}
        <div className="lg:col-span-2 space-y-4">
          {selectedEKG ? (
            <>
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>EKG Waveform</CardTitle>
                      <CardDescription>
                        {selectedEKG.leads}-lead, {selectedEKG.samplingRate} Hz, {selectedEKG.duration}s
                      </CardDescription>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setIsPlaying(!isPlaying)}
                      >
                        {isPlaying ? (
                          <Pause className="h-4 w-4" />
                        ) : (
                          <Play className="h-4 w-4" />
                        )}
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setZoom(Math.min(zoom + 0.25, 3))}
                      >
                        <ZoomIn className="h-4 w-4" />
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => setZoom(Math.max(zoom - 0.25, 0.5))}
                      >
                        <ZoomOut className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Mock 12-Lead EKG Display */}
                  <div className="bg-pink-50 dark:bg-pink-950 rounded-lg p-4 mb-4">
                    <div className="grid grid-cols-3 gap-4">
                      {['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'].map((lead) => (
                        <div key={lead} className="space-y-1">
                          <div className="text-xs font-semibold text-gray-700 dark:text-gray-300">{lead}</div>
                          <div className="h-16 bg-white dark:bg-gray-900 rounded border border-pink-200 dark:border-pink-800 relative overflow-hidden">
                            {/* Mock waveform - in production this would be actual EKG data */}
                            <svg className="w-full h-full" viewBox="0 0 100 50">
                              <path
                                d={`M 0,25 ${Array.from({ length: 20 }, (_, i) => {
                                  const x = (i / 20) * 100;
                                  const y = 25 + Math.sin(i * 0.5) * 10 + Math.random() * 3;
                                  return `L ${x},${y}`;
                                }).join(' ')}`}
                                stroke="currentColor"
                                strokeWidth="0.5"
                                fill="none"
                                className="text-red-600"
                              />
                            </svg>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="flex gap-2 mb-4">
                    <Button className="flex-1" onClick={handleInference}>
                      <Brain className="h-4 w-4 mr-2" />
                      Run AI Inference
                    </Button>
                    <Button variant="outline" className="flex-1">
                      <Download className="h-4 w-4 mr-2" />
                      Export PDF
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Inference Results */}
              {inferenceResults && (
                <>
                  <Card className="border-purple-200 dark:border-purple-900">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Brain className="h-5 w-5 text-purple-600" />
                        AI Classification Results
                      </CardTitle>
                      <CardDescription>
                        Model: {inferenceResults.modelName} {inferenceResults.modelVersion} •
                        Time: {inferenceResults.inferenceTime}s
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      <Tabs defaultValue="classification" className="w-full">
                        <TabsList className="grid w-full grid-cols-3">
                          <TabsTrigger value="classification">Classification</TabsTrigger>
                          <TabsTrigger value="measurements">Measurements</TabsTrigger>
                          <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
                        </TabsList>

                        <TabsContent value="classification" className="space-y-3 mt-4">
                          {inferenceResults.classifications
                            .sort((a: any, b: any) => b.probability - a.probability)
                            .map((classification: any, idx: number) => (
                              <div key={idx} className="p-4 border rounded-lg">
                                <div className="flex items-center justify-between mb-3">
                                  <div className="flex items-center gap-2">
                                    <span className="font-semibold">{classification.label}</span>
                                    {classification.probability > 0.7 && (
                                      <Badge className="bg-green-600">Detected</Badge>
                                    )}
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <Badge
                                      variant="outline"
                                      className={
                                        classification.uncertainty === 'Low' ? 'bg-green-50 text-green-800 dark:bg-green-950' :
                                        classification.uncertainty === 'Moderate' ? 'bg-yellow-50 text-yellow-800 dark:bg-yellow-950' :
                                        'bg-red-50 text-red-800 dark:bg-red-950'
                                      }
                                    >
                                      {classification.uncertainty} Uncertainty
                                    </Badge>
                                    <Badge variant="secondary">
                                      {(classification.probability * 100).toFixed(1)}%
                                    </Badge>
                                  </div>
                                </div>
                                <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-3 mb-2">
                                  <div
                                    className={`h-3 rounded-full ${
                                      classification.probability > 0.7 ? 'bg-green-600' :
                                      classification.probability > 0.4 ? 'bg-yellow-600' :
                                      'bg-orange-600'
                                    }`}
                                    style={{ width: `${classification.probability * 100}%` }}
                                  />
                                </div>
                                <p className="text-xs text-gray-600 dark:text-gray-400">
                                  {classification.description}
                                </p>
                              </div>
                            ))}

                          {/* Calibration Metrics */}
                          <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-900">
                            <p className="font-semibold text-sm mb-3">Calibration Metrics</p>
                            <div className="grid grid-cols-3 gap-4">
                              <div className="text-center">
                                <p className="text-xs text-gray-600 mb-1">ECE</p>
                                <p className="text-xl font-bold text-blue-600">
                                  {inferenceResults.calibration.expectedCalibrationError.toFixed(3)}
                                </p>
                              </div>
                              <div className="text-center">
                                <p className="text-xs text-gray-600 mb-1">Reliability</p>
                                <p className="text-xl font-bold text-green-600">
                                  {(inferenceResults.calibration.reliability * 100).toFixed(1)}%
                                </p>
                              </div>
                              <div className="text-center">
                                <p className="text-xs text-gray-600 mb-1">Coverage</p>
                                <p className="text-xl font-bold text-purple-600">
                                  {(inferenceResults.calibration.coverage * 100).toFixed(1)}%
                                </p>
                              </div>
                            </div>
                          </div>
                        </TabsContent>

                        <TabsContent value="measurements" className="mt-4">
                          <div className="grid grid-cols-2 gap-4">
                            <div className="p-3 border rounded-lg">
                              <p className="text-xs text-gray-600 mb-1">Heart Rate</p>
                              <p className="text-2xl font-bold">
                                {inferenceResults.measurements.heartRate}
                                <span className="text-sm font-normal ml-1">bpm</span>
                              </p>
                            </div>
                            <div className="p-3 border rounded-lg">
                              <p className="text-xs text-gray-600 mb-1">PR Interval</p>
                              <p className="text-2xl font-bold">
                                {inferenceResults.measurements.prInterval || 'N/A'}
                                {inferenceResults.measurements.prInterval && <span className="text-sm font-normal ml-1">ms</span>}
                              </p>
                            </div>
                            <div className="p-3 border rounded-lg">
                              <p className="text-xs text-gray-600 mb-1">QRS Duration</p>
                              <p className="text-2xl font-bold">
                                {inferenceResults.measurements.qrsDuration}
                                <span className="text-sm font-normal ml-1">ms</span>
                              </p>
                            </div>
                            <div className="p-3 border rounded-lg">
                              <p className="text-xs text-gray-600 mb-1">QTc Interval</p>
                              <p className="text-2xl font-bold">
                                {inferenceResults.measurements.qtcInterval}
                                <span className="text-sm font-normal ml-1">ms</span>
                              </p>
                            </div>
                            <div className="p-3 border rounded-lg">
                              <p className="text-xs text-gray-600 mb-1">QRS Axis</p>
                              <p className="text-2xl font-bold">
                                {inferenceResults.measurements.qrsAxis}°
                              </p>
                            </div>
                            <div className="p-3 border rounded-lg">
                              <p className="text-xs text-gray-600 mb-1">T Axis</p>
                              <p className="text-2xl font-bold">
                                {inferenceResults.measurements.tAxis}°
                              </p>
                            </div>
                          </div>
                        </TabsContent>

                        <TabsContent value="recommendations" className="mt-4">
                          <div className="space-y-2">
                            {inferenceResults.recommendations.map((rec: string, idx: number) => (
                              <div key={idx} className="p-3 bg-green-50 dark:bg-green-950 rounded-lg border border-green-200 dark:border-green-900">
                                <div className="flex items-start gap-2">
                                  <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                                  <p className="text-sm">{rec}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        </TabsContent>
                      </Tabs>
                    </CardContent>
                  </Card>
                </>
              )}
            </>
          ) : (
            <Card>
              <CardContent className="py-24">
                <div className="text-center text-gray-500">
                  <Activity className="h-16 w-16 mx-auto mb-4 opacity-50" />
                  <p className="text-lg font-semibold">No EKG Selected</p>
                  <p className="text-sm mt-2">Select an EKG recording from the list to analyze</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
