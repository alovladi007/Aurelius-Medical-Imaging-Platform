"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  FileImage,
  Search,
  Download,
  Upload,
  ZoomIn,
  ZoomOut,
  RotateCw,
  Eye,
  AlertTriangle,
  Shield,
  FileText,
  Activity,
  Layers,
  ChevronLeft,
  ChevronRight
} from 'lucide-react';

export default function DICOMBrowserPage() {
  const [selectedStudy, setSelectedStudy] = useState<any>(null);
  const [selectedImage, setSelectedImage] = useState<any>(null);
  const [windowLevel, setWindowLevel] = useState(40);
  const [windowWidth, setWindowWidth] = useState(400);
  const [showPHIMask, setShowPHIMask] = useState(false);
  const [inferenceResults, setInferenceResults] = useState<any>(null);

  const stats = {
    totalStudies: 890000,
    totalImages: 24000000,
    storageUsed: '24 TB',
    avgInferenceTime: '1.2s',
    phiDetections: 342
  };

  const studies = [
    {
      id: 'STUDY-001',
      patientId: 'P-2024-0198',
      patientName: 'DOE^JOHN',
      studyDate: '2024-11-12',
      modality: 'CR',
      description: 'Chest X-ray PA and Lateral',
      seriesCount: 2,
      imageCount: 2,
      bodyPart: 'Chest',
      institution: 'Massachusetts General Hospital',
      referringPhysician: 'Dr. Smith',
      status: 'Completed'
    },
    {
      id: 'STUDY-002',
      patientId: 'P-2024-0199',
      patientName: 'SMITH^JANE',
      studyDate: '2024-11-13',
      modality: 'CT',
      description: 'CT Chest with Contrast',
      seriesCount: 5,
      imageCount: 342,
      bodyPart: 'Chest',
      institution: 'Brigham and Women\'s Hospital',
      referringPhysician: 'Dr. Johnson',
      status: 'Completed'
    },
    {
      id: 'STUDY-003',
      patientId: 'P-2024-0200',
      patientName: 'BROWN^ROBERT',
      studyDate: '2024-11-14',
      modality: 'MR',
      description: 'MRI Brain without Contrast',
      seriesCount: 8,
      imageCount: 256,
      bodyPart: 'Brain',
      institution: 'Massachusetts General Hospital',
      referringPhysician: 'Dr. Williams',
      status: 'Completed'
    }
  ];

  const images = [
    {
      id: 'IMG-001',
      seriesNumber: 1,
      instanceNumber: 1,
      view: 'PA',
      dimensions: '2048x2048',
      bitDepth: 16,
      pixelSpacing: '0.143mm',
      hasPHI: true,
      phiRegions: [
        { x: 10, y: 10, width: 200, height: 30, text: 'Patient Name: John Doe' },
        { x: 10, y: 50, width: 180, height: 25, text: 'DOB: 01/15/1957' }
      ]
    },
    {
      id: 'IMG-002',
      seriesNumber: 2,
      instanceNumber: 1,
      view: 'Lateral',
      dimensions: '2048x2048',
      bitDepth: 16,
      pixelSpacing: '0.143mm',
      hasPHI: false,
      phiRegions: []
    }
  ];

  const cxrInference = {
    modelName: 'cxr_model',
    modelVersion: 'v2.1.0',
    timestamp: '2024-11-15T10:30:00Z',
    inferenceTime: 1.23,
    findings: [
      {
        label: 'Cardiomegaly',
        probability: 0.87,
        uncertainty: 'Low',
        bbox: { x: 512, y: 768, width: 512, height: 384 }
      },
      {
        label: 'Pleural Effusion',
        probability: 0.34,
        uncertainty: 'Moderate',
        bbox: { x: 256, y: 1024, width: 384, height: 256 }
      },
      {
        label: 'Pneumonia',
        probability: 0.12,
        uncertainty: 'High',
        bbox: null
      },
      {
        label: 'Pneumothorax',
        probability: 0.03,
        uncertainty: 'Low',
        bbox: null
      },
      {
        label: 'Consolidation',
        probability: 0.45,
        uncertainty: 'Moderate',
        bbox: { x: 768, y: 896, width: 256, height: 256 }
      },
      {
        label: 'Edema',
        probability: 0.28,
        uncertainty: 'Moderate',
        bbox: null
      },
      {
        label: 'Atelectasis',
        probability: 0.19,
        uncertainty: 'High',
        bbox: null
      },
      {
        label: 'No Finding',
        probability: 0.08,
        uncertainty: 'Low',
        bbox: null
      }
    ],
    calibration: {
      expectedCalibrationError: 0.042,
      reliability: 0.96,
      coverage: 0.89
    }
  };

  const handleInference = () => {
    // Simulate inference call
    setInferenceResults(cxrInference);
  };

  const handleDownload = (withPHI: boolean) => {
    if (withPHI && selectedImage?.hasPHI) {
      const reason = prompt('PHI detected. Enter reason for override:');
      if (!reason) {
        alert('Download blocked: PHI present and no override reason provided');
        return;
      }
      console.log(`PHI override: ${reason}`);
      // Log to audit trail
    }
    // Proceed with download
    alert('Download initiated (mock)');
  };

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <FileImage className="h-8 w-8 text-primary" />
            DICOM Browser & Inference
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Medical image viewer with AI inference and PHI protection
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Upload className="h-4 w-4 mr-2" />
            Upload DICOM
          </Button>
          <Button>
            <Search className="h-4 w-4 mr-2" />
            Search Studies
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Total Studies</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-primary">{stats.totalStudies.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Total Images</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-blue-600">{stats.totalImages.toLocaleString()}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Storage Used</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-green-600">{stats.storageUsed}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Inference</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-purple-600">{stats.avgInferenceTime}</p>
            <p className="text-xs text-gray-600">Per image</p>
          </CardContent>
        </Card>
        <Card className="border-orange-500">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">PHI Detections</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-orange-600">{stats.phiDetections}</p>
            <p className="text-xs text-gray-600">Last 24h</p>
          </CardContent>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Study List */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Studies</CardTitle>
              <CardDescription>Select a study to view images</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {studies.map((study) => (
                  <Card
                    key={study.id}
                    className={`cursor-pointer transition-all ${
                      selectedStudy?.id === study.id ? 'border-primary border-2' : 'hover:shadow-md'
                    }`}
                    onClick={() => {
                      setSelectedStudy(study);
                      setSelectedImage(images[0]);
                      setInferenceResults(null);
                    }}
                  >
                    <CardContent className="pt-4">
                      <div className="flex items-start justify-between mb-2">
                        <div>
                          <Badge variant="outline">{study.modality}</Badge>
                          <Badge variant="secondary" className="ml-1">{study.status}</Badge>
                        </div>
                      </div>
                      <p className="font-semibold text-sm">{study.description}</p>
                      <div className="mt-2 space-y-1 text-xs text-gray-600">
                        <div className="flex justify-between">
                          <span>Patient:</span>
                          <span className="font-semibold">{study.patientName}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Date:</span>
                          <span>{study.studyDate}</span>
                        </div>
                        <div className="flex justify-between">
                          <span>Images:</span>
                          <span>{study.imageCount}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Image Viewer */}
        <div className="lg:col-span-2 space-y-4">
          {selectedStudy && selectedImage ? (
            <>
              <Card>
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <div>
                      <CardTitle>Image Viewer</CardTitle>
                      <CardDescription>
                        {selectedStudy.description} - {selectedImage.view}
                      </CardDescription>
                    </div>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm">
                        <ChevronLeft className="h-4 w-4" />
                      </Button>
                      <Button variant="outline" size="sm">
                        <ChevronRight className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {/* Image Display Area */}
                  <div className="relative bg-black rounded-lg aspect-square flex items-center justify-center mb-4">
                    {/* Mock DICOM Image */}
                    <div className="w-full h-full bg-gradient-to-br from-gray-900 to-gray-700 rounded-lg flex items-center justify-center relative">
                      <div className="text-white text-center">
                        <FileImage className="h-24 w-24 mx-auto mb-4 opacity-50" />
                        <p className="text-sm opacity-75">DICOM Image Placeholder</p>
                        <p className="text-xs opacity-50 mt-2">{selectedImage.dimensions}</p>
                      </div>

                      {/* PHI Mask Overlay */}
                      {showPHIMask && selectedImage.hasPHI && (
                        <>
                          {selectedImage.phiRegions.map((region: any, idx: number) => (
                            <div
                              key={idx}
                              className="absolute bg-red-500 bg-opacity-50 border-2 border-red-600"
                              style={{
                                left: `${(region.x / 2048) * 100}%`,
                                top: `${(region.y / 2048) * 100}%`,
                                width: `${(region.width / 2048) * 100}%`,
                                height: `${(region.height / 2048) * 100}%`
                              }}
                            >
                              <div className="text-xs text-white p-1">PHI</div>
                            </div>
                          ))}
                        </>
                      )}

                      {/* Inference Bounding Boxes */}
                      {inferenceResults && inferenceResults.findings.map((finding: any, idx: number) => (
                        finding.bbox && (
                          <div
                            key={idx}
                            className={`absolute border-2 ${
                              finding.probability > 0.7 ? 'border-green-500' :
                              finding.probability > 0.4 ? 'border-yellow-500' :
                              'border-orange-500'
                            }`}
                            style={{
                              left: `${(finding.bbox.x / 2048) * 100}%`,
                              top: `${(finding.bbox.y / 2048) * 100}%`,
                              width: `${(finding.bbox.width / 2048) * 100}%`,
                              height: `${(finding.bbox.height / 2048) * 100}%`
                            }}
                          >
                            <div className="absolute -top-6 left-0 bg-black bg-opacity-75 text-white text-xs px-2 py-1 rounded">
                              {finding.label}: {(finding.probability * 100).toFixed(0)}%
                            </div>
                          </div>
                        )
                      ))}
                    </div>
                  </div>

                  {/* Window/Level Controls */}
                  <div className="space-y-4 mb-4">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <label className="text-sm font-semibold">Window Level: {windowLevel}</label>
                        <Eye className="h-4 w-4 text-gray-600" />
                      </div>
                      <Slider
                        value={[windowLevel]}
                        onValueChange={(v) => setWindowLevel(v[0])}
                        min={-100}
                        max={100}
                        step={1}
                        className="w-full"
                      />
                    </div>
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <label className="text-sm font-semibold">Window Width: {windowWidth}</label>
                        <Layers className="h-4 w-4 text-gray-600" />
                      </div>
                      <Slider
                        value={[windowWidth]}
                        onValueChange={(v) => setWindowWidth(v[0])}
                        min={1}
                        max={2000}
                        step={10}
                        className="w-full"
                      />
                    </div>
                  </div>

                  {/* Viewer Controls */}
                  <div className="flex flex-wrap gap-2 mb-4">
                    <Button variant="outline" size="sm">
                      <ZoomIn className="h-4 w-4 mr-1" />
                      Zoom In
                    </Button>
                    <Button variant="outline" size="sm">
                      <ZoomOut className="h-4 w-4 mr-1" />
                      Zoom Out
                    </Button>
                    <Button variant="outline" size="sm">
                      <RotateCw className="h-4 w-4 mr-1" />
                      Rotate
                    </Button>
                    <Button
                      variant={showPHIMask ? "default" : "outline"}
                      size="sm"
                      onClick={() => setShowPHIMask(!showPHIMask)}
                    >
                      <Shield className="h-4 w-4 mr-1" />
                      PHI Mask
                    </Button>
                  </div>

                  {/* PHI Warning */}
                  {selectedImage.hasPHI && (
                    <div className="p-3 bg-orange-50 dark:bg-orange-950 rounded-lg border border-orange-200 dark:border-orange-900 mb-4">
                      <div className="flex items-start gap-2">
                        <AlertTriangle className="h-5 w-5 text-orange-600 flex-shrink-0 mt-0.5" />
                        <div>
                          <p className="font-semibold text-sm text-orange-900 dark:text-orange-100">
                            PHI Detected via OCR
                          </p>
                          <p className="text-xs text-orange-700 dark:text-orange-300 mt-1">
                            {selectedImage.phiRegions.length} region(s) contain burned-in PHI.
                            Download requires override with reason.
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Action Buttons */}
                  <div className="flex gap-2">
                    <Button
                      className="flex-1"
                      onClick={handleInference}
                      disabled={selectedStudy.modality !== 'CR'}
                    >
                      <Activity className="h-4 w-4 mr-2" />
                      Run AI Inference
                    </Button>
                    <Button
                      variant="outline"
                      className="flex-1"
                      onClick={() => handleDownload(false)}
                    >
                      <Download className="h-4 w-4 mr-2" />
                      Download (De-ID)
                    </Button>
                    {selectedImage.hasPHI && (
                      <Button
                        variant="outline"
                        className="flex-1"
                        onClick={() => handleDownload(true)}
                      >
                        <Shield className="h-4 w-4 mr-2" />
                        Download (Override)
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Inference Results */}
              {inferenceResults && (
                <Card className="border-purple-200 dark:border-purple-900">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Activity className="h-5 w-5 text-purple-600" />
                      CXR AI Inference Results
                    </CardTitle>
                    <CardDescription>
                      Model: {inferenceResults.modelName} {inferenceResults.modelVersion} •
                      Time: {inferenceResults.inferenceTime}s
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Tabs defaultValue="findings" className="w-full">
                      <TabsList className="grid w-full grid-cols-2">
                        <TabsTrigger value="findings">Findings</TabsTrigger>
                        <TabsTrigger value="calibration">Calibration</TabsTrigger>
                      </TabsList>

                      <TabsContent value="findings" className="space-y-2 mt-4">
                        {inferenceResults.findings
                          .filter((f: any) => f.probability > 0.1)
                          .sort((a: any, b: any) => b.probability - a.probability)
                          .map((finding: any, idx: number) => (
                            <div key={idx} className="p-3 border rounded-lg">
                              <div className="flex items-center justify-between mb-2">
                                <span className="font-semibold text-sm">{finding.label}</span>
                                <div className="flex items-center gap-2">
                                  <Badge
                                    variant="outline"
                                    className={
                                      finding.uncertainty === 'Low' ? 'bg-green-50 text-green-800 dark:bg-green-950' :
                                      finding.uncertainty === 'Moderate' ? 'bg-yellow-50 text-yellow-800 dark:bg-yellow-950' :
                                      'bg-red-50 text-red-800 dark:bg-red-950'
                                    }
                                  >
                                    {finding.uncertainty} Uncertainty
                                  </Badge>
                                  <Badge variant="secondary">
                                    {(finding.probability * 100).toFixed(1)}%
                                  </Badge>
                                </div>
                              </div>
                              <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                                <div
                                  className={`h-2 rounded-full ${
                                    finding.probability > 0.7 ? 'bg-green-600' :
                                    finding.probability > 0.4 ? 'bg-yellow-600' :
                                    'bg-orange-600'
                                  }`}
                                  style={{ width: `${finding.probability * 100}%` }}
                                />
                              </div>
                            </div>
                          ))}
                      </TabsContent>

                      <TabsContent value="calibration" className="mt-4">
                        <div className="grid grid-cols-3 gap-4">
                          <div className="p-4 border rounded-lg text-center">
                            <p className="text-xs text-gray-600 mb-1">ECE</p>
                            <p className="text-2xl font-bold text-primary">
                              {inferenceResults.calibration.expectedCalibrationError.toFixed(3)}
                            </p>
                            <p className="text-xs text-gray-600 mt-1">Target: &lt;0.05</p>
                          </div>
                          <div className="p-4 border rounded-lg text-center">
                            <p className="text-xs text-gray-600 mb-1">Reliability</p>
                            <p className="text-2xl font-bold text-green-600">
                              {(inferenceResults.calibration.reliability * 100).toFixed(1)}%
                            </p>
                            <p className="text-xs text-gray-600 mt-1">Target: &gt;90%</p>
                          </div>
                          <div className="p-4 border rounded-lg text-center">
                            <p className="text-xs text-gray-600 mb-1">Coverage</p>
                            <p className="text-2xl font-bold text-blue-600">
                              {(inferenceResults.calibration.coverage * 100).toFixed(1)}%
                            </p>
                            <p className="text-xs text-gray-600 mt-1">Target: &gt;85%</p>
                          </div>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>
              )}
            </>
          ) : (
            <Card>
              <CardContent className="py-24">
                <div className="text-center text-gray-500">
                  <FileImage className="h-16 w-16 mx-auto mb-4 opacity-50" />
                  <p className="text-lg font-semibold">No Study Selected</p>
                  <p className="text-sm mt-2">Select a study from the list to view images</p>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
