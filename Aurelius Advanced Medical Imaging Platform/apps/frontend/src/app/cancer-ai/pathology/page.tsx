'use client'

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import {
  Microscope,
  Upload,
  Scan,
  ZoomIn,
  ZoomOut,
  Download,
  Share2,
  AlertTriangle,
  CheckCircle,
  Grid3x3,
  Layers,
  Eye,
  Activity
} from 'lucide-react'

export default function PathologyPage() {
  const [selectedSlide, setSelectedSlide] = useState<string | null>('WSI-001')
  const [zoomLevel, setZoomLevel] = useState(50)

  const slides = [
    {
      id: 'WSI-001',
      name: 'Breast Tissue - Biopsy',
      patientId: 'PT-45782',
      uploadDate: '2024-01-15',
      status: 'analyzed',
      magnification: '40x',
      fileSize: '2.4 GB',
      findings: [
        { type: 'Ductal Carcinoma', confidence: 94.2, location: 'Upper right quadrant' },
        { type: 'Atypical cells', confidence: 87.5, location: 'Margin area' }
      ],
      thumbnail: '/api/placeholder/400/300'
    },
    {
      id: 'WSI-002',
      name: 'Prostate Tissue - Core',
      patientId: 'PT-45760',
      uploadDate: '2024-01-14',
      status: 'analyzing',
      magnification: '20x',
      fileSize: '1.8 GB',
      findings: [],
      thumbnail: '/api/placeholder/400/300'
    },
    {
      id: 'WSI-003',
      name: 'Skin Lesion - Excision',
      patientId: 'PT-45755',
      uploadDate: '2024-01-13',
      status: 'analyzed',
      magnification: '40x',
      fileSize: '1.2 GB',
      findings: [
        { type: 'Melanoma', confidence: 91.8, location: 'Central region' }
      ],
      thumbnail: '/api/placeholder/400/300'
    }
  ]

  const selectedSlideData = slides.find(s => s.id === selectedSlide)

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-pink-50 to-slate-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
          Digital Pathology & WSI Analysis
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          AI-powered whole slide imaging analysis for cancer detection
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Viewer */}
        <div className="lg:col-span-2 space-y-6">
          {/* Slide Viewer */}
          <Card className="bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm overflow-hidden">
            {/* Viewer Controls */}
            <div className="p-4 bg-slate-100 dark:bg-slate-900 border-b border-slate-200 dark:border-slate-700">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-pink-100 dark:bg-pink-900/50 rounded-lg">
                    <Microscope className="h-5 w-5 text-pink-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-slate-900 dark:text-white">
                      {selectedSlideData?.name}
                    </h3>
                    <p className="text-sm text-slate-600 dark:text-slate-400">
                      {selectedSlideData?.patientId} • {selectedSlideData?.magnification}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm">
                    <Grid3x3 className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <Layers className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <Share2 className="h-4 w-4" />
                  </Button>
                  <Button variant="outline" size="sm">
                    <Download className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </div>

            {/* Viewer Canvas */}
            <div className="relative bg-slate-900 aspect-video flex items-center justify-center">
              {selectedSlideData ? (
                <div className="text-center">
                  <Microscope className="h-24 w-24 text-slate-600 mx-auto mb-4" />
                  <p className="text-slate-400 mb-2">Whole Slide Imaging Viewer</p>
                  <p className="text-sm text-slate-500">
                    Interactive pathology viewer would load here
                  </p>
                  <div className="mt-6 flex items-center justify-center gap-4">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setZoomLevel(Math.max(10, zoomLevel - 10))}
                      className="bg-slate-800 border-slate-700 text-white hover:bg-slate-700"
                    >
                      <ZoomOut className="h-4 w-4" />
                    </Button>
                    <span className="text-white text-sm font-medium">{zoomLevel}%</span>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setZoomLevel(Math.min(200, zoomLevel + 10))}
                      className="bg-slate-800 border-slate-700 text-white hover:bg-slate-700"
                    >
                      <ZoomIn className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              ) : (
                <div className="text-center">
                  <Eye className="h-24 w-24 text-slate-600 mx-auto mb-4" />
                  <p className="text-slate-400">Select a slide to view</p>
                </div>
              )}
            </div>

            {/* Zoom Control */}
            <div className="p-4 bg-slate-100 dark:bg-slate-900 border-t border-slate-200 dark:border-slate-700">
              <div className="flex items-center gap-4">
                <span className="text-sm text-slate-600 dark:text-slate-400 min-w-[100px]">
                  Zoom Level
                </span>
                <input
                  type="range"
                  min="10"
                  max="200"
                  value={zoomLevel}
                  onChange={(e) => setZoomLevel(Number(e.target.value))}
                  className="flex-1 h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-pink-600"
                />
                <span className="text-sm font-medium text-slate-900 dark:text-white min-w-[50px] text-right">
                  {zoomLevel}%
                </span>
              </div>
            </div>
          </Card>

          {/* AI Findings */}
          {selectedSlideData && selectedSlideData.findings.length > 0 && (
            <Card className="p-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-orange-100 dark:bg-orange-900/50 rounded-lg">
                  <Activity className="h-5 w-5 text-orange-600" />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                    AI Detected Findings
                  </h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    {selectedSlideData.findings.length} abnormalities detected
                  </p>
                </div>
              </div>

              <div className="space-y-4">
                {selectedSlideData.findings.map((finding, index) => (
                  <div
                    key={index}
                    className="p-4 bg-gradient-to-r from-orange-50 to-red-50 dark:from-orange-900/20 dark:to-red-900/20 rounded-lg border border-orange-200 dark:border-orange-800"
                  >
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex items-center gap-3">
                        <AlertTriangle className="h-5 w-5 text-orange-600" />
                        <div>
                          <p className="font-semibold text-slate-900 dark:text-white">
                            {finding.type}
                          </p>
                          <p className="text-sm text-slate-600 dark:text-slate-400">
                            Location: {finding.location}
                          </p>
                        </div>
                      </div>
                      <Badge className="bg-orange-600 text-white">
                        {finding.confidence}% Confidence
                      </Badge>
                    </div>
                    <div className="flex gap-2">
                      <Button size="sm" variant="outline" className="border-orange-300 dark:border-orange-700">
                        <Eye className="h-3 w-3 mr-2" />
                        View Region
                      </Button>
                      <Button size="sm" variant="outline" className="border-orange-300 dark:border-orange-700">
                        <Download className="h-3 w-3 mr-2" />
                        Export
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </Card>
          )}
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Upload Section */}
          <Card className="p-6 bg-gradient-to-br from-pink-500 to-purple-500 text-white">
            <Upload className="h-8 w-8 mb-4 opacity-80" />
            <h3 className="text-lg font-bold mb-2">Upload New Slide</h3>
            <p className="text-sm text-pink-100 mb-6">
              Upload whole slide images for AI-powered pathology analysis
            </p>
            <Button className="w-full bg-white text-pink-600 hover:bg-pink-50">
              <Upload className="h-4 w-4 mr-2" />
              Select Files
            </Button>
            <p className="text-xs text-pink-200 mt-3">
              Supports: .svs, .ndpi, .tiff, .mrxs
            </p>
          </Card>

          {/* Slide List */}
          <Card className="p-4 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white mb-4">
              Recent Slides ({slides.length})
            </h3>
            <div className="space-y-2">
              {slides.map((slide) => (
                <button
                  key={slide.id}
                  onClick={() => setSelectedSlide(slide.id)}
                  className={`w-full p-3 rounded-lg text-left transition-all ${
                    selectedSlide === slide.id
                      ? 'bg-pink-100 dark:bg-pink-900/30 border-2 border-pink-500'
                      : 'bg-slate-50 dark:bg-slate-900/50 border-2 border-transparent hover:border-slate-300 dark:hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <p className="text-sm font-medium text-slate-900 dark:text-white truncate">
                      {slide.name}
                    </p>
                    {slide.status === 'analyzed' ? (
                      <CheckCircle className="h-4 w-4 text-green-600 flex-shrink-0" />
                    ) : (
                      <Scan className="h-4 w-4 text-orange-600 flex-shrink-0 animate-pulse" />
                    )}
                  </div>
                  <div className="flex items-center justify-between">
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      {slide.patientId}
                    </p>
                    <p className="text-xs text-slate-500 dark:text-slate-400">
                      {slide.fileSize}
                    </p>
                  </div>
                </button>
              ))}
            </div>
          </Card>

          {/* Stats */}
          <Card className="p-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white mb-4">
              Analysis Statistics
            </h3>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-slate-600 dark:text-slate-400">Total Slides</span>
                  <span className="font-medium text-slate-900 dark:text-white">
                    {slides.length}
                  </span>
                </div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-slate-600 dark:text-slate-400">Analyzed</span>
                  <span className="font-medium text-green-600">
                    {slides.filter(s => s.status === 'analyzed').length}
                  </span>
                </div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-slate-600 dark:text-slate-400">Processing</span>
                  <span className="font-medium text-orange-600">
                    {slides.filter(s => s.status === 'analyzing').length}
                  </span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-slate-600 dark:text-slate-400">Findings</span>
                  <span className="font-medium text-red-600">
                    {slides.reduce((sum, s) => sum + s.findings.length, 0)}
                  </span>
                </div>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
