'use client'

import { useEffect, useRef, useState } from 'react'
import {
  ZoomIn,
  ZoomOut,
  RotateCw,
  Move,
  Ruler,
  Download,
  Maximize,
  Settings
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/card'

interface DicomViewerProps {
  studyId: string
  seriesId?: string
}

export default function DicomViewer({ studyId, seriesId }: DicomViewerProps) {
  const viewportRef = useRef<HTMLDivElement>(null)
  const [activeTool, setActiveTool] = useState<string>('pan')
  const [zoom, setZoom] = useState(1)
  const [brightness, setBrightness] = useState(50)
  const [contrast, setContrast] = useState(50)

  useEffect(() => {
    // Initialize Cornerstone.js viewer
    if (viewportRef.current) {
      initializeViewer()
    }
  }, [studyId, seriesId])

  const initializeViewer = () => {
    // Cornerstone initialization would go here
    console.log('Initializing DICOM viewer for study:', studyId)
  }

  const tools = [
    { name: 'pan', label: 'Pan', icon: Move },
    { name: 'zoom', label: 'Zoom', icon: ZoomIn },
    { name: 'ruler', label: 'Measure', icon: Ruler },
    { name: 'rotate', label: 'Rotate', icon: RotateCw },
  ]

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="bg-slate-800 p-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {tools.map((tool) => (
            <Button
              key={tool.name}
              variant={activeTool === tool.name ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveTool(tool.name)}
              className="text-white"
            >
              <tool.icon className="h-4 w-4 mr-2" />
              {tool.label}
            </Button>
          ))}
        </div>
        <div className="flex items-center space-x-2">
          <Button variant="ghost" size="sm" className="text-white">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button variant="ghost" size="sm" className="text-white">
            <Maximize className="h-4 w-4 mr-2" />
            Fullscreen
          </Button>
          <Button variant="ghost" size="sm" className="text-white">
            <Settings className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* Viewer Area */}
      <div className="flex flex-1">
        {/* Main Viewport */}
        <div className="flex-1 bg-black relative">
          <div
            ref={viewportRef}
            className="w-full h-full flex items-center justify-center"
          >
            {/* Placeholder for DICOM image */}
            <div className="text-center text-slate-400">
              <div className="w-64 h-64 border-4 border-dashed border-slate-700 rounded-lg flex items-center justify-center mb-4 mx-auto">
                <div>
                  <div className="text-6xl mb-4">🏥</div>
                  <p className="text-sm">DICOM Image Viewer</p>
                  <p className="text-xs mt-2">Study: {studyId}</p>
                </div>
              </div>
              <p className="text-xs">
                This is a placeholder. Full DICOM rendering requires Cornerstone.js integration.
              </p>
            </div>
          </div>

          {/* Overlay Info */}
          <div className="absolute top-4 left-4 text-white text-sm space-y-1 bg-black/50 p-3 rounded">
            <div>Patient: John Doe</div>
            <div>Study Date: 2024-10-27</div>
            <div>Modality: CT</div>
            <div>Series: 1/12</div>
            <div>Image: 1/128</div>
          </div>

          {/* Zoom Controls */}
          <div className="absolute bottom-4 left-4 flex items-center space-x-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setZoom(Math.max(0.1, zoom - 0.1))}
            >
              <ZoomOut className="h-4 w-4" />
            </Button>
            <span className="text-white text-sm bg-black/50 px-3 py-1 rounded">
              {Math.round(zoom * 100)}%
            </span>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setZoom(Math.min(5, zoom + 0.1))}
            >
              <ZoomIn className="h-4 w-4" />
            </Button>
          </div>
        </div>

        {/* Side Panel */}
        <div className="w-80 bg-slate-50 dark:bg-slate-900 p-4 overflow-y-auto">
          <Card className="p-4 mb-4">
            <h3 className="font-semibold mb-3">Image Adjustments</h3>
            <div className="space-y-4">
              <div>
                <label className="text-sm text-slate-600 dark:text-slate-400">
                  Brightness
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={brightness}
                  onChange={(e) => setBrightness(Number(e.target.value))}
                  className="w-full"
                />
              </div>
              <div>
                <label className="text-sm text-slate-600 dark:text-slate-400">
                  Contrast
                </label>
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={contrast}
                  onChange={(e) => setContrast(Number(e.target.value))}
                  className="w-full"
                />
              </div>
            </div>
          </Card>

          <Card className="p-4">
            <h3 className="font-semibold mb-3">Series Thumbnails</h3>
            <div className="grid grid-cols-2 gap-2">
              {[1, 2, 3, 4, 5, 6].map((i) => (
                <div
                  key={i}
                  className="aspect-square bg-slate-200 dark:bg-slate-800 rounded cursor-pointer hover:ring-2 hover:ring-primary transition-all"
                >
                  <div className="flex items-center justify-center h-full text-xs text-slate-600">
                    Series {i}
                  </div>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </div>
  )
}
