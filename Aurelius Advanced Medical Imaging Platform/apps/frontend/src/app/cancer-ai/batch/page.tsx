'use client'

import { useState } from 'react'
import { Upload, FileText, Brain, CheckCircle, XCircle, Clock } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

export default function BatchPredictionPage() {
  const [files, setFiles] = useState<File[]>([])
  const [processing, setProcessing] = useState(false)

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      setFiles(Array.from(e.target.files))
    }
  }

  const handleBatchProcess = async () => {
    setProcessing(true)
    // TODO: Implement batch processing with real API
    setTimeout(() => setProcessing(false), 3000)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            Batch Cancer Prediction
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Process multiple medical images for cancer detection simultaneously
          </p>
        </div>

        {/* Upload Section */}
        <Card className="border-2 border-dashed border-slate-300 dark:border-slate-700">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Upload className="w-5 h-5 text-blue-600" />
              Upload Images for Batch Processing
            </CardTitle>
            <CardDescription>
              Select multiple DICOM files, PNG, or JPG images to process
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <input
                type="file"
                multiple
                accept=".dcm,.png,.jpg,.jpeg"
                onChange={handleFileUpload}
                className="block w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
              />

              {files.length > 0 && (
                <div className="mt-4">
                  <p className="text-sm font-medium mb-2">Selected Files ({files.length}):</p>
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {files.map((file, idx) => (
                      <div key={idx} className="flex items-center justify-between p-2 bg-slate-100 dark:bg-slate-800 rounded">
                        <div className="flex items-center gap-2">
                          <FileText className="w-4 h-4 text-blue-600" />
                          <span className="text-sm">{file.name}</span>
                        </div>
                        <span className="text-xs text-slate-500">{(file.size / 1024).toFixed(1)} KB</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <Button
                onClick={handleBatchProcess}
                disabled={files.length === 0 || processing}
                className="w-full"
              >
                {processing ? (
                  <>
                    <Clock className="w-4 h-4 mr-2 animate-spin" />
                    Processing {files.length} images...
                  </>
                ) : (
                  <>
                    <Brain className="w-4 h-4 mr-2" />
                    Start Batch Processing
                  </>
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle>Batch Processing Results</CardTitle>
            <CardDescription>Real-time results will appear here as processing completes</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {processing ? (
                <div className="text-center py-12">
                  <Brain className="w-12 h-12 mx-auto mb-4 text-blue-600 animate-pulse" />
                  <p className="text-slate-600 dark:text-slate-400">
                    Processing images with advanced cancer detection AI...
                  </p>
                </div>
              ) : files.length === 0 ? (
                <div className="text-center py-12 text-slate-500">
                  <Upload className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Upload images to start batch processing</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {files.map((file, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 border rounded-lg">
                      <div className="flex items-center gap-3">
                        <FileText className="w-5 h-5 text-blue-600" />
                        <span className="font-medium">{file.name}</span>
                      </div>
                      <Badge variant="secondary">
                        <Clock className="w-3 h-3 mr-1" />
                        Pending
                      </Badge>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Info Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-600">Supported Formats</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold text-blue-600">DICOM, PNG, JPG</p>
              <p className="text-xs text-slate-500 mt-1">Medical imaging standards</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-600">Max Batch Size</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold text-purple-600">100 images</p>
              <p className="text-xs text-slate-500 mt-1">Per processing job</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-medium text-slate-600">Processing Time</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-2xl font-bold text-green-600">~2-5 sec</p>
              <p className="text-xs text-slate-500 mt-1">Per image average</p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
