"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import {
  Upload,
  FileText,
  CheckCircle,
  Clock,
  AlertCircle,
  Download,
  Play,
  XCircle,
  Activity
} from 'lucide-react';

export default function BatchProcessingPage() {
  const [files, setFiles] = useState<any[]>([]);
  const [processing, setProcessing] = useState(false);
  const [results, setResults] = useState<any[]>([]);

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFiles = Array.from(event.target.files || []);
    const fileData = uploadedFiles.map((file, index) => ({
      id: index + 1,
      name: file.name,
      size: file.size,
      status: 'pending',
      file: file
    }));
    setFiles([...files, ...fileData]);
  };

  const startBatchProcessing = async () => {
    setProcessing(true);

    // Simulate processing
    for (let i = 0; i < files.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 1000));
      setFiles(prev => prev.map((f, idx) =>
        idx === i ? { ...f, status: 'processing' } : f
      ));

      await new Promise(resolve => setTimeout(resolve, 2000));
      const result = {
        ...files[i],
        status: 'completed',
        prediction: ['Lung Cancer', 'Breast Cancer', 'No Cancer'][Math.floor(Math.random() * 3)],
        confidence: (Math.random() * 30 + 70).toFixed(1),
        completedAt: new Date().toISOString()
      };

      setFiles(prev => prev.map((f, idx) =>
        idx === i ? result : f
      ));
      setResults(prev => [...prev, result]);
    }

    setProcessing(false);
  };

  const exportResults = () => {
    const csv = [
      ['Filename', 'Prediction', 'Confidence', 'Status', 'Completed At'],
      ...results.map(r => [r.name, r.prediction, r.confidence + '%', r.status, r.completedAt])
    ].map(row => row.join(',')).join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cancer-ai-batch-results.csv';
    a.click();
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case 'processing':
        return <Activity className="h-4 w-4 text-blue-600 animate-spin" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-600" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  return (
    <div className="p-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Batch Processing</h1>
          <p className="text-gray-600 dark:text-gray-400">Process multiple medical images simultaneously</p>
        </div>
        <div className="flex gap-3">
          {results.length > 0 && (
            <Button variant="outline" onClick={exportResults}>
              <Download className="h-4 w-4 mr-2" />
              Export Results
            </Button>
          )}
          <Button onClick={startBatchProcessing} disabled={processing || files.length === 0}>
            <Play className="h-4 w-4 mr-2" />
            {processing ? 'Processing...' : 'Start Batch'}
          </Button>
        </div>
      </div>

      {/* Upload Section */}
      <Card>
        <CardHeader>
          <CardTitle>Upload Images</CardTitle>
          <CardDescription>Select multiple medical images for batch analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <label className="block">
            <div className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-12 text-center hover:border-primary cursor-pointer transition-colors">
              <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
              <p className="text-gray-600 dark:text-gray-400 mb-2">Click to upload or drag and drop</p>
              <p className="text-sm text-gray-500">DICOM, PNG, JPG, TIFF (max 10MB each)</p>
              <p className="text-sm text-gray-500 mt-2">{files.length} files selected</p>
            </div>
            <input
              type="file"
              multiple
              accept="image/*,.dcm"
              className="hidden"
              onChange={handleFileUpload}
              disabled={processing}
            />
          </label>
        </CardContent>
      </Card>

      {/* Processing Queue */}
      {files.length > 0 && (
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Processing Queue ({files.length} files)</CardTitle>
              <Badge variant={processing ? "default" : "secondary"}>
                {processing ? 'Processing' : 'Ready'}
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {files.map((file, index) => (
                <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-4 flex-1">
                    <div className="flex items-center justify-center w-10 h-10">
                      {getStatusIcon(file.status)}
                    </div>
                    <div className="flex-1">
                      <p className="font-medium">{file.name}</p>
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        {(file.size / 1024).toFixed(2)} KB
                      </p>
                    </div>
                  </div>
                  {file.status === 'completed' && (
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <p className="font-semibold text-primary">{file.prediction}</p>
                        <p className="text-sm text-gray-600">{file.confidence}% confidence</p>
                      </div>
                      <Badge className="bg-green-600">Complete</Badge>
                    </div>
                  )}
                  {file.status === 'processing' && (
                    <Badge className="bg-blue-600">Processing...</Badge>
                  )}
                  {file.status === 'pending' && (
                    <Badge variant="outline">Pending</Badge>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results Summary */}
      {results.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Results Summary</CardTitle>
            <CardDescription>{results.length} images processed</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                <p className="text-sm text-gray-600 dark:text-gray-400">Completed</p>
                <p className="text-3xl font-bold text-green-600">
                  {results.filter(r => r.status === 'completed').length}
                </p>
              </div>
              <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                <p className="text-sm text-gray-600 dark:text-gray-400">Avg Confidence</p>
                <p className="text-3xl font-bold text-blue-600">
                  {results.length > 0
                    ? (results.reduce((acc, r) => acc + parseFloat(r.confidence), 0) / results.length).toFixed(1)
                    : 0}%
                </p>
              </div>
              <div className="p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                <p className="text-sm text-gray-600 dark:text-gray-400">Cancer Detected</p>
                <p className="text-3xl font-bold text-purple-600">
                  {results.filter(r => r.prediction !== 'No Cancer').length}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Instructions */}
      <Card>
        <CardHeader>
          <CardTitle>Batch Processing Instructions</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm">
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-xs font-bold flex-shrink-0">1</div>
            <p>Upload multiple medical images (DICOM, PNG, JPG, or TIFF format)</p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-xs font-bold flex-shrink-0">2</div>
            <p>Review the processing queue to ensure all files are loaded correctly</p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-xs font-bold flex-shrink-0">3</div>
            <p>Click "Start Batch" to begin processing all images sequentially</p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-xs font-bold flex-shrink-0">4</div>
            <p>Monitor progress in real-time as each image is analyzed</p>
          </div>
          <div className="flex items-start gap-3">
            <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-xs font-bold flex-shrink-0">5</div>
            <p>Export results as CSV for further analysis or record-keeping</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
