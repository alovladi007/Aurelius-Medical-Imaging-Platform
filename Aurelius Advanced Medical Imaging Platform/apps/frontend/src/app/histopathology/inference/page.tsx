"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Upload, Brain, Image as ImageIcon, FileText, Download } from 'lucide-react';
import Image from 'next/image';

export default function InferencePage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setSelectedImage(URL.createObjectURL(file));
    setLoading(true);

    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch('/api/histopathology/inference', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        const data = await response.json();
        setPrediction(data);
      } else {
        alert('Error running inference');
      }
    } catch (error) {
      console.error('Inference error:', error);
      alert('Error running inference');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Brain className="h-8 w-8 text-primary" />
          Model Inference
        </h1>
        <p className="text-gray-600 dark:text-gray-400">Run predictions on histopathology images</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Upload Section */}
        <Card>
          <CardHeader>
            <CardTitle>Upload Image</CardTitle>
            <CardDescription>Upload a histopathology image for analysis</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <label className="block">
              <div className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-12 text-center hover:border-primary cursor-pointer transition-colors">
                <Upload className="h-12 w-12 mx-auto mb-4 text-gray-400" />
                <p className="text-gray-600 dark:text-gray-400 mb-2">Click to upload or drag and drop</p>
                <p className="text-sm text-gray-500">PNG, JPG, TIFF up to 10MB</p>
              </div>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageUpload}
              />
            </label>

            {selectedImage && (
              <div className="relative h-64 bg-gray-100 dark:bg-gray-900 rounded-lg overflow-hidden">
                <img
                  src={selectedImage}
                  alt="Uploaded"
                  className="w-full h-full object-contain"
                />
              </div>
            )}
          </CardContent>
        </Card>

        {/* Results Section */}
        <Card>
          <CardHeader>
            <CardTitle>Prediction Results</CardTitle>
            <CardDescription>Model predictions and confidence scores</CardDescription>
          </CardHeader>
          <CardContent>
            {loading ? (
              <div className="text-center py-12">
                <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
                <p className="text-gray-600">Running inference...</p>
              </div>
            ) : prediction ? (
              <div className="space-y-6">
                <div className="text-center p-6 bg-primary/10 rounded-lg">
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Predicted Class</p>
                  <p className="text-3xl font-bold text-primary">{prediction.predictedClass}</p>
                  <p className="text-lg text-gray-600 mt-2">{(prediction.confidence * 100).toFixed(2)}% confidence</p>
                </div>

                <div>
                  <h4 className="font-semibold mb-3">Class Probabilities</h4>
                  <div className="space-y-2">
                    {Object.entries(prediction.probabilities || {}).map(([className, prob]: any) => (
                      <div key={className}>
                        <div className="flex justify-between text-sm mb-1">
                          <span>{className}</span>
                          <span className="font-mono">{(prob * 100).toFixed(2)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                          <div className="bg-primary h-2 rounded-full" style={{ width: `${prob * 100}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex gap-3">
                  <Button className="flex-1">
                    <Download className="h-4 w-4 mr-2" />
                    Download Results
                  </Button>
                  <Button variant="outline" className="flex-1" onClick={() => window.location.href = '/histopathology/gradcam'}>
                    View Grad-CAM
                  </Button>
                </div>
              </div>
            ) : (
              <div className="text-center py-12">
                <Brain className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 dark:text-gray-400">Upload an image to get predictions</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Batch Inference */}
      <Card>
        <CardHeader>
          <CardTitle>Batch Inference</CardTitle>
          <CardDescription>Process multiple images at once</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-4">
            <label>
              <Button>
                <FileText className="h-4 w-4 mr-2" />
                Upload CSV with Image Paths
              </Button>
              <input type="file" accept=".csv" className="hidden" />
            </label>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Or upload a directory of images for bulk processing
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
