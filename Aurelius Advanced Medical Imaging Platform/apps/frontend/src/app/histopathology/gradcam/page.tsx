"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Upload, Eye, Download, Layers } from 'lucide-react';

export default function GradCAMPage() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [gradcamImage, setGradcamImage] = useState(null);
  const [layer, setLayer] = useState('layer4');
  const [loading, setLoading] = useState(false);

  const generateGradCAM = async () => {
    if (!selectedImage) return;

    setLoading(true);
    try {
      const response = await fetch('/api/histopathology/gradcam', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ imagePath: selectedImage, layer })
      });

      if (response.ok) {
        const data = await response.json();
        setGradcamImage(data.gradcamUrl);
      }
    } catch (error) {
      console.error('Grad-CAM error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
      setGradcamImage(null);
    }
  };

  return (
    <div className="p-8 space-y-8">
      <div>
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <Eye className="h-8 w-8 text-primary" />
          Grad-CAM Visualization
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Visualize what your models are learning with Class Activation Maps
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Configuration */}
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Target Layer</label>
              <Select value={layer} onValueChange={setLayer}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="layer1">Layer 1</SelectItem>
                  <SelectItem value="layer2">Layer 2</SelectItem>
                  <SelectItem value="layer3">Layer 3</SelectItem>
                  <SelectItem value="layer4">Layer 4 (Default)</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <label className="block">
              <Button className="w-full">
                <Upload className="h-4 w-4 mr-2" />
                Upload Image
              </Button>
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleImageUpload}
              />
            </label>

            <Button
              onClick={generateGradCAM}
              disabled={!selectedImage || loading}
              className="w-full"
            >
              <Layers className="h-4 w-4 mr-2" />
              {loading ? 'Generating...' : 'Generate Grad-CAM'}
            </Button>
          </CardContent>
        </Card>

        {/* Original Image */}
        <Card>
          <CardHeader>
            <CardTitle>Original Image</CardTitle>
          </CardHeader>
          <CardContent>
            {selectedImage ? (
              <div className="relative h-96 bg-gray-100 dark:bg-gray-900 rounded-lg overflow-hidden">
                <img src={selectedImage} alt="Original" className="w-full h-full object-contain" />
              </div>
            ) : (
              <div className="h-96 flex items-center justify-center bg-gray-100 dark:bg-gray-900 rounded-lg">
                <p className="text-gray-500">No image uploaded</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Grad-CAM Result */}
        <Card>
          <CardHeader>
            <CardTitle>Grad-CAM Overlay</CardTitle>
          </CardHeader>
          <CardContent>
            {gradcamImage ? (
              <div>
                <div className="relative h-96 bg-gray-100 dark:bg-gray-900 rounded-lg overflow-hidden mb-4">
                  <img src={gradcamImage} alt="Grad-CAM" className="w-full h-full object-contain" />
                </div>
                <Button className="w-full">
                  <Download className="h-4 w-4 mr-2" />
                  Download Visualization
                </Button>
              </div>
            ) : (
              <div className="h-96 flex items-center justify-center bg-gray-100 dark:bg-gray-900 rounded-lg">
                <p className="text-gray-500">Generate Grad-CAM to see visualization</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Legend */}
      <Card>
        <CardHeader>
          <CardTitle>Understanding Grad-CAM</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <p><span className="inline-block w-4 h-4 bg-red-600 rounded mr-2"></span><strong>Red areas:</strong> High importance - features the model focuses on most</p>
          <p><span className="inline-block w-4 h-4 bg-yellow-600 rounded mr-2"></span><strong>Yellow areas:</strong> Medium importance</p>
          <p><span className="inline-block w-4 h-4 bg-blue-600 rounded mr-2"></span><strong>Blue areas:</strong> Low importance - less relevant features</p>
          <p className="mt-4 text-gray-600 dark:text-gray-400">
            Grad-CAM helps visualize which regions of the image the model considers important for its prediction.
            This is crucial for understanding model behavior and building trust in AI-assisted diagnosis.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
