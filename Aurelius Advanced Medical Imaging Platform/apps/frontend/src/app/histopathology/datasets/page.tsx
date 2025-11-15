"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Upload,
  FolderOpen,
  Database,
  Image,
  CheckCircle,
  XCircle,
  AlertCircle,
  Download,
  Trash2,
  RefreshCw,
  Plus
} from 'lucide-react';

export default function DatasetsPage() {
  const [datasets, setDatasets] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [selectedDataset, setSelectedDataset] = useState(null);

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const response = await fetch('/api/histopathology/datasets');
      if (response.ok) {
        const data = await response.json();
        setDatasets(data.datasets);
      } else {
        // Mock data
        setDatasets([
          {
            id: 1,
            name: 'Brain Cancer MRI',
            path: '/data/raw/brain_cancer',
            numClasses: 4,
            totalImages: 7023,
            status: 'ready',
            createdAt: '2025-11-10'
          },
          {
            id: 2,
            name: 'Lung Histopathology',
            path: '/data/raw/lung',
            numClasses: 2,
            totalImages: 1248,
            status: 'processing',
            createdAt: '2025-11-14'
          },
          {
            id: 3,
            name: 'Synthetic Test Data',
            path: '/data/raw/synthetic',
            numClasses: 2,
            totalImages: 400,
            status: 'ready',
            createdAt: '2025-11-15'
          }
        ]);
      }
    } catch (error) {
      console.error('Error loading datasets:', error);
    }
  };

  const handleUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    setUploading(true);
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    try {
      const response = await fetch('/api/histopathology/datasets/upload', {
        method: 'POST',
        body: formData
      });

      if (response.ok) {
        loadDatasets();
        alert('Dataset uploaded successfully!');
      } else {
        alert('Error uploading dataset');
      }
    } catch (error) {
      console.error('Error uploading:', error);
      alert('Error uploading dataset');
    } finally {
      setUploading(false);
    }
  };

  const generateSyntheticData = async () => {
    try {
      const response = await fetch('/api/histopathology/datasets/generate-synthetic', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          numClasses: 2,
          samplesPerClass: 200
        })
      });

      if (response.ok) {
        alert('Synthetic dataset generated successfully!');
        loadDatasets();
      } else {
        alert('Error generating synthetic data');
      }
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="p-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Dataset Management</h1>
          <p className="text-gray-600 dark:text-gray-400">
            Organize and prepare your histopathology datasets
          </p>
        </div>
        <div className="flex gap-3">
          <Button onClick={generateSyntheticData} variant="outline">
            <Plus className="h-4 w-4 mr-2" />
            Generate Synthetic
          </Button>
          <label>
            <Button disabled={uploading}>
              <Upload className="h-4 w-4 mr-2" />
              {uploading ? 'Uploading...' : 'Upload Dataset'}
            </Button>
            <input
              type="file"
              multiple
              webkitdirectory=""
              directory=""
              className="hidden"
              onChange={handleUpload}
            />
          </label>
        </div>
      </div>

      {/* Datasets Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {datasets.map((dataset: any) => (
          <Card key={dataset.id} className="hover:shadow-lg transition-shadow">
            <CardHeader>
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  <Database className="h-5 w-5 text-primary" />
                  <CardTitle className="text-lg">{dataset.name}</CardTitle>
                </div>
                <Badge variant={dataset.status === 'ready' ? 'default' : 'secondary'}>
                  {dataset.status}
                </Badge>
              </div>
              <CardDescription className="font-mono text-xs">{dataset.path}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Images</p>
                  <p className="text-2xl font-bold">{dataset.totalImages.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Classes</p>
                  <p className="text-2xl font-bold">{dataset.numClasses}</p>
                </div>
              </div>
              <div className="text-xs text-gray-600 dark:text-gray-400">
                Created {dataset.createdAt}
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="flex-1"
                  onClick={() => setSelectedDataset(dataset)}
                >
                  <FolderOpen className="h-3 w-3 mr-1" />
                  View
                </Button>
                <Button variant="outline" size="sm">
                  <Download className="h-3 w-3" />
                </Button>
                <Button variant="outline" size="sm" className="text-red-600">
                  <Trash2 className="h-3 w-3" />
                </Button>
              </div>
            </CardContent>
          </Card>
        ))}

        {/* Empty State */}
        {datasets.length === 0 && (
          <Card className="col-span-full">
            <CardContent className="flex flex-col items-center justify-center py-16">
              <Database className="h-16 w-16 text-gray-400 mb-4" />
              <h3 className="text-xl font-semibold mb-2">No datasets yet</h3>
              <p className="text-gray-600 dark:text-gray-400 mb-6 text-center max-w-md">
                Upload your histopathology images or generate synthetic data to get started
              </p>
              <div className="flex gap-3">
                <Button onClick={generateSyntheticData}>
                  <Plus className="h-4 w-4 mr-2" />
                  Generate Synthetic Data
                </Button>
                <label>
                  <Button variant="outline">
                    <Upload className="h-4 w-4 mr-2" />
                    Upload Dataset
                  </Button>
                  <input
                    type="file"
                    multiple
                    webkitdirectory=""
                    directory=""
                    className="hidden"
                    onChange={handleUpload}
                  />
                </label>
              </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Dataset Details Modal */}
      {selectedDataset && (
        <Card className="border-primary">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>{selectedDataset.name}</CardTitle>
              <Button variant="ghost" size="sm" onClick={() => setSelectedDataset(null)}>
                ✕
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h4 className="font-semibold mb-2">Dataset Information</h4>
              <dl className="grid grid-cols-2 gap-4">
                <div>
                  <dt className="text-sm text-gray-600 dark:text-gray-400">Path</dt>
                  <dd className="font-mono text-sm">{selectedDataset.path}</dd>
                </div>
                <div>
                  <dt className="text-sm text-gray-600 dark:text-gray-400">Status</dt>
                  <dd><Badge>{selectedDataset.status}</Badge></dd>
                </div>
                <div>
                  <dt className="text-sm text-gray-600 dark:text-gray-400">Total Images</dt>
                  <dd className="text-lg font-bold">{selectedDataset.totalImages.toLocaleString()}</dd>
                </div>
                <div>
                  <dt className="text-sm text-gray-600 dark:text-gray-400">Number of Classes</dt>
                  <dd className="text-lg font-bold">{selectedDataset.numClasses}</dd>
                </div>
              </dl>
            </div>
            <div className="flex gap-3 pt-4">
              <Button className="flex-1">
                <RefreshCw className="h-4 w-4 mr-2" />
                Create Splits
              </Button>
              <Button className="flex-1">
                Start Training
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
