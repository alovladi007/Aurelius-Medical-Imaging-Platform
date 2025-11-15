"use client";

import { useState } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import {
  Activity,
  Brain,
  FileText,
  TrendingUp,
  Upload,
  History,
  Settings,
  AlertCircle
} from 'lucide-react';

export default function CancerAIDashboard() {
  const [stats] = useState({
    totalPredictions: 0,
    recentPredictions: 0,
    avgConfidence: 0,
    avgInferenceTime: 0
  });

  const features = [
    {
      title: 'New Prediction',
      description: 'Upload medical images for cancer detection analysis',
      icon: Upload,
      href: '/cancer-ai/predict',
      color: 'text-blue-600'
    },
    {
      title: 'Batch Processing',
      description: 'Analyze multiple images simultaneously',
      icon: FileText,
      href: '/cancer-ai/batch',
      color: 'text-green-600'
    },
    {
      title: 'Prediction History',
      description: 'View and manage past predictions',
      icon: History,
      href: '/cancer-ai/history',
      color: 'text-purple-600'
    },
    {
      title: 'Analytics',
      description: 'Performance metrics and trends',
      icon: TrendingUp,
      href: '/cancer-ai/analytics',
      color: 'text-orange-600'
    },
    {
      title: 'Model Information',
      description: 'View AI model details and capabilities',
      icon: Brain,
      href: '/cancer-ai/model-info',
      color: 'text-pink-600'
    },
    {
      title: 'Settings',
      description: 'Configure preferences and thresholds',
      icon: Settings,
      href: '/cancer-ai/settings',
      color: 'text-gray-600'
    }
  ];

  const cancerTypes = [
    { name: 'No Cancer', count: 0, color: 'bg-green-500' },
    { name: 'Lung Cancer', count: 0, color: 'bg-red-500' },
    { name: 'Breast Cancer', count: 0, color: 'bg-pink-500' },
    { name: 'Prostate Cancer', count: 0, color: 'bg-blue-500' },
    { name: 'Colorectal Cancer', count: 0, color: 'bg-yellow-500' }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold mb-2">Advanced Cancer AI</h1>
        <p className="text-gray-600 dark:text-gray-400">
          State-of-the-art multimodal cancer detection system
        </p>
      </div>

      {/* Medical Disclaimer */}
      <Card className="border-orange-200 bg-orange-50 dark:bg-orange-950">
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-orange-600" />
            <CardTitle className="text-orange-900 dark:text-orange-100">
              Medical Disclaimer
            </CardTitle>
          </div>
        </CardHeader>
        <CardContent className="text-sm text-orange-800 dark:text-orange-200">
          This system is for research and educational purposes only. It is NOT approved for clinical
          diagnosis or treatment decisions. Always consult qualified healthcare professionals for
          medical decisions.
        </CardContent>
      </Card>

      {/* Statistics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Total Predictions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.totalPredictions}</div>
            <p className="text-xs text-gray-500 mt-1">All time</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Recent Predictions
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.recentPredictions}</div>
            <p className="text-xs text-gray-500 mt-1">Last 24 hours</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Avg Confidence
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">
              {(stats.avgConfidence * 100).toFixed(1)}%
            </div>
            <p className="text-xs text-gray-500 mt-1">Model confidence</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-gray-600">
              Avg Inference Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-bold">{stats.avgInferenceTime}ms</div>
            <p className="text-xs text-gray-500 mt-1">Processing speed</p>
          </CardContent>
        </Card>
      </div>

      {/* Feature Cards */}
      <div>
        <h2 className="text-2xl font-semibold mb-4">Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature) => {
            const Icon = feature.icon;
            return (
              <Link key={feature.title} href={feature.href}>
                <Card className="hover:shadow-lg transition-shadow cursor-pointer h-full">
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div className={`p-2 rounded-lg bg-gray-100 dark:bg-gray-800 ${feature.color}`}>
                        <Icon className="h-6 w-6" />
                      </div>
                      <CardTitle className="text-lg">{feature.title}</CardTitle>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <CardDescription>{feature.description}</CardDescription>
                  </CardContent>
                </Card>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Cancer Types Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Predictions by Cancer Type</CardTitle>
          <CardDescription>Distribution of cancer type predictions</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {cancerTypes.map((type) => (
              <div key={type.name}>
                <div className="flex justify-between text-sm mb-1">
                  <span>{type.name}</span>
                  <span className="font-semibold">{type.count}</span>
                </div>
                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div
                    className={`${type.color} h-2 rounded-full`}
                    style={{ width: '0%' }}
                  />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Supported Cancer Types Info */}
      <Card>
        <CardHeader>
          <CardTitle>Supported Cancer Types</CardTitle>
          <CardDescription>The AI model can detect the following cancer types</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="flex items-start gap-3">
              <div className="mt-1">
                <Activity className="h-5 w-5 text-red-500" />
              </div>
              <div>
                <h4 className="font-semibold">Lung Cancer</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Detection from CT scans and X-rays
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="mt-1">
                <Activity className="h-5 w-5 text-pink-500" />
              </div>
              <div>
                <h4 className="font-semibold">Breast Cancer</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Analysis of mammography and ultrasound
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="mt-1">
                <Activity className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <h4 className="font-semibold">Prostate Cancer</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  MRI and ultrasound image analysis
                </p>
              </div>
            </div>
            <div className="flex items-start gap-3">
              <div className="mt-1">
                <Activity className="h-5 w-5 text-yellow-500" />
              </div>
              <div>
                <h4 className="font-semibold">Colorectal Cancer</h4>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  CT colonography and endoscopy images
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Quick Start Guide */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Start Guide</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-4">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold">
              1
            </div>
            <div>
              <h4 className="font-semibold mb-1">Upload Medical Image</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Support DICOM, PNG, JPG, and other medical imaging formats
              </p>
            </div>
          </div>
          <div className="flex gap-4">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold">
              2
            </div>
            <div>
              <h4 className="font-semibold mb-1">Add Clinical Information</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Provide patient age, gender, smoking history, and family history for improved accuracy
              </p>
            </div>
          </div>
          <div className="flex gap-4">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold">
              3
            </div>
            <div>
              <h4 className="font-semibold mb-1">Get AI Analysis</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Receive cancer type prediction, confidence score, and clinical recommendations
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* CTA */}
      <div className="flex justify-center">
        <Link href="/cancer-ai/predict">
          <Button size="lg" className="gap-2">
            <Upload className="h-5 w-5" />
            Start New Prediction
          </Button>
        </Link>
      </div>
    </div>
  );
}
