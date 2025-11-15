"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  BarChart3,
  TrendingUp,
  PieChart,
  Activity,
  Calendar,
  Download,
  AlertCircle,
  CheckCircle,
  Target,
  Zap
} from 'lucide-react';

export default function AnalyticsPage() {
  const [timeRange, setTimeRange] = useState('30d');

  // Performance metrics
  const performanceMetrics = {
    overall: {
      accuracy: 94.2,
      precision: 96.8,
      recall: 93.5,
      f1Score: 95.1,
      auroc: 0.97
    },
    byType: [
      { type: 'Lung Cancer', accuracy: 95.3, samples: 1247 },
      { type: 'Breast Cancer', accuracy: 96.1, samples: 1893 },
      { type: 'Colorectal Cancer', accuracy: 92.8, samples: 876 },
      { type: 'Prostate Cancer', accuracy: 94.5, samples: 1034 },
      { type: 'No Cancer', accuracy: 93.7, samples: 2156 }
    ],
    byModality: [
      { modality: 'CT', accuracy: 94.8, volume: 3421 },
      { modality: 'MRI', accuracy: 95.2, volume: 1876 },
      { modality: 'X-Ray', accuracy: 92.3, volume: 2134 },
      { modality: 'Mammography', accuracy: 96.5, volume: 1893 }
    ]
  };

  // Temporal trends
  const monthlyStats = [
    { month: 'Jun', predictions: 856, accuracy: 93.2, avgConfidence: 91.5 },
    { month: 'Jul', predictions: 923, accuracy: 93.8, avgConfidence: 92.1 },
    { month: 'Aug', predictions: 1045, accuracy: 94.1, avgConfidence: 92.8 },
    { month: 'Sep', predictions: 1183, accuracy: 94.5, avgConfidence: 93.2 },
    { month: 'Oct', predictions: 1267, accuracy: 94.2, avgConfidence: 93.5 },
    { month: 'Nov', predictions: 1432, accuracy: 94.7, avgConfidence: 94.1 }
  ];

  // Cancer distribution
  const cancerDistribution = [
    { type: 'Lung Cancer', count: 1247, percentage: 17.2, color: 'bg-red-500' },
    { type: 'Breast Cancer', count: 1893, percentage: 26.1, color: 'bg-pink-500' },
    { type: 'Colorectal Cancer', count: 876, percentage: 12.1, color: 'bg-orange-500' },
    { type: 'Prostate Cancer', count: 1034, percentage: 14.3, color: 'bg-blue-500' },
    { type: 'No Cancer', count: 2156, percentage: 29.8, color: 'bg-green-500' },
    { type: 'Other', count: 100, percentage: 1.4, color: 'bg-gray-500' }
  ];

  // Confidence distribution
  const confidenceRanges = [
    { range: '90-100%', count: 5892, percentage: 81.3, color: 'bg-green-600' },
    { range: '80-89%', count: 987, percentage: 13.6, color: 'bg-yellow-600' },
    { range: '70-79%', count: 287, percentage: 4.0, color: 'bg-orange-600' },
    { range: '< 70%', count: 80, percentage: 1.1, color: 'bg-red-600' }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <BarChart3 className="h-8 w-8 text-primary" />
            Analytics & Insights
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Performance metrics and trends for cancer AI predictions
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Calendar className="h-4 w-4 mr-2" />
            {timeRange === '7d' ? 'Last 7 Days' : timeRange === '30d' ? 'Last 30 Days' : 'Last 90 Days'}
          </Button>
          <Button>
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
        </div>
      </div>

      {/* Overall Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Overall Accuracy</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-primary">{performanceMetrics.overall.accuracy}%</p>
            <div className="flex items-center gap-1 mt-2">
              <TrendingUp className="h-3 w-3 text-green-600" />
              <p className="text-xs text-green-600">+2.3% vs last month</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Precision</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-blue-600">{performanceMetrics.overall.precision}%</p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">True positives rate</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Recall</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-green-600">{performanceMetrics.overall.recall}%</p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">Sensitivity</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">F1-Score</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-purple-600">{performanceMetrics.overall.f1Score}%</p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">Harmonic mean</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">AUROC</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-orange-600">{performanceMetrics.overall.auroc}</p>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">Area under curve</p>
          </CardContent>
        </Card>
      </div>

      {/* Monthly Trends */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Trends</CardTitle>
          <CardDescription>Monthly prediction volume and accuracy metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            {/* Predictions Volume Trend */}
            <div>
              <p className="text-sm font-semibold mb-3">Prediction Volume</p>
              <div className="flex items-end justify-between gap-2 h-48">
                {monthlyStats.map((stat, index) => (
                  <div key={index} className="flex-1 flex flex-col items-center">
                    <div className="flex-1 flex items-end w-full">
                      <div
                        className="w-full bg-primary rounded-t transition-all hover:opacity-80"
                        style={{ height: `${(stat.predictions / 1500) * 100}%` }}
                      />
                    </div>
                    <p className="text-xs font-semibold mt-2">{stat.month}</p>
                    <p className="text-xs text-gray-600">{stat.predictions}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Accuracy Trend */}
            <div>
              <p className="text-sm font-semibold mb-3">Accuracy Over Time</p>
              <div className="relative h-32">
                <div className="absolute inset-0 flex items-end">
                  {monthlyStats.map((stat, index) => (
                    <div key={index} className="flex-1 flex flex-col items-center justify-end">
                      <div
                        className="w-full bg-green-500 rounded-t"
                        style={{ height: `${stat.accuracy}%` }}
                      />
                      <p className="text-xs mt-1">{stat.accuracy}%</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Average Confidence Trend */}
            <div>
              <p className="text-sm font-semibold mb-3">Average Confidence</p>
              <div className="flex items-center gap-2">
                {monthlyStats.map((stat, index) => (
                  <div key={index} className="flex-1">
                    <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                      <div
                        className="bg-blue-600 h-2 rounded-full"
                        style={{ width: `${stat.avgConfidence}%` }}
                      />
                    </div>
                    <p className="text-xs text-center mt-1">{stat.month}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Performance by Cancer Type and Modality */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* By Cancer Type */}
        <Card>
          <CardHeader>
            <CardTitle>Performance by Cancer Type</CardTitle>
            <CardDescription>Accuracy across different cancer types</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {performanceMetrics.byType.map((item, index) => (
                <div key={index}>
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <p className="text-sm font-semibold">{item.type}</p>
                      <p className="text-xs text-gray-600">{item.samples} samples</p>
                    </div>
                    <Badge variant="outline">{item.accuracy}%</Badge>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                    <div
                      className="bg-primary h-2 rounded-full"
                      style={{ width: `${item.accuracy}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* By Modality */}
        <Card>
          <CardHeader>
            <CardTitle>Performance by Modality</CardTitle>
            <CardDescription>Accuracy across imaging modalities</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {performanceMetrics.byModality.map((item, index) => (
                <div key={index}>
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <p className="text-sm font-semibold">{item.modality}</p>
                      <p className="text-xs text-gray-600">{item.volume} images</p>
                    </div>
                    <Badge variant="outline">{item.accuracy}%</Badge>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                    <div
                      className="bg-green-600 h-2 rounded-full"
                      style={{ width: `${item.accuracy}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Cancer Distribution and Confidence */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Cancer Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Cancer Type Distribution</CardTitle>
            <CardDescription>Breakdown of predictions by cancer type</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {cancerDistribution.map((item, index) => (
                <div key={index} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${item.color}`} />
                      <p className="text-sm font-medium">{item.type}</p>
                    </div>
                    <div className="text-right">
                      <p className="text-sm font-bold">{item.percentage}%</p>
                      <p className="text-xs text-gray-600">{item.count} cases</p>
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                    <div
                      className={`${item.color} h-2 rounded-full`}
                      style={{ width: `${item.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Confidence Distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Confidence Distribution</CardTitle>
            <CardDescription>Prediction confidence score ranges</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {confidenceRanges.map((item, index) => (
                <div key={index}>
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <p className="text-sm font-semibold">{item.range}</p>
                      <p className="text-xs text-gray-600">{item.count} predictions</p>
                    </div>
                    <div className="text-right">
                      <Badge variant="outline">{item.percentage}%</Badge>
                    </div>
                  </div>
                  <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-3">
                    <div
                      className={`${item.color} h-3 rounded-full`}
                      style={{ width: `${item.percentage * 5}%` }}
                    />
                  </div>
                </div>
              ))}
              <div className="mt-4 p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                <div className="flex items-start gap-2">
                  <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-sm">High Confidence Rate</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      81.3% of all predictions have confidence scores above 90%, indicating strong model reliability
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Model Quality Indicators */}
      <Card>
        <CardHeader>
          <CardTitle>Model Quality Indicators</CardTitle>
          <CardDescription>Key performance and reliability metrics</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 border rounded-lg">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 bg-green-100 dark:bg-green-950 rounded-full flex items-center justify-center">
                  <Target className="h-5 w-5 text-green-600" />
                </div>
                <div>
                  <p className="font-semibold">Calibration Score</p>
                  <p className="text-2xl font-bold text-green-600">0.95</p>
                </div>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Model predictions are well-calibrated with actual outcomes
              </p>
            </div>

            <div className="p-4 border rounded-lg">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 bg-blue-100 dark:bg-blue-950 rounded-full flex items-center justify-center">
                  <Activity className="h-5 w-5 text-blue-600" />
                </div>
                <div>
                  <p className="font-semibold">Stability Index</p>
                  <p className="text-2xl font-bold text-blue-600">98.2%</p>
                </div>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Consistent performance across different time periods
              </p>
            </div>

            <div className="p-4 border rounded-lg">
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 bg-purple-100 dark:bg-purple-950 rounded-full flex items-center justify-center">
                  <Zap className="h-5 w-5 text-purple-600" />
                </div>
                <div>
                  <p className="font-semibold">Avg Response Time</p>
                  <p className="text-2xl font-bold text-purple-600">1.2s</p>
                </div>
              </div>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Fast inference suitable for clinical workflows
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Insights and Recommendations */}
      <Card className="border-blue-200 dark:border-blue-900">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <AlertCircle className="h-5 w-5 text-blue-600" />
            Key Insights & Recommendations
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 text-sm">
            <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg">
              <p className="font-semibold text-green-900 dark:text-green-100 mb-1">
                ✓ Strong Performance Across All Cancer Types
              </p>
              <p className="text-green-800 dark:text-green-200">
                All cancer types show accuracy above 92%, with breast cancer detection achieving 96.1% accuracy
              </p>
            </div>
            <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
              <p className="font-semibold text-blue-900 dark:text-blue-100 mb-1">
                ✓ Mammography Excels
              </p>
              <p className="text-blue-800 dark:text-blue-200">
                Mammography modality shows highest accuracy at 96.5%, indicating excellent performance for breast cancer screening
              </p>
            </div>
            <div className="p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg">
              <p className="font-semibold text-yellow-900 dark:text-yellow-100 mb-1">
                ⚠ X-Ray Performance
              </p>
              <p className="text-yellow-800 dark:text-yellow-200">
                X-Ray modality shows slightly lower accuracy (92.3%). Consider additional training data or model fine-tuning
              </p>
            </div>
            <div className="p-3 bg-purple-50 dark:bg-purple-950 rounded-lg">
              <p className="font-semibold text-purple-900 dark:text-purple-100 mb-1">
                ↗ Positive Trend
              </p>
              <p className="text-purple-800 dark:text-purple-200">
                Prediction volume and accuracy both trending upward over the past 6 months, showing continuous improvement
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
