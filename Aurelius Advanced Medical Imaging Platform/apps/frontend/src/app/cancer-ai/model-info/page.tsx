'use client'

import { Card } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Brain,
  Activity,
  TrendingUp,
  Database,
  Zap,
  Shield,
  CheckCircle,
  Download,
  RefreshCw,
  BarChart3
} from 'lucide-react'

export default function ModelInfoPage() {
  const models = [
    {
      name: 'Lung Cancer Detection',
      version: 'v2.3.1',
      status: 'active',
      accuracy: 94.3,
      sensitivity: 92.1,
      specificity: 96.5,
      trainingData: '125,000 studies',
      lastUpdated: '2024-01-10',
      architecture: 'ResNet-152 + Attention',
      cancerTypes: ['NSCLC', 'SCLC', 'Metastatic']
    },
    {
      name: 'Breast Cancer Detection',
      version: 'v2.2.5',
      status: 'active',
      accuracy: 91.7,
      sensitivity: 89.3,
      specificity: 94.1,
      trainingData: '98,000 studies',
      lastUpdated: '2024-01-05',
      architecture: 'EfficientNet-B7',
      cancerTypes: ['Ductal', 'Lobular', 'Inflammatory']
    },
    {
      name: 'Brain Tumor Classification',
      version: 'v2.1.8',
      status: 'active',
      accuracy: 92.1,
      sensitivity: 90.5,
      specificity: 93.7,
      trainingData: '87,500 studies',
      lastUpdated: '2023-12-28',
      architecture: 'DenseNet-201',
      cancerTypes: ['Glioblastoma', 'Meningioma', 'Pituitary']
    },
    {
      name: 'Liver Cancer Detection',
      version: 'v2.0.3',
      status: 'active',
      accuracy: 88.9,
      sensitivity: 86.2,
      specificity: 91.6,
      trainingData: '64,000 studies',
      lastUpdated: '2023-12-15',
      architecture: 'VGG-19 + LSTM',
      cancerTypes: ['HCC', 'Cholangiocarcinoma', 'Metastatic']
    }
  ]

  const performanceMetrics = [
    { label: 'Total Models Deployed', value: '12', icon: Brain, color: 'text-purple-600' },
    { label: 'Active Models', value: '4', icon: Activity, color: 'text-green-600' },
    { label: 'Avg. Accuracy', value: '91.8%', icon: TrendingUp, color: 'text-blue-600' },
    { label: 'Total Training Data', value: '374K+', icon: Database, color: 'text-orange-600' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
          AI Model Information
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          Detailed specifications and performance metrics for all deployed models
        </p>
      </div>

      {/* Performance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        {performanceMetrics.map((metric) => (
          <Card key={metric.label} className="p-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
            <div className="flex items-center justify-between mb-2">
              <metric.icon className={`h-8 w-8 ${metric.color}`} />
              <Badge variant="outline" className="bg-slate-50 dark:bg-slate-900">
                Live
              </Badge>
            </div>
            <p className="text-sm text-slate-600 dark:text-slate-400 mb-1">{metric.label}</p>
            <p className="text-2xl font-bold text-slate-900 dark:text-white">{metric.value}</p>
          </Card>
        ))}
      </div>

      {/* Model Cards */}
      <div className="space-y-4">
        {models.map((model) => (
          <Card key={model.name} className="p-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm hover:shadow-lg transition-all">
            <div className="flex items-start justify-between mb-6">
              <div className="flex items-start gap-4">
                <div className="p-3 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg">
                  <Brain className="h-6 w-6 text-white" />
                </div>
                <div>
                  <div className="flex items-center gap-3 mb-2">
                    <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                      {model.name}
                    </h3>
                    <Badge className="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                      <CheckCircle className="h-3 w-3 mr-1" />
                      {model.status}
                    </Badge>
                    <Badge variant="outline">
                      {model.version}
                    </Badge>
                  </div>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Last updated: {new Date(model.lastUpdated).toLocaleDateString('en-US', {
                      month: 'long',
                      day: 'numeric',
                      year: 'numeric'
                    })}
                  </p>
                </div>
              </div>

              <div className="flex gap-2">
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4 mr-2" />
                  Export
                </Button>
                <Button variant="outline" size="sm">
                  <RefreshCw className="h-4 w-4 mr-2" />
                  Retrain
                </Button>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-blue-700 dark:text-blue-300">Accuracy</span>
                  <TrendingUp className="h-4 w-4 text-blue-600" />
                </div>
                <p className="text-3xl font-bold text-blue-900 dark:text-blue-100">{model.accuracy}%</p>
                <div className="w-full bg-blue-200 dark:bg-blue-800 rounded-full h-2 mt-2">
                  <div
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${model.accuracy}%` }}
                  />
                </div>
              </div>

              <div className="bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-green-700 dark:text-green-300">Sensitivity</span>
                  <Zap className="h-4 w-4 text-green-600" />
                </div>
                <p className="text-3xl font-bold text-green-900 dark:text-green-100">{model.sensitivity}%</p>
                <div className="w-full bg-green-200 dark:bg-green-800 rounded-full h-2 mt-2">
                  <div
                    className="bg-green-600 h-2 rounded-full"
                    style={{ width: `${model.sensitivity}%` }}
                  />
                </div>
              </div>

              <div className="bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-800/20 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-purple-700 dark:text-purple-300">Specificity</span>
                  <Shield className="h-4 w-4 text-purple-600" />
                </div>
                <p className="text-3xl font-bold text-purple-900 dark:text-purple-100">{model.specificity}%</p>
                <div className="w-full bg-purple-200 dark:bg-purple-800 rounded-full h-2 mt-2">
                  <div
                    className="bg-purple-600 h-2 rounded-full"
                    style={{ width: `${model.specificity}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Model Details */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
                  Technical Specifications
                </h4>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600 dark:text-slate-400">Architecture:</span>
                    <span className="font-medium text-slate-900 dark:text-white">{model.architecture}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600 dark:text-slate-400">Training Dataset:</span>
                    <span className="font-medium text-slate-900 dark:text-white">{model.trainingData}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-600 dark:text-slate-400">Version:</span>
                    <span className="font-medium text-slate-900 dark:text-white">{model.version}</span>
                  </div>
                </div>
              </div>

              <div>
                <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300 mb-3">
                  Supported Cancer Types
                </h4>
                <div className="flex flex-wrap gap-2">
                  {model.cancerTypes.map((type) => (
                    <Badge key={type} variant="outline" className="bg-slate-50 dark:bg-slate-900">
                      {type}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Model Comparison Chart */}
      <Card className="mt-6 p-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-bold text-slate-900 dark:text-white mb-1">
              Model Performance Comparison
            </h3>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Comparative analysis of all deployed models
            </p>
          </div>
          <Button variant="outline" size="sm">
            <BarChart3 className="h-4 w-4 mr-2" />
            View Detailed Analytics
          </Button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {models.map((model) => (
            <div key={model.name} className="p-4 bg-slate-50 dark:bg-slate-900/50 rounded-lg">
              <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-3">
                {model.name.split(' ')[0]}
              </p>
              <div className="space-y-2">
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-500">Accuracy</span>
                    <span className="font-medium">{model.accuracy}%</span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
                    <div
                      className="bg-blue-600 h-1.5 rounded-full"
                      style={{ width: `${model.accuracy}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-500">Sensitivity</span>
                    <span className="font-medium">{model.sensitivity}%</span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
                    <div
                      className="bg-green-600 h-1.5 rounded-full"
                      style={{ width: `${model.sensitivity}%` }}
                    />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-500">Specificity</span>
                    <span className="font-medium">{model.specificity}%</span>
                  </div>
                  <div className="w-full bg-slate-200 dark:bg-slate-700 rounded-full h-1.5">
                    <div
                      className="bg-purple-600 h-1.5 rounded-full"
                      style={{ width: `${model.specificity}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}
