'use client'

import { useState } from 'react'
import { Brain, Play, CheckCircle, Clock, TrendingUp } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

export default function MLPage() {
  const [selectedModel, setSelectedModel] = useState<string | null>(null)

  const models = [
    {
      id: '1',
      name: 'Lung Nodule Detection',
      description: 'Detect and classify lung nodules in CT scans',
      accuracy: 94.5,
      modality: ['CT'],
      status: 'active',
      runs: 1240
    },
    {
      id: '2',
      name: 'Brain Tumor Segmentation',
      description: 'Segment and classify brain tumors in MRI',
      accuracy: 92.1,
      modality: ['MRI'],
      status: 'active',
      runs: 892
    },
    {
      id: '3',
      name: 'Fracture Detection',
      description: 'Identify fractures in X-ray images',
      accuracy: 89.7,
      modality: ['X-Ray'],
      status: 'active',
      runs: 2156
    },
    {
      id: '4',
      name: 'Cardiac Segmentation',
      description: 'Segment cardiac structures in CT/MRI',
      accuracy: 91.3,
      modality: ['CT', 'MRI'],
      status: 'active',
      runs: 743
    },
  ]

  const recentInferences = [
    { id: '1', studyId: 'STD-001', model: 'Lung Nodule Detection', status: 'completed', confidence: 95.2, time: '2 min ago' },
    { id: '2', studyId: 'STD-002', model: 'Brain Tumor Segmentation', status: 'running', confidence: null, time: '5 min ago' },
    { id: '3', studyId: 'STD-003', model: 'Fracture Detection', status: 'completed', confidence: 88.9, time: '12 min ago' },
  ]

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
      <div className="max-w-7xl mx-auto p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            AI/ML Inference
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Run AI models on medical imaging studies
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          {[
            { label: 'Total Inferences', value: '5,031', icon: Brain, color: 'text-purple-600' },
            { label: 'Active Models', value: '12', icon: CheckCircle, color: 'text-green-600' },
            { label: 'Avg Accuracy', value: '92.4%', icon: TrendingUp, color: 'text-blue-600' },
            { label: 'Running', value: '3', icon: Clock, color: 'text-yellow-600' },
          ].map((stat, i) => (
            <Card key={i}>
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-slate-600 dark:text-slate-400">{stat.label}</p>
                    <p className="text-2xl font-bold text-slate-900 dark:text-white mt-1">{stat.value}</p>
                  </div>
                  <stat.icon className={`h-8 w-8 ${stat.color}`} />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Available Models */}
          <div className="lg:col-span-2">
            <Card>
              <CardHeader>
                <CardTitle>Available AI Models</CardTitle>
                <CardDescription>Select a model to run inference on your studies</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4">
                  {models.map((model) => (
                    <div
                      key={model.id}
                      className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                        selectedModel === model.id
                          ? 'border-primary bg-primary/5'
                          : 'border-slate-200 hover:border-slate-300'
                      }`}
                      onClick={() => setSelectedModel(model.id)}
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex-1">
                          <h3 className="font-semibold text-slate-900 dark:text-white mb-1">
                            {model.name}
                          </h3>
                          <p className="text-sm text-slate-600 dark:text-slate-400 mb-3">
                            {model.description}
                          </p>
                          <div className="flex items-center space-x-4 text-xs">
                            <span className="text-green-600 font-semibold">
                              {model.accuracy}% accuracy
                            </span>
                            <span className="text-slate-500">
                              {model.modality.join(', ')}
                            </span>
                            <span className="text-slate-500">
                              {model.runs.toLocaleString()} runs
                            </span>
                          </div>
                        </div>
                        <Button size="sm">
                          <Play className="h-4 w-4 mr-2" />
                          Run
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Recent Inferences */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Inferences</CardTitle>
              <CardDescription>Latest AI analysis results</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentInferences.map((inference) => (
                  <div key={inference.id} className="p-3 bg-slate-50 dark:bg-slate-800 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium text-slate-900 dark:text-white">
                        {inference.studyId}
                      </span>
                      {inference.status === 'completed' ? (
                        <CheckCircle className="h-4 w-4 text-green-600" />
                      ) : (
                        <Clock className="h-4 w-4 text-yellow-600 animate-pulse" />
                      )}
                    </div>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mb-1">
                      {inference.model}
                    </p>
                    {inference.confidence && (
                      <p className="text-xs text-green-600 font-semibold">
                        {inference.confidence}% confidence
                      </p>
                    )}
                    <p className="text-xs text-slate-500 mt-2">{inference.time}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
