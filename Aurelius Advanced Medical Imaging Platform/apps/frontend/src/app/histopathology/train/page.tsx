"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Play, Square, Cpu, Zap, TrendingUp, Clock } from 'lucide-react';

export default function TrainPage() {
  const [training, setTraining] = useState(false);
  const [config, setConfig] = useState({
    datasetPath: '/data/raw/brain_cancer',
    model: 'resnet50',
    batchSize: 32,
    epochs: 50,
    learningRate: 0.001,
    optimizer: 'adam',
    scheduler: 'cosine',
    augmentation: true,
    mixedPrecision: true,
    numWorkers: 4,
    experimentName: 'brain-cancer-' + Date.now()
  });
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [currentEpoch, setCurrentEpoch] = useState(0);

  const startTraining = async () => {
    setTraining(true);
    try {
      const response = await fetch('/api/histopathology/train', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });

      if (response.ok) {
        const reader = response.body?.getReader();
        if (reader) {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const text = new TextDecoder().decode(value);
            const lines = text.split('\n');
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                setCurrentEpoch(data.epoch);
                setTrainingMetrics(data);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Training error:', error);
      alert('Error starting training');
    } finally {
      setTraining(false);
    }
  };

  const stopTraining = async () => {
    await fetch('/api/histopathology/train/stop', { method: 'POST' });
    setTraining(false);
  };

  return (
    <div className="p-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Train Models</h1>
          <p className="text-gray-600 dark:text-gray-400">Configure and train histopathology models</p>
        </div>
        {training ? (
          <Button onClick={stopTraining} variant="destructive">
            <Square className="h-4 w-4 mr-2" />
            Stop Training
          </Button>
        ) : (
          <Button onClick={startTraining} size="lg">
            <Play className="h-4 w-4 mr-2" />
            Start Training
          </Button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Configuration Panel */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Training Configuration</CardTitle>
              <CardDescription>Configure your model training parameters</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <Tabs defaultValue="model">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="model">Model</TabsTrigger>
                  <TabsTrigger value="training">Training</TabsTrigger>
                  <TabsTrigger value="advanced">Advanced</TabsTrigger>
                </TabsList>

                <TabsContent value="model" className="space-y-4 mt-4">
                  <div className="space-y-2">
                    <Label>Model Architecture</Label>
                    <Select value={config.model} onValueChange={(v) => setConfig({...config, model: v})}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="resnet18">ResNet-18</SelectItem>
                        <SelectItem value="resnet34">ResNet-34</SelectItem>
                        <SelectItem value="resnet50">ResNet-50 (Recommended)</SelectItem>
                        <SelectItem value="resnet101">ResNet-101</SelectItem>
                        <SelectItem value="efficientnet_b0">EfficientNet-B0</SelectItem>
                        <SelectItem value="efficientnet_b3">EfficientNet-B3</SelectItem>
                        <SelectItem value="efficientnet_b5">EfficientNet-B5</SelectItem>
                        <SelectItem value="vit_base">Vision Transformer (ViT) Base</SelectItem>
                        <SelectItem value="vit_large">Vision Transformer (ViT) Large</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <Label>Dataset Path</Label>
                    <Input
                      value={config.datasetPath}
                      onChange={(e) => setConfig({...config, datasetPath: e.target.value})}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label>Experiment Name</Label>
                    <Input
                      value={config.experimentName}
                      onChange={(e) => setConfig({...config, experimentName: e.target.value})}
                    />
                  </div>
                </TabsContent>

                <TabsContent value="training" className="space-y-4 mt-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Batch Size</Label>
                      <Input
                        type="number"
                        value={config.batchSize}
                        onChange={(e) => setConfig({...config, batchSize: parseInt(e.target.value)})}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Max Epochs</Label>
                      <Input
                        type="number"
                        value={config.epochs}
                        onChange={(e) => setConfig({...config, epochs: parseInt(e.target.value)})}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Learning Rate</Label>
                      <Input
                        type="number"
                        step="0.0001"
                        value={config.learningRate}
                        onChange={(e) => setConfig({...config, learningRate: parseFloat(e.target.value)})}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label>Optimizer</Label>
                      <Select value={config.optimizer} onValueChange={(v) => setConfig({...config, optimizer: v})}>
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="adam">Adam</SelectItem>
                          <SelectItem value="adamw">AdamW</SelectItem>
                          <SelectItem value="sgd">SGD</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </TabsContent>

                <TabsContent value="advanced" className="space-y-4 mt-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="mixedPrecision"
                        checked={config.mixedPrecision}
                        onChange={(e) => setConfig({...config, mixedPrecision: e.target.checked})}
                        className="h-4 w-4"
                      />
                      <Label htmlFor="mixedPrecision">Mixed Precision (FP16)</Label>
                    </div>
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="augmentation"
                        checked={config.augmentation}
                        onChange={(e) => setConfig({...config, augmentation: e.target.checked})}
                        className="h-4 w-4"
                      />
                      <Label htmlFor="augmentation">Data Augmentation</Label>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Learning Rate Scheduler</Label>
                    <Select value={config.scheduler} onValueChange={(v) => setConfig({...config, scheduler: v})}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="cosine">Cosine Annealing</SelectItem>
                        <SelectItem value="step">Step LR</SelectItem>
                        <SelectItem value="reduce">Reduce on Plateau</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Training Progress */}
          {training && (
            <Card className="border-primary">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Cpu className="h-5 w-5 animate-pulse text-primary" />
                  Training in Progress
                </CardTitle>
                <CardDescription>Epoch {currentEpoch} / {config.epochs}</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                  <div
                    className="bg-primary h-2 rounded-full transition-all"
                    style={{ width: `${(currentEpoch / config.epochs) * 100}%` }}
                  />
                </div>
                {trainingMetrics && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Train Loss</p>
                      <p className="text-2xl font-bold">{trainingMetrics.trainLoss?.toFixed(4)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Val Loss</p>
                      <p className="text-2xl font-bold">{trainingMetrics.valLoss?.toFixed(4)}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Accuracy</p>
                      <p className="text-2xl font-bold text-primary">{trainingMetrics.accuracy?.toFixed(2)}%</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Time</p>
                      <p className="text-2xl font-bold">{trainingMetrics.elapsed || '0s'}</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>

        {/* Info Panel */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Quick Info</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex items-center gap-2">
                <Cpu className="h-4 w-4 text-primary" />
                <span className="text-gray-600 dark:text-gray-400">GPU: CUDA Detected</span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-primary" />
                <span className="text-gray-600 dark:text-gray-400">Mixed Precision: Enabled</span>
              </div>
              <div className="flex items-center gap-2">
                <Clock className="h-4 w-4 text-primary" />
                <span className="text-gray-600 dark:text-gray-400">Est. Time: ~2h</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Recommended Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <p className="text-gray-600 dark:text-gray-400">
                <strong>ResNet-50:</strong> Batch 32, LR 0.001
              </p>
              <p className="text-gray-600 dark:text-gray-400">
                <strong>EfficientNet-B3:</strong> Batch 16, LR 0.0005
              </p>
              <p className="text-gray-600 dark:text-gray-400">
                <strong>ViT:</strong> Batch 16, LR 0.0001
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
