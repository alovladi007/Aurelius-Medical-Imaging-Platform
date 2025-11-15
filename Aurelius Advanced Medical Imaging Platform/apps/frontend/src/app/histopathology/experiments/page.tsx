"use client";

import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { FlaskConical, TrendingUp, Clock, Cpu, CheckCircle, XCircle, PlayCircle, ExternalLink } from 'lucide-react';

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState([]);
  const [selectedExp, setSelectedExp] = useState(null);

  useEffect(() => {
    loadExperiments();
  }, []);

  const loadExperiments = async () => {
    try {
      const response = await fetch('/api/histopathology/experiments');
      if (response.ok) {
        const data = await response.json();
        setExperiments(data.experiments);
      } else {
        // Mock data
        setExperiments([
          {
            id: 'exp-001',
            name: 'ResNet-50 Brain Cancer',
            model: 'resnet50',
            status: 'completed',
            accuracy: 95.6,
            precision: 94.2,
            recall: 93.8,
            f1Score: 94.0,
            trainLoss: 0.145,
            valLoss: 0.168,
            epochs: 50,
            batchSize: 32,
            learningRate: 0.001,
            duration: '2h 15m',
            startTime: '2025-11-14 10:00',
            endTime: '2025-11-14 12:15'
          },
          {
            id: 'exp-002',
            name: 'EfficientNet-B3 Lung',
            model: 'efficientnet_b3',
            status: 'running',
            accuracy: 92.1,
            epochs: 30,
            currentEpoch: 18,
            batchSize: 16,
            learningRate: 0.0005,
            duration: '1h 30m',
            startTime: '2025-11-15 09:00'
          },
          {
            id: 'exp-003',
            name: 'ViT Base Colorectal',
            model: 'vit_base',
            status: 'failed',
            error: 'CUDA out of memory',
            epochs: 40,
            batchSize: 16,
            startTime: '2025-11-13 14:00',
            endTime: '2025-11-13 14:05'
          }
        ]);
      }
    } catch (error) {
      console.error('Error loading experiments:', error);
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge className="bg-green-600"><CheckCircle className="h-3 w-3 mr-1" />Completed</Badge>;
      case 'running':
        return <Badge className="bg-blue-600"><PlayCircle className="h-3 w-3 mr-1" />Running</Badge>;
      case 'failed':
        return <Badge variant="destructive"><XCircle className="h-3 w-3 mr-1" />Failed</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  return (
    <div className="p-8 space-y-8">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <FlaskConical className="h-8 w-8 text-primary" />
            Experiments
          </h1>
          <p className="text-gray-600 dark:text-gray-400">Track and compare training experiments with MLflow</p>
        </div>
        <Button onClick={() => window.open('http://localhost:11000', '_blank')}>
          <ExternalLink className="h-4 w-4 mr-2" />
          Open MLflow UI
        </Button>
      </div>

      {/* Experiments Table */}
      <Card>
        <CardHeader>
          <CardTitle>All Experiments</CardTitle>
          <CardDescription>View and compare all training runs</CardDescription>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Experiment</TableHead>
                <TableHead>Model</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Accuracy</TableHead>
                <TableHead>Duration</TableHead>
                <TableHead>Started</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {experiments.map((exp: any) => (
                <TableRow key={exp.id} className="cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-900" onClick={() => setSelectedExp(exp)}>
                  <TableCell className="font-medium">{exp.name}</TableCell>
                  <TableCell><Badge variant="outline">{exp.model}</Badge></TableCell>
                  <TableCell>{getStatusBadge(exp.status)}</TableCell>
                  <TableCell>{exp.accuracy ? `${exp.accuracy}%` : '-'}</TableCell>
                  <TableCell>{exp.duration || '-'}</TableCell>
                  <TableCell className="text-sm text-gray-600">{exp.startTime}</TableCell>
                  <TableCell>
                    <Button variant="ghost" size="sm">View</Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Selected Experiment Details */}
      {selectedExp && (
        <Card className="border-primary">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>{selectedExp.name}</CardTitle>
              <Button variant="ghost" size="sm" onClick={() => setSelectedExp(null)}>✕</Button>
            </div>
            <CardDescription>Experiment ID: {selectedExp.id}</CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs defaultValue="metrics">
              <TabsList>
                <TabsTrigger value="metrics">Metrics</TabsTrigger>
                <TabsTrigger value="config">Configuration</TabsTrigger>
                <TabsTrigger value="artifacts">Artifacts</TabsTrigger>
              </TabsList>

              <TabsContent value="metrics" className="space-y-4">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="text-center p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                    <p className="text-sm text-gray-600 dark:text-gray-400">Accuracy</p>
                    <p className="text-3xl font-bold text-blue-600">{selectedExp.accuracy || '-'}%</p>
                  </div>
                  <div className="text-center p-4 bg-green-50 dark:bg-green-950 rounded-lg">
                    <p className="text-sm text-gray-600 dark:text-gray-400">Precision</p>
                    <p className="text-3xl font-bold text-green-600">{selectedExp.precision || '-'}%</p>
                  </div>
                  <div className="text-center p-4 bg-orange-50 dark:bg-orange-950 rounded-lg">
                    <p className="text-sm text-gray-600 dark:text-gray-400">Recall</p>
                    <p className="text-3xl font-bold text-orange-600">{selectedExp.recall || '-'}%</p>
                  </div>
                  <div className="text-center p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
                    <p className="text-sm text-gray-600 dark:text-gray-400">F1 Score</p>
                    <p className="text-3xl font-bold text-purple-600">{selectedExp.f1Score || '-'}%</p>
                  </div>
                </div>
                {selectedExp.trainLoss && (
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Train Loss</p>
                      <p className="text-2xl font-bold">{selectedExp.trainLoss}</p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Val Loss</p>
                      <p className="text-2xl font-bold">{selectedExp.valLoss}</p>
                    </div>
                  </div>
                )}
              </TabsContent>

              <TabsContent value="config" className="space-y-2">
                <dl className="grid grid-cols-2 gap-4">
                  <div><dt className="text-sm text-gray-600">Model</dt><dd className="font-mono">{selectedExp.model}</dd></div>
                  <div><dt className="text-sm text-gray-600">Epochs</dt><dd className="font-mono">{selectedExp.epochs}</dd></div>
                  <div><dt className="text-sm text-gray-600">Batch Size</dt><dd className="font-mono">{selectedExp.batchSize}</dd></div>
                  <div><dt className="text-sm text-gray-600">Learning Rate</dt><dd className="font-mono">{selectedExp.learningRate}</dd></div>
                  <div><dt className="text-sm text-gray-600">Started</dt><dd className="font-mono text-sm">{selectedExp.startTime}</dd></div>
                  <div><dt className="text-sm text-gray-600">Ended</dt><dd className="font-mono text-sm">{selectedExp.endTime || 'Running'}</dd></div>
                </dl>
              </TabsContent>

              <TabsContent value="artifacts">
                <div className="space-y-2">
                  <Button variant="outline" className="w-full justify-start">
                    <Download className="h-4 w-4 mr-2" />
                    Download Model Checkpoint
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <Download className="h-4 w-4 mr-2" />
                    Download Confusion Matrix
                  </Button>
                  <Button variant="outline" className="w-full justify-start">
                    <Download className="h-4 w-4 mr-2" />
                    Download Training Curves
                  </Button>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
