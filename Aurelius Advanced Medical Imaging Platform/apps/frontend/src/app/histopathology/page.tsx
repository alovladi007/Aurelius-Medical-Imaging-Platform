"use client";

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Database,
  Play,
  FlaskConical,
  Image,
  Brain,
  BarChart3,
  Upload,
  Download,
  Clock,
  CheckCircle,
  Cpu,
  Microscope,
  Sparkles,
  TrendingUp,
  FileText,
  Zap
} from 'lucide-react';

export default function HistopathologyDashboard() {
  const [stats, setStats] = useState({
    totalDatasets: 0,
    totalImages: 0,
    trainedModels: 0,
    activeExperiments: 0,
    totalFeatures: 0,
    avgAccuracy: 0
  });

  const [recentExperiments, setRecentExperiments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      // Fetch real data from backend
      const response = await fetch('/api/histopathology/dashboard');
      if (response.ok) {
        const data = await response.json();
        setStats(data.stats);
        setRecentExperiments(data.recentExperiments || []);
      } else {
        // Mock data if backend not available
        setStats({
          totalDatasets: 3,
          totalImages: 1248,
          trainedModels: 5,
          activeExperiments: 2,
          totalFeatures: 127,
          avgAccuracy: 94.2
        });
        setRecentExperiments([
          {
            id: 1,
            name: 'ResNet-50 Brain Cancer',
            status: 'completed',
            accuracy: 95.6,
            date: '2025-11-14'
          },
          {
            id: 2,
            name: 'EfficientNet-B3 Lung',
            status: 'running',
            accuracy: 92.1,
            date: '2025-11-15'
          }
        ]);
      }
    } catch (error) {
      console.error('Error loading dashboard:', error);
    } finally {
      setLoading(false);
    }
  };

  const features = [
    {
      title: 'Dataset Management',
      description: 'Upload, organize, and prepare histopathology datasets',
      icon: Database,
      href: '/histopathology/datasets',
      color: 'text-blue-600',
      bgColor: 'bg-blue-50 dark:bg-blue-950',
      stats: `${stats.totalDatasets} datasets`
    },
    {
      title: 'Train Models',
      description: 'Train ResNet, EfficientNet, or ViT on your data',
      icon: Play,
      href: '/histopathology/train',
      color: 'text-green-600',
      bgColor: 'bg-green-50 dark:bg-green-950',
      stats: `${stats.trainedModels} models`
    },
    {
      title: 'Experiments',
      description: 'Track and compare training experiments with MLflow',
      icon: FlaskConical,
      href: '/histopathology/experiments',
      color: 'text-purple-600',
      bgColor: 'bg-purple-50 dark:bg-purple-950',
      stats: `${stats.activeExperiments} active`
    },
    {
      title: 'Feature Extraction',
      description: 'Extract 100+ quantitative features from images',
      icon: Sparkles,
      href: '/histopathology/features',
      color: 'text-orange-600',
      bgColor: 'bg-orange-50 dark:bg-orange-950',
      stats: `${stats.totalFeatures} features`
    },
    {
      title: 'Inference',
      description: 'Run predictions on new images with trained models',
      icon: Brain,
      href: '/histopathology/inference',
      color: 'text-pink-600',
      bgColor: 'bg-pink-50 dark:bg-pink-950',
      stats: 'Single & Batch'
    },
    {
      title: 'Grad-CAM Visualization',
      description: 'Visualize what your models are learning',
      icon: Microscope,
      href: '/histopathology/gradcam',
      color: 'text-indigo-600',
      bgColor: 'bg-indigo-50 dark:bg-indigo-950',
      stats: 'Explainability'
    },
    {
      title: 'Results & Metrics',
      description: 'View performance metrics and evaluation results',
      icon: BarChart3,
      href: '/histopathology/results',
      color: 'text-red-600',
      bgColor: 'bg-red-50 dark:bg-red-950',
      stats: `${stats.avgAccuracy}% avg`
    },
    {
      title: 'Documentation',
      description: 'Guides, tutorials, and API documentation',
      icon: FileText,
      href: '/histopathology/docs',
      color: 'text-gray-600',
      bgColor: 'bg-gray-50 dark:bg-gray-950',
      stats: 'Guides'
    }
  ];

  const quickStats = [
    {
      title: 'Total Images',
      value: stats.totalImages.toLocaleString(),
      icon: Image,
      trend: '+12%',
      trendUp: true
    },
    {
      title: 'Trained Models',
      value: stats.trainedModels,
      icon: Brain,
      trend: '+2',
      trendUp: true
    },
    {
      title: 'Avg Accuracy',
      value: `${stats.avgAccuracy}%`,
      icon: TrendingUp,
      trend: '+3.2%',
      trendUp: true
    },
    {
      title: 'Active Experiments',
      value: stats.activeExperiments,
      icon: Zap,
      trend: 'Running',
      trendUp: true
    }
  ];

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
          <p className="text-gray-600 dark:text-gray-400">Loading histopathology dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-4xl font-bold mb-2 flex items-center gap-3">
          <Microscope className="h-10 w-10 text-primary" />
          Quantitative Histopathology
        </h1>
        <p className="text-gray-600 dark:text-gray-400 text-lg">
          Advanced ML pipeline for cancer tissue analysis and research
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {quickStats.map((stat, index) => (
          <Card key={index}>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">{stat.title}</CardTitle>
              <stat.icon className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stat.value}</div>
              <p className={`text-xs ${stat.trendUp ? 'text-green-600' : 'text-red-600'} flex items-center gap-1`}>
                {stat.trendUp ? <TrendingUp className="h-3 w-3" /> : null}
                {stat.trend}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Feature Grid */}
      <div>
        <h2 className="text-2xl font-bold mb-4">Capabilities</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => (
            <Link key={index} href={feature.href}>
              <Card className="h-full hover:shadow-lg transition-shadow cursor-pointer">
                <CardHeader>
                  <div className={`w-12 h-12 rounded-lg ${feature.bgColor} flex items-center justify-center mb-3`}>
                    <feature.icon className={`h-6 w-6 ${feature.color}`} />
                  </div>
                  <CardTitle className="text-lg">{feature.title}</CardTitle>
                  <CardDescription>{feature.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <Badge variant="secondary">{feature.stats}</Badge>
                </CardContent>
              </Card>
            </Link>
          ))}
        </div>
      </div>

      {/* Recent Experiments */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-2xl font-bold">Recent Experiments</h2>
          <Link href="/histopathology/experiments">
            <Button variant="outline">View All</Button>
          </Link>
        </div>

        <Card>
          <CardContent className="pt-6">
            {recentExperiments.length === 0 ? (
              <div className="text-center py-12">
                <FlaskConical className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-600 dark:text-gray-400 mb-4">No experiments yet</p>
                <Link href="/histopathology/train">
                  <Button>Start Training</Button>
                </Link>
              </div>
            ) : (
              <div className="space-y-4">
                {recentExperiments.map((exp: any) => (
                  <div key={exp.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center gap-4">
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                        exp.status === 'completed' ? 'bg-green-100 dark:bg-green-950' :
                        exp.status === 'running' ? 'bg-blue-100 dark:bg-blue-950' :
                        'bg-gray-100 dark:bg-gray-950'
                      }`}>
                        {exp.status === 'completed' ? <CheckCircle className="h-5 w-5 text-green-600" /> :
                         exp.status === 'running' ? <Cpu className="h-5 w-5 text-blue-600 animate-pulse" /> :
                         <Clock className="h-5 w-5 text-gray-600" />}
                      </div>
                      <div>
                        <p className="font-medium">{exp.name}</p>
                        <p className="text-sm text-gray-600 dark:text-gray-400">{exp.date}</p>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      {exp.accuracy && (
                        <div className="text-right">
                          <p className="text-2xl font-bold text-primary">{exp.accuracy}%</p>
                          <p className="text-xs text-gray-600">Accuracy</p>
                        </div>
                      )}
                      <Badge variant={exp.status === 'completed' ? 'default' : 'secondary'}>
                        {exp.status}
                      </Badge>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Getting Started Guide */}
      <Card className="border-primary">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-primary" />
            Getting Started
          </CardTitle>
          <CardDescription>Quick workflow to train your first model</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center text-sm font-bold">
                1
              </div>
              <div>
                <p className="font-medium">Prepare Dataset</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Upload your histopathology images or generate synthetic data
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center text-sm font-bold">
                2
              </div>
              <div>
                <p className="font-medium">Configure Training</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Choose model architecture (ResNet, EfficientNet, ViT) and hyperparameters
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center text-sm font-bold">
                3
              </div>
              <div>
                <p className="font-medium">Train & Track</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  Start training and monitor progress with MLflow integration
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-primary text-white flex items-center justify-center text-sm font-bold">
                4
              </div>
              <div>
                <p className="font-medium">Evaluate & Deploy</p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  View results, extract features, and run inference on new images
                </p>
              </div>
            </div>
          </div>
          <div className="mt-6 flex gap-3">
            <Link href="/histopathology/datasets">
              <Button>Get Started</Button>
            </Link>
            <Link href="/histopathology/docs">
              <Button variant="outline">View Documentation</Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
