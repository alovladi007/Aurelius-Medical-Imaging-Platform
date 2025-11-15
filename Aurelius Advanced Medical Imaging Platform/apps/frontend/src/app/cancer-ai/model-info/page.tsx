"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Brain,
  Cpu,
  Database,
  Zap,
  CheckCircle,
  Info,
  Download,
  BookOpen,
  Shield,
  Activity,
  Layers,
  FileCode
} from 'lucide-react';

export default function ModelInfoPage() {
  const [selectedModel, setSelectedModel] = useState('primary');

  const modelVersions = [
    {
      id: 'primary',
      name: 'Cancer AI v3.2',
      status: 'Production',
      deployed: '2025-10-15',
      accuracy: 94.2,
      description: 'Primary multimodal cancer detection model'
    },
    {
      id: 'v3.1',
      name: 'Cancer AI v3.1',
      status: 'Archived',
      deployed: '2025-08-20',
      accuracy: 93.1,
      description: 'Previous stable version'
    },
    {
      id: 'experimental',
      name: 'Cancer AI v4.0-beta',
      status: 'Testing',
      deployed: '2025-11-01',
      accuracy: 95.7,
      description: 'Experimental version with enhanced vision transformer'
    }
  ];

  const supportedCancers = [
    { name: 'Lung Cancer', types: ['NSCLC', 'SCLC', 'Adenocarcinoma'], accuracy: 95.3, icon: '🫁' },
    { name: 'Breast Cancer', types: ['Ductal', 'Lobular', 'Triple-negative'], accuracy: 96.1, icon: '🎗️' },
    { name: 'Colorectal Cancer', types: ['Colon', 'Rectal', 'Polyps'], accuracy: 92.8, icon: '🔬' },
    { name: 'Prostate Cancer', types: ['Adenocarcinoma', 'Neuroendocrine'], accuracy: 94.5, icon: '⚕️' },
    { name: 'Skin Cancer', types: ['Melanoma', 'Basal Cell', 'Squamous Cell'], accuracy: 97.2, icon: '🔍' },
    { name: 'Brain Tumors', types: ['Glioblastoma', 'Meningioma', 'Astrocytoma'], accuracy: 91.8, icon: '🧠' }
  ];

  const supportedModalities = [
    { name: 'CT Scan', formats: ['DICOM', 'NIfTI'], resolution: 'Up to 512x512x512', preprocessing: 'HU windowing, 3D patches' },
    { name: 'MRI', formats: ['DICOM', 'NIfTI'], resolution: 'Up to 256x256x256', preprocessing: 'N4 bias correction, normalization' },
    { name: 'X-Ray', formats: ['DICOM', 'PNG', 'JPEG'], resolution: 'Up to 4096x4096', preprocessing: 'CLAHE, resize' },
    { name: 'Mammography', formats: ['DICOM'], resolution: 'Up to 3328x2560', preprocessing: 'Breast segmentation, enhancement' },
    { name: 'Ultrasound', formats: ['DICOM', 'MP4'], resolution: 'Variable', preprocessing: 'Speckle reduction, contrast adjustment' },
    { name: 'PET/CT', formats: ['DICOM'], resolution: 'Fused modality', preprocessing: 'SUV calculation, registration' }
  ];

  const technicalSpecs = {
    architecture: {
      backbone: 'EfficientNetV2-L + Vision Transformer (ViT-L/16)',
      inputSize: 'Variable (224x224 to 512x512)',
      parameters: '304M trainable parameters',
      layers: '48 transformer blocks + CNN backbone',
      attention: 'Multi-head self-attention (16 heads)'
    },
    training: {
      dataset: '2.4M medical images across 6 cancer types',
      epochs: '150 epochs with early stopping',
      optimizer: 'AdamW (lr=1e-4, weight_decay=0.01)',
      augmentation: 'RandomRotation, RandomFlip, ColorJitter, Cutout',
      hardware: '8x NVIDIA A100 80GB GPUs',
      time: '14 days training time'
    },
    inference: {
      precision: 'FP16 mixed precision',
      batchSize: '1-32 images',
      latency: '1.2s average (single image)',
      throughput: '~50 images/minute',
      memory: '8GB VRAM required',
      optimization: 'TorchScript compiled, ONNX export available'
    }
  };

  const performanceBenchmarks = [
    { metric: 'Overall Accuracy', value: '94.2%', benchmark: 'Top 5% of published models', color: 'text-green-600' },
    { metric: 'Precision', value: '96.8%', benchmark: 'Exceeds clinical requirements', color: 'text-blue-600' },
    { metric: 'Recall (Sensitivity)', value: '93.5%', benchmark: 'Above 90% threshold', color: 'text-purple-600' },
    { metric: 'Specificity', value: '95.1%', benchmark: 'Low false positive rate', color: 'text-orange-600' },
    { metric: 'AUROC', value: '0.97', benchmark: 'Excellent discriminative ability', color: 'text-pink-600' },
    { metric: 'F1-Score', value: '95.1%', benchmark: 'Balanced performance', color: 'text-indigo-600' }
  ];

  const features = [
    {
      title: 'Multimodal Support',
      description: 'Handles CT, MRI, X-Ray, Mammography, Ultrasound, and PET/CT imaging',
      icon: Layers
    },
    {
      title: 'Explainable AI',
      description: 'Grad-CAM visualizations show exactly where the model detects abnormalities',
      icon: Brain
    },
    {
      title: 'Uncertainty Quantification',
      description: 'Provides confidence scores and uncertainty estimates for every prediction',
      icon: Activity
    },
    {
      title: 'Real-time Inference',
      description: 'Fast predictions (1.2s average) suitable for clinical workflows',
      icon: Zap
    },
    {
      title: 'HIPAA Compliant',
      description: 'All data processing follows HIPAA guidelines with encryption and audit logging',
      icon: Shield
    },
    {
      title: 'Continuous Learning',
      description: 'Model can be fine-tuned on institution-specific data while preserving privacy',
      icon: Database
    }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Brain className="h-8 w-8 text-primary" />
            Model Information
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Technical specifications and capabilities of cancer AI models
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <FileCode className="h-4 w-4 mr-2" />
            API Docs
          </Button>
          <Button>
            <Download className="h-4 w-4 mr-2" />
            Model Card
          </Button>
        </div>
      </div>

      {/* Model Version Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Model Versions</CardTitle>
          <CardDescription>Available model versions and deployment status</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {modelVersions.map((version) => (
              <div
                key={version.id}
                className={`p-4 border-2 rounded-lg cursor-pointer transition-all ${
                  selectedModel === version.id
                    ? 'border-primary bg-primary/5'
                    : 'border-gray-200 dark:border-gray-800 hover:border-primary/50'
                }`}
                onClick={() => setSelectedModel(version.id)}
              >
                <div className="flex items-center justify-between mb-2">
                  <p className="font-semibold">{version.name}</p>
                  <Badge
                    variant={
                      version.status === 'Production'
                        ? 'default'
                        : version.status === 'Testing'
                        ? 'outline'
                        : 'secondary'
                    }
                    className={
                      version.status === 'Production'
                        ? 'bg-green-600'
                        : version.status === 'Testing'
                        ? 'border-yellow-600 text-yellow-600'
                        : ''
                    }
                  >
                    {version.status}
                  </Badge>
                </div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-3">{version.description}</p>
                <div className="flex items-center justify-between text-sm">
                  <span className="text-gray-600">Accuracy:</span>
                  <span className="font-bold text-primary">{version.accuracy}%</span>
                </div>
                <div className="flex items-center justify-between text-sm mt-1">
                  <span className="text-gray-600">Deployed:</span>
                  <span className="text-gray-800 dark:text-gray-200">{version.deployed}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Key Features */}
      <Card>
        <CardHeader>
          <CardTitle>Key Capabilities</CardTitle>
          <CardDescription>Core features and functionality</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {features.map((feature, index) => (
              <div key={index} className="p-4 border rounded-lg">
                <div className="flex items-start gap-3">
                  <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center flex-shrink-0">
                    <feature.icon className="h-5 w-5 text-primary" />
                  </div>
                  <div>
                    <p className="font-semibold text-sm mb-1">{feature.title}</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400">{feature.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Technical Specifications */}
      <Card>
        <CardHeader>
          <CardTitle>Technical Specifications</CardTitle>
          <CardDescription>Architecture, training, and inference details</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="architecture">
            <TabsList>
              <TabsTrigger value="architecture">Architecture</TabsTrigger>
              <TabsTrigger value="training">Training</TabsTrigger>
              <TabsTrigger value="inference">Inference</TabsTrigger>
            </TabsList>

            <TabsContent value="architecture" className="space-y-3 mt-4">
              {Object.entries(technicalSpecs.architecture).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded">
                  <span className="font-semibold text-sm capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}:</span>
                  <span className="text-sm text-gray-700 dark:text-gray-300">{value}</span>
                </div>
              ))}
            </TabsContent>

            <TabsContent value="training" className="space-y-3 mt-4">
              {Object.entries(technicalSpecs.training).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded">
                  <span className="font-semibold text-sm capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}:</span>
                  <span className="text-sm text-gray-700 dark:text-gray-300">{value}</span>
                </div>
              ))}
            </TabsContent>

            <TabsContent value="inference" className="space-y-3 mt-4">
              {Object.entries(technicalSpecs.inference).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded">
                  <span className="font-semibold text-sm capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}:</span>
                  <span className="text-sm text-gray-700 dark:text-gray-300">{value}</span>
                </div>
              ))}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Supported Cancer Types */}
      <Card>
        <CardHeader>
          <CardTitle>Supported Cancer Types</CardTitle>
          <CardDescription>Cancer types the model can detect with subtypes</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {supportedCancers.map((cancer, index) => (
              <div key={index} className="p-4 border rounded-lg">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-2xl">{cancer.icon}</span>
                  <div className="flex-1">
                    <p className="font-semibold">{cancer.name}</p>
                    <Badge variant="outline" className="mt-1">{cancer.accuracy}% accuracy</Badge>
                  </div>
                </div>
                <div>
                  <p className="text-xs font-semibold text-gray-600 dark:text-gray-400 mb-1">Subtypes:</p>
                  <div className="flex flex-wrap gap-1">
                    {cancer.types.map((type, idx) => (
                      <Badge key={idx} variant="secondary" className="text-xs">
                        {type}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Supported Imaging Modalities */}
      <Card>
        <CardHeader>
          <CardTitle>Supported Imaging Modalities</CardTitle>
          <CardDescription>Compatible medical imaging formats and preprocessing</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {supportedModalities.map((modality, index) => (
              <div key={index} className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <p className="font-semibold">{modality.name}</p>
                  <div className="flex gap-2">
                    {modality.formats.map((format, idx) => (
                      <Badge key={idx} variant="outline">
                        {format}
                      </Badge>
                    ))}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Max Resolution:</p>
                    <p className="font-mono text-xs mt-1">{modality.resolution}</p>
                  </div>
                  <div>
                    <p className="text-gray-600 dark:text-gray-400">Preprocessing:</p>
                    <p className="text-xs mt-1">{modality.preprocessing}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Performance Benchmarks */}
      <Card>
        <CardHeader>
          <CardTitle>Performance Benchmarks</CardTitle>
          <CardDescription>Model performance compared to clinical and research standards</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {performanceBenchmarks.map((item, index) => (
              <div key={index} className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <p className="font-semibold text-sm">{item.metric}</p>
                  <p className={`text-2xl font-bold ${item.color}`}>{item.value}</p>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                  <p className="text-xs text-gray-600 dark:text-gray-400">{item.benchmark}</p>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Usage Guidelines */}
      <Card className="border-blue-200 dark:border-blue-900">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="h-5 w-5 text-blue-600" />
            Clinical Usage Guidelines
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 text-sm">
            <div className="p-3 bg-blue-50 dark:bg-blue-950 rounded-lg">
              <p className="font-semibold text-blue-900 dark:text-blue-100 mb-1">
                ✓ Intended Use
              </p>
              <p className="text-blue-800 dark:text-blue-200">
                This AI model is designed to assist clinicians in detecting cancer from medical imaging. It should be used as a decision support tool, not as a replacement for professional medical judgment.
              </p>
            </div>
            <div className="p-3 bg-yellow-50 dark:bg-yellow-950 rounded-lg">
              <p className="font-semibold text-yellow-900 dark:text-yellow-100 mb-1">
                ⚠ Clinical Validation Required
              </p>
              <p className="text-yellow-800 dark:text-yellow-200">
                All AI predictions must be validated by qualified radiologists or oncologists. The model's predictions should complement, not replace, clinical expertise and patient history.
              </p>
            </div>
            <div className="p-3 bg-purple-50 dark:bg-purple-950 rounded-lg">
              <p className="font-semibold text-purple-900 dark:text-purple-100 mb-1">
                📋 Regulatory Status
              </p>
              <p className="text-purple-800 dark:text-purple-200">
                This model is for research and clinical decision support purposes. Users must ensure compliance with local regulations and institutional review board requirements.
              </p>
            </div>
            <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg">
              <p className="font-semibold text-green-900 dark:text-green-100 mb-1">
                🔒 Data Privacy
              </p>
              <p className="text-green-800 dark:text-green-200">
                All patient data is processed in compliance with HIPAA regulations. Images are encrypted in transit and at rest, and no patient data is stored beyond the inference session.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Citations and References */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BookOpen className="h-5 w-5 text-primary" />
            Citations & References
          </CardTitle>
          <CardDescription>Research papers and datasets used in model development</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 text-sm">
            <div className="p-3 border rounded-lg">
              <p className="font-semibold mb-1">EfficientNetV2: Smaller Models and Faster Training</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Tan, M., & Le, Q. (2021). International Conference on Machine Learning (ICML)
              </p>
            </div>
            <div className="p-3 border rounded-lg">
              <p className="font-semibold mb-1">An Image is Worth 16x16 Words: Transformers for Image Recognition</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Dosovitskiy, A., et al. (2021). International Conference on Learning Representations (ICLR)
              </p>
            </div>
            <div className="p-3 border rounded-lg">
              <p className="font-semibold mb-1">The Cancer Imaging Archive (TCIA)</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Clark, K., et al. (2013). Journal of Digital Imaging - Public dataset repository
              </p>
            </div>
            <div className="p-3 border rounded-lg">
              <p className="font-semibold mb-1">Grad-CAM: Visual Explanations from Deep Networks</p>
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Selvaraju, R. R., et al. (2017). IEEE International Conference on Computer Vision (ICCV)
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
