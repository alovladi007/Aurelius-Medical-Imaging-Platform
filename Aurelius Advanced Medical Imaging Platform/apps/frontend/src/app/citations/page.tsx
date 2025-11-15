"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Quote,
  Search,
  BookOpen,
  FileText,
  CheckCircle,
  AlertCircle,
  ExternalLink,
  Code,
  Anchor,
  Package
} from 'lucide-react';

export default function CitationsPage() {
  const [searchQuery, setSearchQuery] = useState('Anticoagulation for atrial fibrillation in CKD');
  const [selectedCitation, setSelectedCitation] = useState<any>(null);

  const stats = {
    totalGuidelines: 230,
    versionedDocs: 189,
    avgAnchorPrecision: '98.7%',
    chunkedDocs: 12400,
    citationCoverage: '100%'
  };

  // Anchored retrieval results with version@line references
  const anchoredResults = [
    {
      id: 'ANCHOR-001',
      guideline: '2024 ACC/AHA Atrial Fibrillation Guidelines',
      version: 'v2024.1',
      lineNumber: 1247,
      excerpt: 'For patients with AF and moderate-to-severe CKD (eGFR 15-49 mL/min/1.73m²), NOACs are recommended over warfarin with dose adjustments.',
      context: {
        before: 'Anticoagulation in Special Populations\n\nRenal Impairment:\n',
        match: 'For patients with AF and moderate-to-severe CKD (eGFR 15-49 mL/min/1.73m²), NOACs are recommended over warfarin with dose adjustments.',
        after: '\n\nDose Reduction Criteria:\n- Apixaban 2.5mg BID if 2+ criteria (age ≥80, weight ≤60kg, Cr ≥1.5)'
      },
      evidenceLevel: 'Class I, Level A',
      page: 47,
      section: '4.2 Anticoagulation in Renal Impairment',
      anchored: true,
      versionable: true
    },
    {
      id: 'ANCHOR-002',
      guideline: 'KDIGO 2021 CKD Guidelines',
      version: 'v2021.3',
      lineNumber: 2894,
      excerpt: 'Avoid NOACs if eGFR <15 mL/min/1.73m². Consider warfarin for ESRD on dialysis with close INR monitoring.',
      context: {
        before: 'Anticoagulation Recommendations by CKD Stage\n\nStage G5 (eGFR <15):\n',
        match: 'Avoid NOACs if eGFR <15 mL/min/1.73m². Consider warfarin for ESRD on dialysis with close INR monitoring.',
        after: '\n\nRationale: Insufficient data on NOAC safety in advanced CKD. Warfarin cleared independently of kidneys.'
      },
      evidenceLevel: 'Class IIa, Level B',
      page: 112,
      section: '8.3 Anticoagulation in Advanced CKD',
      anchored: true,
      versionable: true
    }
  ];

  // Chunked retrieval results (traditional RAG)
  const chunkedResults = [
    {
      id: 'CHUNK-001',
      guideline: '2024 ACC/AHA Atrial Fibrillation Guidelines',
      chunkId: 'chunk_234_768',
      excerpt: 'Anticoagulation therapy is recommended for stroke prevention in patients with atrial fibrillation. NOACs are generally preferred over warfarin in patients with moderate CKD. Dose adjustments are necessary based on renal function.',
      score: 0.87,
      chunkSize: 512,
      overlaps: true,
      contextLoss: 'Moderate',
      anchored: false,
      versionable: false
    },
    {
      id: 'CHUNK-002',
      guideline: 'ESC 2020 AF Management Guidelines',
      chunkId: 'chunk_891_256',
      excerpt: 'In patients with chronic kidney disease, anticoagulation strategies must be individualized. Renal function assessment is critical for dosing.',
      score: 0.73,
      chunkSize: 512,
      overlaps: true,
      contextLoss: 'High',
      anchored: false,
      versionable: false
    }
  ];

  const comparisonMetrics = {
    anchored: {
      precision: 98.7,
      recall: 94.3,
      exactness: 100,
      versionTracking: 'Full',
      lineNumber: 'Yes',
      contextWindow: 'Variable',
      citationFormat: 'guideline@version:line'
    },
    chunked: {
      precision: 82.4,
      recall: 89.1,
      exactness: 67,
      versionTracking: 'None',
      lineNumber: 'No',
      contextWindow: 'Fixed (512 tokens)',
      citationFormat: 'guideline + chunk_id'
    }
  };

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Quote className="h-8 w-8 text-primary" />
            Citation Systems Comparison
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Anchored (version@line) vs. Chunked retrieval comparison
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <BookOpen className="h-4 w-4 mr-2" />
            View Guidelines
          </Button>
          <Button>
            <Code className="h-4 w-4 mr-2" />
            API Docs
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Total Guidelines</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-primary">{stats.totalGuidelines}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Versioned Docs</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-blue-600">{stats.versionedDocs}</p>
            <p className="text-xs text-gray-600">Anchored</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Anchor Precision</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-green-600">{stats.avgAnchorPrecision}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Chunked Docs</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-purple-600">{stats.chunkedDocs.toLocaleString()}</p>
            <p className="text-xs text-gray-600">Traditional RAG</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Citation Coverage</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-orange-600">{stats.citationCoverage}</p>
            <p className="text-xs text-gray-600">All claims</p>
          </CardContent>
        </Card>
      </div>

      {/* Search */}
      <Card>
        <CardHeader>
          <CardTitle>Search Query</CardTitle>
          <CardDescription>Compare citation quality across both methods</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Enter clinical query..."
                className="pl-9"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Button>
              <Search className="h-4 w-4 mr-2" />
              Search Both
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Comparison Tabs */}
      <Tabs defaultValue="side-by-side" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="side-by-side">Side-by-Side</TabsTrigger>
          <TabsTrigger value="metrics">Metrics Comparison</TabsTrigger>
          <TabsTrigger value="technical">Technical Details</TabsTrigger>
        </TabsList>

        <TabsContent value="side-by-side" className="space-y-4 mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Anchored Results */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-4">
                <Anchor className="h-5 w-5 text-green-600" />
                <h2 className="text-xl font-bold">Anchored Retrieval</h2>
                <Badge className="bg-green-600">Recommended</Badge>
              </div>

              {anchoredResults.map((result) => (
                <Card key={result.id} className="border-green-200 dark:border-green-900">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <Badge variant="outline">v{result.version}</Badge>
                          <Badge variant="secondary" className="text-xs">
                            Line {result.lineNumber}
                          </Badge>
                          <Badge className="bg-green-600 text-xs">
                            {result.evidenceLevel}
                          </Badge>
                        </div>
                        <CardTitle className="text-base">{result.guideline}</CardTitle>
                        <CardDescription className="mt-1">
                          {result.section} • Page {result.page}
                        </CardDescription>
                      </div>
                      <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    {/* Context Window */}
                    <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg mb-3 text-sm font-mono">
                      <div className="text-gray-500 dark:text-gray-500 text-xs mb-1">
                        {result.context.before}
                      </div>
                      <div className="bg-yellow-100 dark:bg-yellow-900 p-2 rounded my-1">
                        {result.context.match}
                      </div>
                      <div className="text-gray-500 dark:text-gray-500 text-xs mt-1">
                        {result.context.after}
                      </div>
                    </div>

                    {/* Citation Format */}
                    <div className="p-2 bg-blue-50 dark:bg-blue-950 rounded border border-blue-200 dark:border-blue-900 text-xs font-mono mb-3">
                      <Quote className="h-3 w-3 inline mr-1" />
                      {result.guideline}@{result.version}:{result.lineNumber}
                    </div>

                    {/* Features */}
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="outline" className="bg-green-50 text-green-800 dark:bg-green-950 text-xs">
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Version Tracked
                      </Badge>
                      <Badge variant="outline" className="bg-green-50 text-green-800 dark:bg-green-950 text-xs">
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Exact Line
                      </Badge>
                      <Badge variant="outline" className="bg-green-50 text-green-800 dark:bg-green-950 text-xs">
                        <CheckCircle className="h-3 w-3 mr-1" />
                        Full Context
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {/* Chunked Results */}
            <div className="space-y-4">
              <div className="flex items-center gap-2 mb-4">
                <Package className="h-5 w-5 text-orange-600" />
                <h2 className="text-xl font-bold">Chunked Retrieval</h2>
                <Badge variant="outline">Traditional RAG</Badge>
              </div>

              {chunkedResults.map((result) => (
                <Card key={result.id} className="border-orange-200 dark:border-orange-900">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <Badge variant="outline">{result.chunkId}</Badge>
                          <Badge variant="secondary" className="text-xs">
                            Score: {(result.score * 100).toFixed(0)}%
                          </Badge>
                          <Badge variant="outline" className="bg-orange-50 text-orange-800 dark:bg-orange-950 text-xs">
                            {result.contextLoss} Context Loss
                          </Badge>
                        </div>
                        <CardTitle className="text-base">{result.guideline}</CardTitle>
                        <CardDescription className="mt-1">
                          Chunk size: {result.chunkSize} tokens
                        </CardDescription>
                      </div>
                      <AlertCircle className="h-5 w-5 text-orange-600 flex-shrink-0" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    {/* Chunk Content */}
                    <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded-lg mb-3 text-sm">
                      <div className="bg-yellow-100 dark:bg-yellow-900 p-2 rounded">
                        {result.excerpt}
                      </div>
                    </div>

                    {/* Citation Format */}
                    <div className="p-2 bg-orange-50 dark:bg-orange-950 rounded border border-orange-200 dark:border-orange-900 text-xs font-mono mb-3">
                      <Quote className="h-3 w-3 inline mr-1" />
                      {result.guideline} + {result.chunkId}
                    </div>

                    {/* Limitations */}
                    <div className="flex flex-wrap gap-2">
                      <Badge variant="outline" className="bg-red-50 text-red-800 dark:bg-red-950 text-xs">
                        <AlertCircle className="h-3 w-3 mr-1" />
                        No Version
                      </Badge>
                      <Badge variant="outline" className="bg-red-50 text-red-800 dark:bg-red-950 text-xs">
                        <AlertCircle className="h-3 w-3 mr-1" />
                        No Line Number
                      </Badge>
                      <Badge variant="outline" className="bg-orange-50 text-orange-800 dark:bg-orange-950 text-xs">
                        <AlertCircle className="h-3 w-3 mr-1" />
                        Fixed Window
                      </Badge>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="metrics" className="mt-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Anchored Metrics */}
            <Card className="border-green-200 dark:border-green-900">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Anchor className="h-5 w-5 text-green-600" />
                  Anchored Retrieval Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg text-center">
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Precision</p>
                      <p className="text-3xl font-bold text-green-600">{comparisonMetrics.anchored.precision}%</p>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg text-center">
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Recall</p>
                      <p className="text-3xl font-bold text-green-600">{comparisonMetrics.anchored.recall}%</p>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg text-center">
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Exactness</p>
                      <p className="text-3xl font-bold text-green-600">{comparisonMetrics.anchored.exactness}%</p>
                    </div>
                    <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg text-center">
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Version Tracking</p>
                      <p className="text-xl font-bold text-green-600">{comparisonMetrics.anchored.versionTracking}</p>
                    </div>
                  </div>

                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded">
                      <span>Line Number Support:</span>
                      <span className="font-semibold text-green-600">{comparisonMetrics.anchored.lineNumber}</span>
                    </div>
                    <div className="flex justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded">
                      <span>Context Window:</span>
                      <span className="font-semibold">{comparisonMetrics.anchored.contextWindow}</span>
                    </div>
                    <div className="flex justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded">
                      <span>Citation Format:</span>
                      <span className="font-mono text-xs">{comparisonMetrics.anchored.citationFormat}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Chunked Metrics */}
            <Card className="border-orange-200 dark:border-orange-900">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Package className="h-5 w-5 text-orange-600" />
                  Chunked Retrieval Metrics
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-orange-50 dark:bg-orange-950 rounded-lg text-center">
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Precision</p>
                      <p className="text-3xl font-bold text-orange-600">{comparisonMetrics.chunked.precision}%</p>
                    </div>
                    <div className="p-4 bg-orange-50 dark:bg-orange-950 rounded-lg text-center">
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Recall</p>
                      <p className="text-3xl font-bold text-orange-600">{comparisonMetrics.chunked.recall}%</p>
                    </div>
                    <div className="p-4 bg-orange-50 dark:bg-orange-950 rounded-lg text-center">
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Exactness</p>
                      <p className="text-3xl font-bold text-orange-600">{comparisonMetrics.chunked.exactness}%</p>
                    </div>
                    <div className="p-4 bg-orange-50 dark:bg-orange-950 rounded-lg text-center">
                      <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Version Tracking</p>
                      <p className="text-xl font-bold text-orange-600">{comparisonMetrics.chunked.versionTracking}</p>
                    </div>
                  </div>

                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded">
                      <span>Line Number Support:</span>
                      <span className="font-semibold text-red-600">{comparisonMetrics.chunked.lineNumber}</span>
                    </div>
                    <div className="flex justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded">
                      <span>Context Window:</span>
                      <span className="font-semibold">{comparisonMetrics.chunked.contextWindow}</span>
                    </div>
                    <div className="flex justify-between p-2 bg-gray-50 dark:bg-gray-900 rounded">
                      <span>Citation Format:</span>
                      <span className="font-mono text-xs">{comparisonMetrics.chunked.citationFormat}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="technical" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Technical Implementation Details</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <Anchor className="h-4 w-4 text-green-600" />
                    Anchored Retrieval Architecture
                  </h3>
                  <div className="space-y-2 text-sm">
                    <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                      <p className="font-semibold mb-1">Document Processing</p>
                      <ul className="list-disc list-inside text-xs space-y-1 text-gray-700 dark:text-gray-300">
                        <li>Guidelines parsed with line-level granularity</li>
                        <li>Version control via Git (SHA + timestamp)</li>
                        <li>Each line indexed with surrounding context</li>
                        <li>Embeddings created at sentence level</li>
                      </ul>
                    </div>
                    <div className="p-3 bg-green-50 dark:bg-green-950 rounded-lg">
                      <p className="font-semibold mb-1">Retrieval Process</p>
                      <ul className="list-disc list-inside text-xs space-y-1 text-gray-700 dark:text-gray-300">
                        <li>Vector search returns exact line matches</li>
                        <li>Context dynamically expanded around match</li>
                        <li>Citation includes guideline@version:line</li>
                        <li>Verifiable back to source document</li>
                      </ul>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="font-semibold mb-3 flex items-center gap-2">
                    <Package className="h-4 w-4 text-orange-600" />
                    Chunked Retrieval Architecture
                  </h3>
                  <div className="space-y-2 text-sm">
                    <div className="p-3 bg-orange-50 dark:bg-orange-950 rounded-lg">
                      <p className="font-semibold mb-1">Document Processing</p>
                      <ul className="list-disc list-inside text-xs space-y-1 text-gray-700 dark:text-gray-300">
                        <li>Fixed-size chunks (512 tokens)</li>
                        <li>Overlap between chunks (64 tokens)</li>
                        <li>No version tracking</li>
                        <li>Chunk-level embeddings</li>
                      </ul>
                    </div>
                    <div className="p-3 bg-orange-50 dark:bg-orange-950 rounded-lg">
                      <p className="font-semibold mb-1">Retrieval Process</p>
                      <ul className="list-disc list-inside text-xs space-y-1 text-gray-700 dark:text-gray-300">
                        <li>Vector search returns chunk IDs</li>
                        <li>Fixed context window (chunk boundaries)</li>
                        <li>Citation includes guideline + chunk_id</li>
                        <li>Difficult to verify exact source</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
}
