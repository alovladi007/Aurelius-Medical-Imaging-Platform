"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Search,
  BookOpen,
  FileText,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  ExternalLink,
  Filter,
  Download,
  Brain,
  Zap,
  Library,
  Quote,
  Star,
  Clock
} from 'lucide-react';

export default function RAGRetrievalPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedDocument, setSelectedDocument] = useState<any>(null);

  const stats = {
    documents: '12.4K',
    policies: 487,
    guidelines: 231,
    avgRelevance: '94.2%',
    avgTime: '0.3s'
  };

  const exampleQueries = [
    'Anticoagulation for atrial fibrillation in CKD',
    'Sepsis bundle compliance requirements',
    'Pneumonia empiric antibiotic selection',
    'Insulin sliding scale protocols',
    'VTE prophylaxis in surgery patients'
  ];

  const retrievedDocuments = [
    {
      id: 'DOC-001',
      title: '2024 ACC/AHA Guideline for the Management of Atrial Fibrillation',
      type: 'Clinical Guideline',
      source: 'American College of Cardiology',
      date: '2024-01-15',
      relevanceScore: 0.96,
      evidenceLevel: 'Level A (High)',
      snippet: 'For patients with AF and moderate-to-severe CKD (eGFR 15-49 mL/min/1.73m²), NOACs are recommended over warfarin with dose adjustments: Apixaban 2.5mg BID if 2+ criteria (age ≥80, weight ≤60kg, Cr ≥1.5); Rivaroxaban 15mg daily; Edoxaban 30mg daily. Avoid dabigatran if eGFR <30.',
      sections: [
        {
          heading: 'Anticoagulation in Renal Impairment',
          content: 'Dose reduction of NOACs is required for moderate-to-severe CKD to balance efficacy and bleeding risk. Regular monitoring of renal function is essential...',
          page: 47
        },
        {
          heading: 'CHA₂DS₂-VASc Score Application',
          content: 'Anticoagulation is recommended for CHA₂DS₂-VASc ≥2 in men or ≥3 in women. CKD itself increases stroke risk independently...',
          page: 32
        }
      ],
      citations: 23,
      downloads: 4521,
      lastUpdated: '2024-01-15'
    },
    {
      id: 'DOC-002',
      title: 'Institutional Policy: Anticoagulation Management in Special Populations',
      type: 'Hospital Policy',
      source: 'Massachusetts General Hospital',
      date: '2023-11-01',
      relevanceScore: 0.93,
      evidenceLevel: 'Institutional',
      snippet: 'For AF patients with eGFR 30-50, preferred agents are apixaban (2.5mg BID) or rivaroxaban (15mg daily). Requires pharmacy consult if eGFR <30. Contraindications: active bleeding, platelets <50K, recent neurosurgery (<7 days). Anti-Xa monitoring NOT routinely recommended but consider in extremes of weight or renal dysfunction.',
      sections: [
        {
          heading: 'NOAC Dosing Table',
          content: 'Comprehensive table of NOAC dosing adjustments by indication and renal function...',
          page: 3
        },
        {
          heading: 'Monitoring Requirements',
          content: 'Check CBC, CMP at baseline and q3-6 months. More frequent if eGFR trending down...',
          page: 5
        }
      ],
      citations: 8,
      downloads: 1247,
      lastUpdated: '2023-11-01'
    },
    {
      id: 'DOC-003',
      title: 'Direct Oral Anticoagulants in Chronic Kidney Disease: A Meta-Analysis',
      type: 'Research Article',
      source: 'JAMA Cardiology',
      date: '2023-08-22',
      relevanceScore: 0.89,
      evidenceLevel: 'Level A (Meta-analysis)',
      snippet: 'Meta-analysis of 12 RCTs (n=94,656) comparing NOACs vs warfarin in CKD. NOACs reduced stroke/SE by 19% (HR 0.81, 95% CI 0.72-0.91) and major bleeding by 14% (HR 0.86, 0.79-0.94) vs warfarin in CKD stages 3-4. Benefits consistent across eGFR 30-60. Apixaban showed most favorable bleeding profile.',
      sections: [
        {
          heading: 'Results by eGFR Subgroup',
          content: 'eGFR 30-49: NOAC superiority maintained. eGFR 15-29: Limited data, use with caution...',
          page: 5
        },
        {
          heading: 'Safety Outcomes',
          content: 'Major bleeding events were lower with NOACs across all CKD stages analyzed...',
          page: 7
        }
      ],
      citations: 142,
      downloads: 8934,
      lastUpdated: '2023-08-22'
    },
    {
      id: 'DOC-004',
      title: 'CQL Clinical Quality Measure: Anticoagulation for AF',
      type: 'CQL Policy',
      source: 'CMS eCQM',
      date: '2024-01-01',
      relevanceScore: 0.87,
      evidenceLevel: 'Quality Measure',
      snippet: 'Measure: Percentage of AF patients with CHA₂DS₂-VASc ≥2 (men) or ≥3 (women) prescribed anticoagulation. Exclusions: mechanical valve, mitral stenosis, hospice, pregnancy, ESRD on dialysis. Numerator: Prescription of warfarin OR NOAC with appropriate dosing. Target: >90% compliance.',
      sections: [
        {
          heading: 'Measure Logic',
          content: 'CQL expression defining eligible population, exclusions, and numerator criteria...',
          page: 1
        },
        {
          heading: 'Data Elements Required',
          content: 'ICD-10: I48.x (AF), medications: warfarin, apixaban, rivaroxaban, edoxaban, dabigatran...',
          page: 2
        }
      ],
      citations: 5,
      downloads: 672,
      lastUpdated: '2024-01-01'
    },
    {
      id: 'DOC-005',
      title: 'Kidney Disease: Improving Global Outcomes (KDIGO) 2021 Guidelines',
      type: 'Clinical Guideline',
      source: 'KDIGO',
      date: '2021-10-01',
      relevanceScore: 0.84,
      evidenceLevel: 'Level B (Moderate)',
      snippet: 'For CKD patients requiring anticoagulation: (1) Calculate CrCl using Cockcroft-Gault for drug dosing decisions, (2) Avoid NOACs if eGFR <15, (3) Consider warfarin for ESRD on dialysis with close INR monitoring, (4) Reassess bleeding risk using HAS-BLED score, (5) Shared decision-making for eGFR 15-30.',
      sections: [
        {
          heading: 'Anticoagulation Recommendations',
          content: 'Detailed recommendations for anticoagulation across CKD stages G1-G5...',
          page: 112
        }
      ],
      citations: 87,
      downloads: 5621,
      lastUpdated: '2021-10-01'
    }
  ];

  const filters = [
    { name: 'Clinical Guidelines', count: 231, active: true },
    { name: 'Hospital Policies', count: 487, active: false },
    { name: 'Research Articles', count: 8942, active: false },
    { name: 'CQL Policies', count: 156, active: false },
    { name: 'Drug Monographs', count: 2634, active: false }
  ];

  const semanticConcepts = [
    { concept: 'Anticoagulation', relevance: 0.98 },
    { concept: 'Chronic Kidney Disease', relevance: 0.96 },
    { concept: 'Atrial Fibrillation', relevance: 0.95 },
    { concept: 'Direct Oral Anticoagulants (NOACs)', relevance: 0.94 },
    { concept: 'Stroke Prevention', relevance: 0.89 },
    { concept: 'Bleeding Risk', relevance: 0.87 },
    { concept: 'Dose Adjustment', relevance: 0.85 }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Library className="h-8 w-8 text-primary" />
            RAG Knowledge Retrieval
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Policy-aware semantic search across guidelines, policies, and literature
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Filter className="h-4 w-4 mr-2" />
            Advanced Filters
          </Button>
          <Button>
            <Download className="h-4 w-4 mr-2" />
            Export Results
          </Button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Total Documents</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-primary">{stats.documents}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Hospital Policies</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-blue-600">{stats.policies}</p>
            <p className="text-xs text-gray-600">Institutional</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Clinical Guidelines</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-green-600">{stats.guidelines}</p>
            <p className="text-xs text-gray-600">Evidence-based</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Avg Relevance</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-purple-600">{stats.avgRelevance}</p>
            <p className="text-xs text-gray-600">Semantic match</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Retrieval Time</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-3xl font-bold text-orange-600">{stats.avgTime}</p>
            <p className="text-xs text-gray-600">Vector search</p>
          </CardContent>
        </Card>
      </div>

      {/* Search Bar */}
      <Card>
        <CardHeader>
          <CardTitle>Semantic Search</CardTitle>
          <CardDescription>Ask clinical questions in natural language</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-2 mb-4">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-3 h-4 w-4 text-gray-400" />
              <Input
                placeholder="Search guidelines, policies, and literature..."
                className="pl-9"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Button>
              <Brain className="h-4 w-4 mr-2" />
              Search
            </Button>
          </div>

          {/* Example Queries */}
          <div>
            <p className="text-sm font-semibold mb-2">Example Queries:</p>
            <div className="flex flex-wrap gap-2">
              {exampleQueries.map((query, idx) => (
                <Button
                  key={idx}
                  variant="outline"
                  size="sm"
                  className="text-xs"
                  onClick={() => setSearchQuery(query)}
                >
                  {query}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Filters & Semantic Concepts */}
        <div className="space-y-6">
          {/* Filters */}
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Document Types</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {filters.map((filter, idx) => (
                  <div key={idx} className="flex items-center justify-between p-2 hover:bg-gray-50 dark:hover:bg-gray-900 rounded cursor-pointer">
                    <div className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={filter.active}
                        className="rounded"
                      />
                      <span className="text-sm">{filter.name}</span>
                    </div>
                    <Badge variant="secondary" className="text-xs">{filter.count}</Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Semantic Concepts */}
          <Card className="border-purple-200 dark:border-purple-900">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Brain className="h-5 w-5 text-purple-600" />
                Semantic Concepts
              </CardTitle>
              <CardDescription>Extracted from query</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {semanticConcepts.map((concept, idx) => (
                  <div key={idx} className="space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold">{concept.concept}</span>
                      <span className="text-xs text-purple-600">{Math.round(concept.relevance * 100)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-1.5">
                      <div
                        className="bg-purple-600 h-1.5 rounded-full"
                        style={{ width: `${concept.relevance * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Retrieved Documents */}
        <div className="lg:col-span-3 space-y-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-600">
              Found <span className="font-bold text-primary">{retrievedDocuments.length}</span> relevant documents
            </p>
            <div className="flex gap-2">
              <Button variant="outline" size="sm">
                <Clock className="h-3 w-3 mr-1" />
                Most Recent
              </Button>
              <Button variant="outline" size="sm">
                <TrendingUp className="h-3 w-3 mr-1" />
                Highest Relevance
              </Button>
            </div>
          </div>

          {retrievedDocuments.map((doc) => (
            <Card
              key={doc.id}
              className={`cursor-pointer transition-all ${
                selectedDocument?.id === doc.id ? 'border-primary border-2' : 'hover:shadow-lg'
              }`}
              onClick={() => setSelectedDocument(selectedDocument?.id === doc.id ? null : doc)}
            >
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <Badge variant="outline">{doc.type}</Badge>
                      <Badge
                        className={
                          doc.evidenceLevel.includes('Level A') ? 'bg-green-600' :
                          doc.evidenceLevel.includes('Level B') ? 'bg-blue-600' :
                          doc.evidenceLevel.includes('Meta-analysis') ? 'bg-purple-600' :
                          'bg-gray-600'
                        }
                      >
                        {doc.evidenceLevel}
                      </Badge>
                      <Badge variant="secondary" className="text-xs">
                        {doc.source}
                      </Badge>
                    </div>
                    <CardTitle className="text-lg mb-2">{doc.title}</CardTitle>
                    <div className="flex items-center gap-4 text-xs text-gray-600">
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {doc.date}
                      </span>
                      <span className="flex items-center gap-1">
                        <Quote className="h-3 w-3" />
                        {doc.citations} citations
                      </span>
                      <span className="flex items-center gap-1">
                        <Download className="h-3 w-3" />
                        {doc.downloads.toLocaleString()} downloads
                      </span>
                    </div>
                  </div>
                  <div className="ml-4 text-right">
                    <div className="w-16 h-16 rounded-full border-4 border-green-500 flex items-center justify-center">
                      <p className="text-xl font-bold text-green-600">{Math.round(doc.relevanceScore * 100)}</p>
                    </div>
                    <p className="text-xs text-gray-600 mt-1">Relevance</p>
                  </div>
                </div>
              </CardHeader>

              <CardContent>
                {/* Snippet */}
                <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg border border-blue-200 dark:border-blue-900 mb-4">
                  <div className="flex items-start gap-2">
                    <Quote className="h-4 w-4 text-blue-600 flex-shrink-0 mt-1" />
                    <div>
                      <p className="text-xs font-semibold mb-1">Key Excerpt:</p>
                      <p className="text-sm leading-relaxed">{doc.snippet}</p>
                    </div>
                  </div>
                </div>

                {/* Expanded Sections */}
                {selectedDocument?.id === doc.id && (
                  <div className="space-y-4 mt-4">
                    <Tabs defaultValue="sections" className="w-full">
                      <TabsList className="grid w-full grid-cols-3">
                        <TabsTrigger value="sections">Relevant Sections</TabsTrigger>
                        <TabsTrigger value="metadata">Metadata</TabsTrigger>
                        <TabsTrigger value="citations">Citations</TabsTrigger>
                      </TabsList>

                      <TabsContent value="sections" className="space-y-3 mt-4">
                        {doc.sections.map((section, idx) => (
                          <div key={idx} className="p-4 border rounded-lg">
                            <div className="flex items-center justify-between mb-2">
                              <p className="font-semibold text-sm">{section.heading}</p>
                              <Badge variant="outline" className="text-xs">Page {section.page}</Badge>
                            </div>
                            <p className="text-sm text-gray-700 dark:text-gray-300">{section.content}</p>
                            <Button variant="outline" size="sm" className="mt-2 text-xs">
                              <ExternalLink className="h-3 w-3 mr-1" />
                              View Full Section
                            </Button>
                          </div>
                        ))}
                      </TabsContent>

                      <TabsContent value="metadata" className="mt-4">
                        <div className="grid grid-cols-2 gap-4 p-4 border rounded-lg">
                          <div>
                            <p className="text-xs text-gray-600">Document ID</p>
                            <p className="font-semibold text-sm">{doc.id}</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Last Updated</p>
                            <p className="font-semibold text-sm">{doc.lastUpdated}</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Evidence Level</p>
                            <p className="font-semibold text-sm">{doc.evidenceLevel}</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Relevance Score</p>
                            <p className="font-semibold text-sm">{(doc.relevanceScore * 100).toFixed(1)}%</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Source Organization</p>
                            <p className="font-semibold text-sm">{doc.source}</p>
                          </div>
                          <div>
                            <p className="text-xs text-gray-600">Publication Date</p>
                            <p className="font-semibold text-sm">{doc.date}</p>
                          </div>
                        </div>
                      </TabsContent>

                      <TabsContent value="citations" className="mt-4">
                        <div className="p-4 border rounded-lg">
                          <div className="flex items-center gap-2 mb-3">
                            <Quote className="h-5 w-5 text-blue-600" />
                            <p className="font-semibold">Citation Information</p>
                          </div>
                          <div className="p-3 bg-gray-50 dark:bg-gray-900 rounded text-sm font-mono mb-3">
                            {doc.title}. {doc.source}, {doc.date}. Retrieved from institutional knowledge base.
                          </div>
                          <div className="flex gap-2">
                            <Button variant="outline" size="sm">
                              <Download className="h-3 w-3 mr-1" />
                              Export as RIS
                            </Button>
                            <Button variant="outline" size="sm">
                              <Download className="h-3 w-3 mr-1" />
                              Export as BibTeX
                            </Button>
                          </div>
                        </div>
                      </TabsContent>
                    </Tabs>

                    <div className="flex gap-2 mt-4">
                      <Button className="flex-1">
                        <FileText className="h-4 w-4 mr-2" />
                        View Full Document
                      </Button>
                      <Button variant="outline" className="flex-1">
                        <Star className="h-4 w-4 mr-2" />
                        Save to Library
                      </Button>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  );
}
