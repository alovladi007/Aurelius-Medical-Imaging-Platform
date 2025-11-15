"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Input } from '@/components/ui/input';
import {
  GitBranch,
  Brain,
  FileText,
  Search,
  TrendingUp,
  Clock,
  Users,
  Activity,
  Zap,
  Database,
  Network
} from 'lucide-react';

export default function Layer2Page() {
  const [graphStats, setGraphStats] = useState({
    totalNodes: 12e6,
    totalEdges: 45e6,
    patients: 125000,
    encounters: 890000,
    problems: 2.1e6,
    medications: 3.5e6,
    labs: 8.2e6
  });

  const ontologies = [
    { name: 'SNOMED CT', concepts: '350K+', mappings: 'ICD-10, LOINC', status: 'active' },
    { name: 'LOINC', concepts: '95K+', mappings: 'CPT, local codes', status: 'active' },
    { name: 'RxNorm', concepts: '140K+', mappings: 'ATC, NDC', status: 'active' },
    { name: 'ICD-10-CM', concepts: '72K+', mappings: 'SNOMED CT', status: 'active' },
    { name: 'CPT', concepts: '10K+', mappings: 'LOINC', status: 'active' },
    { name: 'Gene Ontology', concepts: '45K+', mappings: 'Disease Ontology', status: 'active' }
  ];

  const reasoningEngines = [
    { name: 'FHIR CQL Engine', purpose: 'Clinical quality measures', status: 'running', queries: 124 },
    { name: 'Drools Rules', purpose: 'Clinical decision support', status: 'running', rules: 450 },
    { name: 'Causal Graph Analyzer', purpose: 'Risk pathways', status: 'running', graphs: 78 },
    { name: 'Counterfactual Simulator', purpose: 'What-if therapy changes', status: 'running', simulations: 23 }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <GitBranch className="h-8 w-8 text-primary" />
            Layer 2: Clinical Knowledge Graph
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Patient-centric temporal graph with biomedical reasoning
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Search className="h-4 w-4 mr-2" />
            Query Graph
          </Button>
          <Button>
            <Brain className="h-4 w-4 mr-2" />
            Run Reasoning
          </Button>
        </div>
      </div>

      {/* Graph Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Total Nodes</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{formatNumber(graphStats.totalNodes)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Total Edges</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{formatNumber(graphStats.totalEdges)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Patients</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{formatNumber(graphStats.patients)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Encounters</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{formatNumber(graphStats.encounters)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Problems</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{formatNumber(graphStats.problems)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Medications</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{formatNumber(graphStats.medications)}</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Lab Results</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold">{formatNumber(graphStats.labs)}</p>
          </CardContent>
        </Card>
      </div>

      {/* Graph Structure */}
      <Card>
        <CardHeader>
          <CardTitle>Graph Structure</CardTitle>
          <CardDescription>Patient-centric temporal model</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
              <h4 className="font-semibold mb-2">Core Entities</h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                <div><strong>Patient</strong> → Demographics, identifiers</div>
                <div><strong>Encounter</strong> → Visits, admissions</div>
                <div><strong>Problem</strong> → Diagnoses, conditions</div>
                <div><strong>Medication</strong> → Prescriptions, administrations</div>
                <div><strong>Lab</strong> → Test results, panels</div>
                <div><strong>Imaging</strong> → Studies, reports</div>
                <div><strong>Procedure</strong> → Surgical, interventional</div>
                <div><strong>Vital</strong> → BP, HR, temp, SpO2</div>
              </div>
            </div>
            <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg">
              <h4 className="font-semibold mb-2">Temporal Relationships</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                All entities linked with temporal edges (BEFORE, DURING, AFTER, CONCURRENT) enabling temporal reasoning and queries
              </p>
            </div>
            <div className="p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
              <h4 className="font-semibold mb-2">Genomic Integration</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Gene → Variant → Phenotype → Disease linkages with evidence scores
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Ontology Hub */}
      <Card>
        <CardHeader>
          <CardTitle>Ontology Hub</CardTitle>
          <CardDescription>Bi-directional terminology mappings</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {ontologies.map((onto, index) => (
              <div key={index} className="p-4 border rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <p className="font-semibold">{onto.name}</p>
                  <Badge variant="default">{onto.status}</Badge>
                </div>
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                  <strong>Concepts:</strong> {onto.concepts}
                </p>
                <p className="text-xs text-gray-500">
                  Maps to: {onto.mappings}
                </p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Reasoning Engines */}
      <Card>
        <CardHeader>
          <CardTitle>Reasoning & Inference</CardTitle>
          <CardDescription>Rule engines, causal graphs, counterfactual simulators</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {reasoningEngines.map((engine, index) => (
              <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-purple-100 dark:bg-purple-950 rounded-full flex items-center justify-center">
                    <Brain className="h-5 w-5 text-purple-600" />
                  </div>
                  <div>
                    <p className="font-semibold">{engine.name}</p>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{engine.purpose}</p>
                  </div>
                </div>
                <div className="flex items-center gap-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold">
                      {engine.queries || engine.rules || engine.graphs || engine.simulations}
                    </p>
                    <p className="text-xs text-gray-600">
                      {engine.queries ? 'Queries' : engine.rules ? 'Rules' : engine.graphs ? 'Graphs' : 'Simulations'}
                    </p>
                  </div>
                  <Badge variant="default" className="bg-purple-600">
                    <Activity className="h-3 w-3 mr-1" />
                    {engine.status}
                  </Badge>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Query Interface */}
      <Card>
        <CardHeader>
          <CardTitle>Graph Query Interface</CardTitle>
          <CardDescription>Cypher-like queries for clinical cohorts</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="builder">
            <TabsList>
              <TabsTrigger value="builder">Query Builder</TabsTrigger>
              <TabsTrigger value="cypher">Cypher Query</TabsTrigger>
              <TabsTrigger value="examples">Examples</TabsTrigger>
            </TabsList>

            <TabsContent value="builder" className="space-y-4">
              <div className="space-y-3">
                <div>
                  <label className="text-sm font-medium mb-2 block">Find Patients</label>
                  <Input placeholder="With diagnosis of..." />
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">In Time Period</label>
                  <div className="grid grid-cols-2 gap-3">
                    <Input type="date" />
                    <Input type="date" />
                  </div>
                </div>
                <div>
                  <label className="text-sm font-medium mb-2 block">With Lab Results</label>
                  <Input placeholder="LOINC code or name..." />
                </div>
                <Button className="w-full">
                  <Search className="h-4 w-4 mr-2" />
                  Build Cohort
                </Button>
              </div>
            </TabsContent>

            <TabsContent value="cypher">
              <div className="space-y-3">
                <div className="font-mono text-sm p-4 bg-gray-900 text-green-400 rounded">
                  MATCH (p:Patient)-[:HAS_PROBLEM]-&gt;(prob:Problem)<br />
                  WHERE prob.code = 'SNOMED:73211009'<br />
                  AND p.age &gt; 50<br />
                  RETURN p, COUNT(prob) as problem_count<br />
                  ORDER BY problem_count DESC
                </div>
                <Button className="w-full">Execute Query</Button>
              </div>
            </TabsContent>

            <TabsContent value="examples">
              <div className="space-y-2 text-sm">
                <div className="p-3 border rounded">
                  <strong>Diabetes Cohort:</strong> Patients with HbA1c &gt; 6.5% in last 6 months
                </div>
                <div className="p-3 border rounded">
                  <strong>Heart Failure with Meds:</strong> HF diagnosis + ACE-I or ARB within 30 days
                </div>
                <div className="p-3 border rounded">
                  <strong>Cancer Surveillance:</strong> CA diagnosis + imaging studies every 3 months
                </div>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>

      {/* Guideline Rules */}
      <Card>
        <CardHeader>
          <CardTitle>Clinical Guideline Rules</CardTitle>
          <CardDescription>CQL/GL rules and trial eligibility templates</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm">
            <div className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-950 rounded">
              <span>Diabetes Care Measures (HEDIS)</span>
              <Badge>87 rules</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-950 rounded">
              <span>Heart Failure Guidelines (ACC/AHA)</span>
              <Badge>124 rules</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-purple-50 dark:bg-purple-950 rounded">
              <span>Cancer Screening (USPSTF)</span>
              <Badge>56 rules</Badge>
            </div>
            <div className="flex items-center justify-between p-3 bg-orange-50 dark:bg-orange-950 rounded">
              <span>Clinical Trial Eligibility</span>
              <Badge>183 templates</Badge>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function formatNumber(num: number): string {
  if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
  if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
  return num.toString();
}
