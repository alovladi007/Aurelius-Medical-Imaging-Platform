"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  TrendingUp,
  Users,
  Database,
  Code,
  FileText,
  Play,
  Download,
  Save,
  Brain,
  BookOpen,
  BarChart3,
  Filter,
  Search
} from 'lucide-react';

export default function ResearcherWorkbenchPage() {
  const [query, setQuery] = useState('');
  const [cohortSize, setCohortSize] = useState(0);

  const savedCohorts = [
    { id: 1, name: 'Diabetic Patients with HbA1c >9', size: 2341, created: '2025-11-10' },
    { id: 2, name: 'Post-MI patients on DAPT', size: 1876, created: '2025-11-08' },
    { id: 3, name: 'NSCLC Stage IIIB-IV EGFR+', size: 456, created: '2025-11-05' },
    { id: 4, name: 'Septic shock survivors', size: 789, created: '2025-11-01' }
  ];

  const recentQueries = [
    {
      query: 'SELECT COUNT(*) FROM patients WHERE diagnosis LIKE "%diabetes%" AND latest_hba1c > 9',
      result: '2,341 patients',
      time: '1.2s',
      timestamp: '2025-11-15 14:23'
    },
    {
      query: 'MATCH (p:Patient)-[:HAS_CONDITION]->(c:Condition {name: "Atrial Fibrillation"}) WHERE p.age > 65 RETURN COUNT(p)',
      result: '4,567 patients',
      time: '0.8s',
      timestamp: '2025-11-15 13:45'
    }
  ];

  const availableDatasets = [
    { name: 'De-identified EHR Data', records: 125000, tables: 15, lastUpdated: '2025-11-14' },
    { name: 'DICOM Image Archive', records: 890000, size: '24 TB', lastUpdated: '2025-11-15' },
    { name: 'Lab Results', records: 2400000, tables: 3, lastUpdated: '2025-11-15' },
    { name: 'Genomics Data', records: 15000, size: '3.2 TB', lastUpdated: '2025-11-10' }
  ];

  const analysisTemplates = [
    {
      name: 'Survival Analysis',
      description: 'Kaplan-Meier curves and Cox proportional hazards',
      language: 'R',
      icon: TrendingUp
    },
    {
      name: 'Cohort Discovery',
      description: 'Complex inclusion/exclusion criteria builder',
      language: 'SQL + Cypher',
      icon: Users
    },
    {
      name: 'Causal Inference',
      description: 'Propensity score matching and DAG analysis',
      language: 'Python',
      icon: Brain
    },
    {
      name: 'Literature Meta-Analysis',
      description: 'Automated literature search and synthesis',
      language: 'Python',
      icon: BookOpen
    }
  ];

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <TrendingUp className="h-8 w-8 text-primary" />
            Research Workbench
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Cohort discovery, causal analysis, literature triage, protocol drafting
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Save className="h-4 w-4 mr-2" />
            Save Workspace
          </Button>
          <Button>
            <Download className="h-4 w-4 mr-2" />
            Export Results
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Sidebar - Cohort Builder */}
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Cohort Builder</CardTitle>
              <CardDescription>Define inclusion/exclusion criteria</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <p className="text-sm font-semibold mb-2">Inclusion Criteria</p>
                <div className="space-y-2">
                  <div className="p-2 border rounded-lg flex items-center justify-between">
                    <span className="text-sm">Diagnosis: Diabetes</span>
                    <Button variant="ghost" size="sm">×</Button>
                  </div>
                  <div className="p-2 border rounded-lg flex items-center justify-between">
                    <span className="text-sm">HbA1c > 9%</span>
                    <Button variant="ghost" size="sm">×</Button>
                  </div>
                  <div className="p-2 border rounded-lg flex items-center justify-between">
                    <span className="text-sm">Age 18-75</span>
                    <Button variant="ghost" size="sm">×</Button>
                  </div>
                </div>
                <Button variant="outline" size="sm" className="w-full mt-2">
                  + Add Criterion
                </Button>
              </div>

              <div>
                <p className="text-sm font-semibold mb-2">Exclusion Criteria</p>
                <div className="space-y-2">
                  <div className="p-2 border rounded-lg flex items-center justify-between">
                    <span className="text-sm">Type 1 Diabetes</span>
                    <Button variant="ghost" size="sm">×</Button>
                  </div>
                  <div className="p-2 border rounded-lg flex items-center justify-between">
                    <span className="text-sm">ESRD on dialysis</span>
                    <Button variant="ghost" size="sm">×</Button>
                  </div>
                </div>
                <Button variant="outline" size="sm" className="w-full mt-2">
                  + Add Criterion
                </Button>
              </div>

              <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
                <p className="text-sm font-semibold mb-1">Estimated Cohort Size</p>
                <p className="text-3xl font-bold text-primary">2,341</p>
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">patients match criteria</p>
              </div>

              <Button className="w-full">
                <Play className="h-4 w-4 mr-2" />
                Execute Query
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Saved Cohorts</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {savedCohorts.map((cohort) => (
                  <div key={cohort.id} className="p-3 border rounded-lg hover:bg-gray-50 dark:hover:bg-gray-900 cursor-pointer transition-all">
                    <p className="font-semibold text-sm">{cohort.name}</p>
                    <div className="flex items-center justify-between mt-2 text-xs">
                      <span className="text-gray-600">n = {cohort.size.toLocaleString()}</span>
                      <span className="text-gray-500">{cohort.created}</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Workspace */}
        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Query Editor</CardTitle>
              <CardDescription>SQL, Cypher (Neo4j), or Python notebook access</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="sql">
                <TabsList>
                  <TabsTrigger value="sql">SQL</TabsTrigger>
                  <TabsTrigger value="cypher">Cypher (Graph)</TabsTrigger>
                  <TabsTrigger value="python">Python</TabsTrigger>
                </TabsList>

                <TabsContent value="sql" className="space-y-4 mt-4">
                  <div>
                    <textarea
                      className="w-full h-40 p-3 font-mono text-sm border rounded-lg bg-gray-50 dark:bg-gray-900"
                      placeholder="SELECT * FROM patients WHERE..."
                      defaultValue={`SELECT
  p.patient_id,
  p.age,
  p.gender,
  c.condition_name,
  l.hba1c_value
FROM patients p
JOIN conditions c ON p.patient_id = c.patient_id
JOIN labs l ON p.patient_id = l.patient_id
WHERE
  c.condition_name LIKE '%diabetes%'
  AND l.hba1c_value > 9
  AND p.age BETWEEN 18 AND 75
LIMIT 100;`}
                    />
                  </div>
                  <div className="flex gap-2">
                    <Button className="flex-1">
                      <Play className="h-4 w-4 mr-2" />
                      Run Query
                    </Button>
                    <Button variant="outline" className="flex-1">
                      <Save className="h-4 w-4 mr-2" />
                      Save Query
                    </Button>
                  </div>
                </TabsContent>

                <TabsContent value="cypher" className="space-y-4 mt-4">
                  <div>
                    <textarea
                      className="w-full h-40 p-3 font-mono text-sm border rounded-lg bg-gray-50 dark:bg-gray-900"
                      placeholder="MATCH (p:Patient)..."
                      defaultValue={`MATCH (p:Patient)-[:HAS_CONDITION]->(c:Condition)
WHERE c.name = "Atrial Fibrillation"
  AND p.age > 65
  AND NOT (p)-[:TAKES_MEDICATION]->(:Medication {name: "Warfarin"})
RETURN p.patient_id, p.age, p.gender
ORDER BY p.age DESC
LIMIT 100;`}
                    />
                  </div>
                  <div className="flex gap-2">
                    <Button className="flex-1">
                      <Play className="h-4 w-4 mr-2" />
                      Run Query
                    </Button>
                    <Button variant="outline" className="flex-1">
                      <BarChart3 className="h-4 w-4 mr-2" />
                      Visualize Graph
                    </Button>
                  </div>
                </TabsContent>

                <TabsContent value="python" className="space-y-4 mt-4">
                  <div>
                    <textarea
                      className="w-full h-40 p-3 font-mono text-sm border rounded-lg bg-gray-50 dark:bg-gray-900"
                      placeholder="import pandas as pd..."
                      defaultValue={`import pandas as pd
from lifelines import KaplanMeierFitter

# Load cohort data
cohort = pd.read_sql(query, conn)

# Survival analysis
kmf = KaplanMeierFitter()
kmf.fit(cohort['duration'], cohort['event'])

# Plot survival curve
kmf.plot_survival_function()
plt.title('Survival in Diabetic Cohort')
plt.show()`}
                    />
                  </div>
                  <div className="flex gap-2">
                    <Button className="flex-1">
                      <Play className="h-4 w-4 mr-2" />
                      Run Cell
                    </Button>
                    <Button variant="outline" className="flex-1">
                      <FileText className="h-4 w-4 mr-2" />
                      Open Jupyter
                    </Button>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Results */}
          <Card>
            <CardHeader>
              <CardTitle>Query Results</CardTitle>
              <CardDescription>2,341 rows returned in 1.2s</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left p-2">Patient ID</th>
                      <th className="text-left p-2">Age</th>
                      <th className="text-left p-2">Gender</th>
                      <th className="text-left p-2">Condition</th>
                      <th className="text-left p-2">HbA1c</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { id: 'P-001', age: 64, gender: 'M', condition: 'Type 2 Diabetes', hba1c: 9.2 },
                      { id: 'P-002', age: 58, gender: 'F', condition: 'Type 2 Diabetes', hba1c: 10.1 },
                      { id: 'P-003', age: 72, gender: 'M', condition: 'Type 2 Diabetes', hba1c: 9.8 },
                      { id: 'P-004', age: 51, gender: 'F', condition: 'Type 2 Diabetes', hba1c: 11.3 },
                      { id: 'P-005', age: 69, gender: 'M', condition: 'Type 2 Diabetes', hba1c: 9.5 }
                    ].map((row, idx) => (
                      <tr key={idx} className="border-b hover:bg-gray-50 dark:hover:bg-gray-900">
                        <td className="p-2 font-mono">{row.id}</td>
                        <td className="p-2">{row.age}</td>
                        <td className="p-2">{row.gender}</td>
                        <td className="p-2">{row.condition}</td>
                        <td className="p-2 font-semibold text-red-600">{row.hba1c}%</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="mt-4 text-center">
                  <Button variant="outline" size="sm">
                    Load More (showing 5 of 2,341)
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Analysis Templates */}
          <Card>
            <CardHeader>
              <CardTitle>Analysis Templates</CardTitle>
              <CardDescription>Pre-built analysis workflows</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {analysisTemplates.map((template, idx) => (
                  <div key={idx} className="p-4 border rounded-lg hover:shadow-lg transition-all cursor-pointer">
                    <div className="flex items-start gap-3">
                      <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                        <template.icon className="h-5 w-5 text-primary" />
                      </div>
                      <div className="flex-1">
                        <p className="font-semibold">{template.name}</p>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">{template.description}</p>
                        <Badge variant="outline" className="mt-2 text-xs">{template.language}</Badge>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Recent Queries */}
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">Recent Queries</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {recentQueries.map((q, idx) => (
                  <div key={idx} className="p-3 border rounded-lg">
                    <p className="font-mono text-xs mb-2 text-gray-700 dark:text-gray-300">{q.query}</p>
                    <div className="flex items-center justify-between text-xs">
                      <Badge variant="outline">{q.result}</Badge>
                      <div className="flex gap-3 text-gray-600">
                        <span>{q.time}</span>
                        <span>{q.timestamp}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Available Datasets */}
      <Card>
        <CardHeader>
          <CardTitle>Available Datasets</CardTitle>
          <CardDescription>De-identified data lake access</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            {availableDatasets.map((dataset, idx) => (
              <div key={idx} className="p-4 border rounded-lg">
                <div className="flex items-center gap-2 mb-3">
                  <Database className="h-5 w-5 text-primary" />
                  <p className="font-semibold text-sm">{dataset.name}</p>
                </div>
                <div className="space-y-1 text-xs">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Records:</span>
                    <span className="font-semibold">{dataset.records.toLocaleString()}</span>
                  </div>
                  {dataset.tables && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Tables:</span>
                      <span className="font-semibold">{dataset.tables}</span>
                    </div>
                  )}
                  {dataset.size && (
                    <div className="flex justify-between">
                      <span className="text-gray-600">Size:</span>
                      <span className="font-semibold">{dataset.size}</span>
                    </div>
                  )}
                  <div className="flex justify-between">
                    <span className="text-gray-600">Updated:</span>
                    <span className="text-gray-500">{dataset.lastUpdated}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
