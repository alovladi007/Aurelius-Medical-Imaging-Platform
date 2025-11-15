"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Calculator,
  BookOpen,
  Search,
  FileText,
  Activity,
  Shield,
  AlertCircle,
  CheckCircle,
  Brain,
  Stethoscope,
  Pill,
  Microscope,
  TrendingUp,
  Lock
} from 'lucide-react';
import Link from 'next/link';

export default function ClinicalToolsPage() {
  const [activeTools, setActiveTools] = useState(0);

  const tools = [
    {
      id: 'calculators',
      name: 'Clinical Calculators',
      description: 'Risk scores, dosing, renal/hepatic adjustments',
      icon: Calculator,
      href: '/prometheus/clinical-tools/calculators',
      count: 45,
      color: 'bg-blue-500',
      examples: ['CHA₂DS₂-VASc', 'Wells Score', 'HEART Score', 'MELD Score']
    },
    {
      id: 'guidelines',
      name: 'Guideline Engine',
      description: 'CQL executor with patient-specific recommendations',
      icon: BookOpen,
      href: '/prometheus/clinical-tools/guidelines',
      count: 230,
      color: 'bg-green-500',
      examples: ['ACS Pathway', 'Sepsis Bundle', 'Diabetes Management', 'DVT Prophylaxis']
    },
    {
      id: 'retrieval',
      name: 'Policy-Aware RAG',
      description: 'Retrieval over UCG + guidelines + literature',
      icon: Search,
      href: '/prometheus/clinical-tools/retrieval',
      count: 12000000,
      color: 'bg-purple-500',
      examples: ['Evidence Search', 'Case Similarity', 'Literature Lookup', 'Guideline Query']
    },
    {
      id: 'trials',
      name: 'Trial Matching',
      description: 'Criteria to UCG filter for patient screening',
      icon: Microscope,
      href: '/prometheus/clinical-tools/trials',
      count: 1847,
      color: 'bg-orange-500',
      examples: ['Oncology Trials', 'Rare Disease', 'Device Studies', 'Phase II/III']
    },
    {
      id: 'orders',
      name: 'Order Sets',
      description: 'Context-aware draft orders with justifications',
      icon: FileText,
      href: '/prometheus/clinical-tools/orders',
      count: 156,
      color: 'bg-pink-500',
      examples: ['Admission Orders', 'Post-Op Care', 'Sepsis Protocol', 'Chest Pain Workup']
    },
    {
      id: 'causal',
      name: 'Causal What-Ifs',
      description: 'Estimate outcome deltas if interventions changed',
      icon: TrendingUp,
      href: '/prometheus/clinical-tools/causal',
      count: 89,
      color: 'bg-indigo-500',
      examples: ['Med Start/Stop', 'Dose Adjustment', 'Procedure Impact', 'Pathway Change']
    },
    {
      id: 'safety',
      name: 'De-bias & Safety',
      description: 'Fairness metrics, content filter, PHI guard',
      icon: Shield,
      href: '/prometheus/clinical-tools/safety',
      count: 24,
      color: 'bg-red-500',
      examples: ['Subgroup Fairness', 'Red Team', 'PHI Detection', 'Bias Metrics']
    }
  ];

  const recentActivity = [
    { tool: 'CHA₂DS₂-VASc Calculator', user: 'Dr. Smith', patient: 'P-2024-456', result: 'Score: 4 (High Risk)', time: '2 min ago' },
    { tool: 'ACS Guideline', user: 'Dr. Johnson', patient: 'P-2024-789', result: 'Dual antiplatelet recommended', time: '5 min ago' },
    { tool: 'Trial Match: Lung Cancer', user: 'Dr. Williams', patient: 'P-2024-123', result: '3 eligible trials', time: '8 min ago' },
    { tool: 'Sepsis Order Set', user: 'Dr. Brown', patient: 'P-2024-321', result: '12 orders drafted', time: '12 min ago' },
    { tool: 'Med What-If: Stop Warfarin', user: 'Dr. Davis', patient: 'P-2024-654', result: 'Stroke risk +2.3%', time: '15 min ago' }
  ];

  const stats = {
    totalQueries: 15847,
    todayQueries: 342,
    avgResponseTime: 1.2,
    citationRate: 98.5,
    safetyBlocks: 0,
    uncertaintyAbstentions: 23
  };

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Stethoscope className="h-8 w-8 text-primary" />
            Clinical Tools & Services
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            The AGI's "hands" - Policy-aware tools with provenance and safety
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Activity className="h-4 w-4 mr-2" />
            Usage Analytics
          </Button>
          <Button>
            <Shield className="h-4 w-4 mr-2" />
            Safety Dashboard
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-6 gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Total Queries</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-primary">{stats.totalQueries.toLocaleString()}</p>
            <p className="text-xs text-gray-600 mt-1">All-time</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Today</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-blue-600">{stats.todayQueries}</p>
            <p className="text-xs text-green-600">+12% vs yesterday</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Response Time</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">{stats.avgResponseTime}s</p>
            <p className="text-xs text-gray-600">Average</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Citations</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-purple-600">{stats.citationRate}%</p>
            <p className="text-xs text-gray-600">Coverage</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Safety Blocks</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-green-600">{stats.safetyBlocks}</p>
            <p className="text-xs text-gray-600">Last 24h</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Abstentions</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-2xl font-bold text-orange-600">{stats.uncertaintyAbstentions}</p>
            <p className="text-xs text-gray-600">High uncertainty</p>
          </CardContent>
        </Card>
      </div>

      {/* Clinical Tools Grid */}
      <Card>
        <CardHeader>
          <CardTitle>Available Clinical Tools</CardTitle>
          <CardDescription>Policy-aware tools with provenance attached to every claim</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {tools.map((tool) => (
              <Link key={tool.id} href={tool.href}>
                <Card className="h-full hover:shadow-lg transition-all cursor-pointer border-2 hover:border-primary">
                  <CardHeader>
                    <div className="flex items-start justify-between">
                      <div className={`w-12 h-12 ${tool.color} rounded-lg flex items-center justify-center`}>
                        <tool.icon className="h-6 w-6 text-white" />
                      </div>
                      <Badge variant="secondary">{tool.count.toLocaleString()}</Badge>
                    </div>
                    <CardTitle className="mt-4">{tool.name}</CardTitle>
                    <CardDescription>{tool.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <p className="text-xs font-semibold text-gray-600 dark:text-gray-400">Examples:</p>
                      <div className="flex flex-wrap gap-1">
                        {tool.examples.map((example, idx) => (
                          <Badge key={idx} variant="outline" className="text-xs">
                            {example}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </Link>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
          <CardDescription>Latest clinical tool usage across the system</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3">
            {recentActivity.map((activity, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="flex items-center gap-3 flex-1">
                  <div className="w-2 h-2 rounded-full bg-green-500" />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <p className="font-semibold text-sm">{activity.tool}</p>
                      <Badge variant="outline" className="text-xs">{activity.patient}</Badge>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      {activity.user} • {activity.result}
                    </p>
                  </div>
                </div>
                <p className="text-xs text-gray-500">{activity.time}</p>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Core Principles */}
      <Card className="border-blue-200 dark:border-blue-900">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-blue-600" />
            Core Design Principles
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
              <div className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-sm">Provenance Everywhere</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    Every sentence links to sources + timestamps. Evidence strength shown (LoE 1A-5).
                  </p>
                </div>
              </div>
            </div>
            <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg">
              <div className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-sm">Policy-Aware Retrieval</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    RAG over UCG + guidelines + literature with purpose-of-use checks and access control.
                  </p>
                </div>
              </div>
            </div>
            <div className="p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
              <div className="flex items-start gap-3">
                <CheckCircle className="h-5 w-5 text-purple-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-sm">Calibration & Abstention</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    If uncertainty high → ask for more info or route to human. Never guess.
                  </p>
                </div>
              </div>
            </div>
            <div className="p-4 bg-orange-50 dark:bg-orange-950 rounded-lg">
              <div className="flex items-start gap-3">
                <Shield className="h-5 w-5 text-orange-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-sm">De-bias & Safety</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    Fairness metrics by subgroup, content red-teamer, PHI guard, "never" rules.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Getting Started */}
      <Card>
        <CardHeader>
          <CardTitle>Getting Started</CardTitle>
          <CardDescription>Quick access to common clinical workflows</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Link href="/prometheus/clinical-tools/calculators">
              <Button variant="outline" className="w-full h-auto py-4 flex flex-col gap-2">
                <Calculator className="h-6 w-6" />
                <span className="font-semibold">Calculate Risk Score</span>
                <span className="text-xs text-gray-600">CHA₂DS₂-VASc, Wells, HEART</span>
              </Button>
            </Link>
            <Link href="/prometheus/clinical-tools/guidelines">
              <Button variant="outline" className="w-full h-auto py-4 flex flex-col gap-2">
                <BookOpen className="h-6 w-6" />
                <span className="font-semibold">Query Guidelines</span>
                <span className="text-xs text-gray-600">Patient-specific recommendations</span>
              </Button>
            </Link>
            <Link href="/prometheus/clinical-tools/trials">
              <Button variant="outline" className="w-full h-auto py-4 flex flex-col gap-2">
                <Microscope className="h-6 w-6" />
                <span className="font-semibold">Match Clinical Trials</span>
                <span className="text-xs text-gray-600">Find eligible studies</span>
              </Button>
            </Link>
            <Link href="/prometheus/clinical-tools/orders">
              <Button variant="outline" className="w-full h-auto py-4 flex flex-col gap-2">
                <FileText className="h-6 w-6" />
                <span className="font-semibold">Draft Order Set</span>
                <span className="text-xs text-gray-600">Context-aware orders</span>
              </Button>
            </Link>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
