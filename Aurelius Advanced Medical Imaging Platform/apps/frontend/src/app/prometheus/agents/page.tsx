"use client";

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  Bot,
  Brain,
  Shield,
  AlertCircle,
  CheckCircle,
  Activity,
  Stethoscope,
  Microscope,
  Heart,
  Pill,
  GraduationCap,
  Zap,
  Eye,
  TrendingUp
} from 'lucide-react';
import Link from 'next/link';

export default function AgentsPage() {
  const [systemStatus, setSystemStatus] = useState('healthy');

  // Core orchestration agents
  const orchestrationAgents = [
    {
      id: 'planner',
      name: 'Planner',
      description: 'Breaks complex tasks into steps',
      status: 'active',
      tasksProcessed: 15847,
      avgTime: 0.8,
      icon: Brain,
      color: 'text-purple-600'
    },
    {
      id: 'router',
      name: 'Router',
      description: 'Picks optimal tools/models',
      status: 'active',
      tasksProcessed: 23901,
      avgTime: 0.3,
      icon: Zap,
      color: 'text-yellow-600'
    },
    {
      id: 'critic',
      name: 'Critic',
      description: 'Checks reasoning quality',
      status: 'active',
      tasksProcessed: 18456,
      avgTime: 1.2,
      icon: Eye,
      color: 'text-blue-600'
    },
    {
      id: 'safety',
      name: 'Safety Officer',
      description: 'Enforces guardrails & policies',
      status: 'active',
      tasksProcessed: 28934,
      avgTime: 0.5,
      icon: Shield,
      color: 'text-green-600'
    },
    {
      id: 'supervisor',
      name: 'Supervisor',
      description: 'Maintains context & memory',
      status: 'active',
      tasksProcessed: 19234,
      avgTime: 0.4,
      icon: Bot,
      color: 'text-indigo-600'
    }
  ];

  // Task-specific agents
  const taskAgents = [
    {
      id: 'triage',
      name: 'Triage/Intake Agent',
      description: 'ED/urgent care symptom parsing → differential → initial orders',
      icon: Activity,
      status: 'active',
      specialty: 'Emergency Medicine',
      patientsToday: 342,
      accuracy: 94.8,
      href: '/prometheus/agents/triage',
      color: 'bg-red-500'
    },
    {
      id: 'diagnostic',
      name: 'Diagnostic Copilot',
      description: 'Synthesizes problems, narrows differential, asks for data',
      icon: Stethoscope,
      status: 'active',
      specialty: 'Internal Medicine',
      patientsToday: 567,
      accuracy: 96.2,
      href: '/prometheus/agents/diagnostic',
      color: 'bg-blue-500'
    },
    {
      id: 'therapy',
      name: 'Therapy Planner',
      description: 'Proposes treatments aligned to guidelines + PGx',
      icon: Pill,
      status: 'active',
      specialty: 'Pharmacology',
      patientsToday: 423,
      accuracy: 97.1,
      href: '/prometheus/agents/therapy',
      color: 'bg-green-500'
    },
    {
      id: 'radiology',
      name: 'Radiology Copilot',
      description: 'Comparative reads, RAD-Lex findings → impression drafts',
      icon: Brain,
      status: 'active',
      specialty: 'Radiology',
      patientsToday: 789,
      accuracy: 95.6,
      href: '/prometheus/agents/radiology',
      color: 'bg-purple-500'
    },
    {
      id: 'pathology',
      name: 'Pathology Copilot',
      description: 'WSI regions of interest + report drafting',
      icon: Microscope,
      status: 'active',
      specialty: 'Pathology',
      patientsToday: 234,
      accuracy: 94.3,
      href: '/prometheus/agents/pathology',
      color: 'bg-pink-500'
    },
    {
      id: 'icu',
      name: 'ICU Agent',
      description: 'Streaming vitals, early warning, closed-loop suggestions',
      icon: Heart,
      status: 'active',
      specialty: 'Critical Care',
      patientsToday: 89,
      accuracy: 98.1,
      href: '/prometheus/agents/icu',
      color: 'bg-orange-500'
    },
    {
      id: 'research',
      name: 'Research Copilot',
      description: 'Cohort discovery, causal analysis, literature triage',
      icon: TrendingUp,
      status: 'active',
      specialty: 'Research',
      patientsToday: 156,
      accuracy: 93.8,
      href: '/prometheus/agents/research',
      color: 'bg-indigo-500'
    },
    {
      id: 'tutor',
      name: 'Student/Tutor',
      description: 'Case-based teaching, viva prompts, adaptive quizzes',
      icon: GraduationCap,
      status: 'active',
      specialty: 'Education',
      patientsToday: 678,
      accuracy: 96.7,
      href: '/prometheus/agents/tutor',
      color: 'bg-yellow-500'
    }
  ];

  const recentActivity = [
    { agent: 'Triage Agent', task: 'Chest pain workup', patient: 'P-2024-001', duration: '2.3s', status: 'completed' },
    { agent: 'Diagnostic Copilot', task: 'Differential diagnosis', patient: 'P-2024-045', duration: '4.1s', status: 'completed' },
    { agent: 'ICU Agent', task: 'Sepsis early warning', patient: 'P-2024-023', duration: '0.8s', status: 'alert' },
    { agent: 'Therapy Planner', task: 'Anticoagulation plan', patient: 'P-2024-067', duration: '3.2s', status: 'completed' },
    { agent: 'Radiology Copilot', task: 'CXR comparison', patient: 'P-2024-089', duration: '1.9s', status: 'completed' }
  ];

  const metrics = {
    totalTasks: 156234,
    activeTasks: 42,
    avgResponseTime: 1.8,
    successRate: 98.2,
    humanOverrides: 12,
    safetyInterventions: 8
  };

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Bot className="h-8 w-8 text-primary" />
            Cognition & Orchestration Layer
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-2">
            Autonomous agents with planning, routing, and safety enforcement
          </p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline">
            <Activity className="h-4 w-4 mr-2" />
            Monitor Performance
          </Button>
          <Button>
            <Shield className="h-4 w-4 mr-2" />
            Safety Dashboard
          </Button>
        </div>
      </div>

      {/* System Status */}
      <Card className={systemStatus === 'healthy' ? 'border-green-500' : 'border-yellow-500'}>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>System Status</CardTitle>
            <Badge className={systemStatus === 'healthy' ? 'bg-green-600' : 'bg-yellow-600'}>
              {systemStatus === 'healthy' ? 'All Systems Operational' : 'Partial Degradation'}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
            <div>
              <p className="text-sm text-gray-600">Total Tasks</p>
              <p className="text-2xl font-bold">{metrics.totalTasks.toLocaleString()}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Active Now</p>
              <p className="text-2xl font-bold text-blue-600">{metrics.activeTasks}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Avg Response</p>
              <p className="text-2xl font-bold text-green-600">{metrics.avgResponseTime}s</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Success Rate</p>
              <p className="text-2xl font-bold text-purple-600">{metrics.successRate}%</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Human Overrides</p>
              <p className="text-2xl font-bold text-orange-600">{metrics.humanOverrides}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Safety Blocks</p>
              <p className="text-2xl font-bold text-red-600">{metrics.safetyInterventions}</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Orchestration Agents */}
      <Card>
        <CardHeader>
          <CardTitle>Core Orchestration Agents</CardTitle>
          <CardDescription>Foundation agents managing workflow, routing, and safety</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
            {orchestrationAgents.map((agent) => (
              <Card key={agent.id} className="border-2">
                <CardContent className="pt-6">
                  <div className="flex flex-col items-center text-center">
                    <div className="w-12 h-12 bg-primary/10 rounded-full flex items-center justify-center mb-3">
                      <agent.icon className={`h-6 w-6 ${agent.color}`} />
                    </div>
                    <p className="font-semibold mb-1">{agent.name}</p>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mb-3">{agent.description}</p>
                    <Badge variant="outline" className="mb-2">
                      {agent.status === 'active' ? <CheckCircle className="h-3 w-3 mr-1 text-green-600" /> : null}
                      {agent.status}
                    </Badge>
                    <div className="w-full space-y-1 text-xs">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Tasks:</span>
                        <span className="font-semibold">{agent.tasksProcessed.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Avg Time:</span>
                        <span className="font-semibold">{agent.avgTime}s</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Task Agents */}
      <Card>
        <CardHeader>
          <CardTitle>Task-Specific Agents</CardTitle>
          <CardDescription>Specialized agents for clinical workflows and education</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {taskAgents.map((agent) => (
              <Link key={agent.id} href={agent.href}>
                <Card className="h-full hover:shadow-lg transition-all cursor-pointer border-2 hover:border-primary">
                  <CardHeader>
                    <div className="flex items-start justify-between mb-3">
                      <div className={`w-10 h-10 ${agent.color} rounded-lg flex items-center justify-center`}>
                        <agent.icon className="h-5 w-5 text-white" />
                      </div>
                      <Badge variant="outline">{agent.specialty}</Badge>
                    </div>
                    <CardTitle className="text-base">{agent.name}</CardTitle>
                    <CardDescription className="text-xs">{agent.description}</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600">Status:</span>
                        <Badge className="bg-green-600">{agent.status}</Badge>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600">Today:</span>
                        <span className="font-semibold">{agent.patientsToday} patients</span>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-gray-600">Accuracy:</span>
                        <span className="font-semibold text-primary">{agent.accuracy}%</span>
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
          <CardTitle>Recent Agent Activity</CardTitle>
          <CardDescription>Live stream of agent tasks across the system</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-2">
            {recentActivity.map((activity, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-900 rounded-lg">
                <div className="flex items-center gap-3 flex-1">
                  <div className={`w-2 h-2 rounded-full ${activity.status === 'completed' ? 'bg-green-500' : 'bg-orange-500'}`} />
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <p className="font-semibold text-sm">{activity.agent}</p>
                      <Badge variant="outline" className="text-xs">{activity.patient}</Badge>
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400">{activity.task}</p>
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-gray-600">{activity.duration}</span>
                  {activity.status === 'completed' ? (
                    <CheckCircle className="h-4 w-4 text-green-600" />
                  ) : (
                    <AlertCircle className="h-4 w-4 text-orange-600" />
                  )}
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Agent Architecture */}
      <Card className="border-blue-200 dark:border-blue-900">
        <CardHeader>
          <CardTitle>Agent Architecture Principles</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-950 rounded-lg">
              <div className="flex items-start gap-3">
                <Brain className="h-5 w-5 text-blue-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-sm">Hierarchical Planning</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    Planner breaks complex tasks into subtasks. Router picks optimal tools/models for each step.
                  </p>
                </div>
              </div>
            </div>
            <div className="p-4 bg-green-50 dark:bg-green-950 rounded-lg">
              <div className="flex items-start gap-3">
                <Shield className="h-5 w-5 text-green-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-sm">Multi-Layer Safety</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    Safety Officer enforces guardrails. Critic validates reasoning. Never auto-actuate clinical decisions.
                  </p>
                </div>
              </div>
            </div>
            <div className="p-4 bg-purple-50 dark:bg-purple-950 rounded-lg">
              <div className="flex items-start gap-3">
                <Eye className="h-5 w-5 text-purple-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-sm">Continuous Critique</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    Critic agent reviews all outputs for logical consistency, evidence quality, and safety.
                  </p>
                </div>
              </div>
            </div>
            <div className="p-4 bg-orange-50 dark:bg-orange-950 rounded-lg">
              <div className="flex items-start gap-3">
                <Bot className="h-5 w-5 text-orange-600 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-semibold text-sm">Contextual Memory</p>
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    Supervisor maintains patient context, conversation history, and task state across sessions.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
