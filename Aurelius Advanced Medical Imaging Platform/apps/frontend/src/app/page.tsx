'use client'

import { useEffect, useState } from 'react'
import {
  Database,
  Users,
  Brain,
  Activity,
  TrendingUp,
  TrendingDown,
  Upload,
  Download,
  Clock,
  CheckCircle,
  AlertTriangle,
  Zap,
  FileText,
  Microscope,
  Beaker
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { MetricsCard } from '@/components/research/MetricsCard'
import { StudyTable } from '@/components/research/StudyTable'
import { ResearchAnalytics } from '@/components/research/ResearchAnalytics'
import apiClient from '@/lib/api-client'
import { formatBytes, formatNumber } from '@/lib/utils'

export default function ResearchDashboard() {
  const [metrics, setMetrics] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      const metricsData = await apiClient.getSystemMetrics().catch(() => ({
        totalStudies: 1248,
        activeUsers: 156,
        mlInferences: 892,
        storageUsed: 245678901234,
        studiesThisWeek: 87,
        aiAnalysisRate: 71.2,
        avgProcessingTime: 45,
        activeResearchers: 23
      }))
      setMetrics(metricsData)
    } catch (error) {
      console.error('Error loading dashboard:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-slate-50 dark:bg-slate-900">
        <div className="text-center">
          <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-primary mb-4"></div>
          <p className="text-slate-600 dark:text-slate-400">Loading research platform...</p>
        </div>
      </div>
    )
  }

  const keyMetrics = [
    {
      title: 'Total Studies',
      value: formatNumber(metrics?.totalStudies || 1248),
      change: '+12.5% from last month',
      changeType: 'positive' as const,
      icon: Database,
      trend: [45, 52, 48, 61, 58, 71, 69, 78, 82, 89, 95, 101]
    },
    {
      title: 'Active Researchers',
      value: formatNumber(metrics?.activeResearchers || 23),
      change: '+3 this week',
      changeType: 'positive' as const,
      icon: Users,
      trend: [18, 19, 20, 21, 19, 22, 21, 23, 22, 23, 24, 23]
    },
    {
      title: 'AI Inferences',
      value: formatNumber(metrics?.mlInferences || 892),
      change: '+23.1% this week',
      changeType: 'positive' as const,
      icon: Brain,
      trend: [450, 520, 580, 640, 690, 720, 760, 800, 820, 850, 870, 892]
    },
    {
      title: 'Processing Rate',
      value: `${metrics?.aiAnalysisRate || 71.2}%`,
      change: '+5.3% improvement',
      changeType: 'positive' as const,
      icon: Zap,
      description: 'AI analysis coverage',
      trend: [55, 58, 61, 63, 65, 67, 68, 69, 70, 71, 71, 71.2]
    }
  ]

  const recentActivities = [
    {
      type: 'upload',
      title: 'New CT Study Uploaded',
      description: 'Chest CT with Contrast - Patient #45782',
      time: '2 minutes ago',
      icon: Upload,
      iconColor: 'text-blue-600',
      iconBg: 'bg-blue-50'
    },
    {
      type: 'analysis',
      title: 'AI Analysis Completed',
      description: 'Lung nodule detection - 3 findings identified',
      time: '5 minutes ago',
      icon: Brain,
      iconColor: 'text-purple-600',
      iconBg: 'bg-purple-50'
    },
    {
      type: 'research',
      title: 'Research Dataset Updated',
      description: 'Cancer detection cohort - 45 new annotations',
      time: '12 minutes ago',
      icon: Beaker,
      iconColor: 'text-green-600',
      iconBg: 'bg-green-50'
    },
    {
      type: 'collaboration',
      title: 'Collaboration Invitation',
      description: 'Dr. Smith invited you to "Brain Tumor Research"',
      time: '18 minutes ago',
      icon: Users,
      iconColor: 'text-orange-600',
      iconBg: 'bg-orange-50'
    },
    {
      type: 'alert',
      title: 'Quality Check Alert',
      description: 'Study #STD-2024-089 requires review',
      time: '25 minutes ago',
      icon: AlertTriangle,
      iconColor: 'text-red-600',
      iconBg: 'bg-red-50'
    }
  ]

  const quickActions = [
    {
      title: 'Upload DICOM Study',
      description: 'Import medical imaging studies',
      icon: Upload,
      color: 'bg-blue-500 hover:bg-blue-600',
      href: '/upload'
    },
    {
      title: 'Run AI Analysis',
      description: 'Analyze studies with ML models',
      icon: Brain,
      color: 'bg-purple-500 hover:bg-purple-600',
      href: '/ml'
    },
    {
      title: 'View Studies',
      description: 'Browse and search DICOM studies',
      icon: Microscope,
      color: 'bg-green-500 hover:bg-green-600',
      href: '/studies'
    },
    {
      title: 'Create Dataset',
      description: 'Build research cohorts',
      icon: Beaker,
      color: 'bg-orange-500 hover:bg-orange-600',
      href: '/research/datasets'
    }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-slate-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      {/* Header */}
      <header className="bg-white/80 dark:bg-slate-900/80 backdrop-blur-lg border-b border-slate-200 dark:border-slate-700 sticky top-0 z-40">
        <div className="mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-primary to-blue-600 bg-clip-text text-transparent">
                Research Dashboard
              </h1>
              <p className="text-slate-600 dark:text-slate-400 mt-1">
                Welcome back, Dr. Johnson - {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
              </p>
            </div>
            <div className="flex items-center gap-3">
              <Button variant="outline" size="sm">
                <Download className="h-4 w-4 mr-2" />
                Export Data
              </Button>
              <Button size="sm" className="bg-gradient-to-r from-primary to-blue-600 hover:from-primary/90 hover:to-blue-600/90">
                <Upload className="h-4 w-4 mr-2" />
                Upload Study
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="mx-auto px-6 py-8 space-y-8">
        {/* Key Metrics */}
        <section>
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
              Key Performance Indicators
            </h2>
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="text-green-600 border-green-600">
                <Activity className="h-3 w-3 mr-1" />
                All Systems Operational
              </Badge>
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {keyMetrics.map((metric, index) => (
              <MetricsCard key={index} {...metric} />
            ))}
          </div>
        </section>

        {/* Quick Actions */}
        <section>
          <h2 className="text-2xl font-bold text-slate-900 dark:text-white mb-6">
            Quick Actions
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {quickActions.map((action, index) => (
              <Card
                key={index}
                className="group cursor-pointer hover:shadow-xl transition-all duration-300 border-0 overflow-hidden"
              >
                <div className={`h-2 ${action.color.split(' ')[0]}`} />
                <CardContent className="pt-6">
                  <div className="flex items-start gap-4">
                    <div className={`p-3 rounded-xl ${action.color} text-white transition-transform group-hover:scale-110`}>
                      <action.icon className="h-6 w-6" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-slate-900 dark:text-white mb-1">
                        {action.title}
                      </h3>
                      <p className="text-sm text-slate-600 dark:text-slate-400">
                        {action.description}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </section>

        {/* Main Content Tabs */}
        <section>
          <Tabs defaultValue="studies" className="space-y-6">
            <TabsList className="grid w-full grid-cols-3 lg:w-[600px]">
              <TabsTrigger value="studies" className="flex items-center gap-2">
                <FileText className="h-4 w-4" />
                Recent Studies
              </TabsTrigger>
              <TabsTrigger value="analytics" className="flex items-center gap-2">
                <TrendingUp className="h-4 w-4" />
                Analytics
              </TabsTrigger>
              <TabsTrigger value="activity" className="flex items-center gap-2">
                <Activity className="h-4 w-4" />
                Activity Feed
              </TabsTrigger>
            </TabsList>

            <TabsContent value="studies" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Recent Research Studies</CardTitle>
                  <CardDescription>
                    Latest DICOM studies uploaded to the platform with AI analysis status
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <StudyTable />
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="analytics" className="space-y-4">
              <ResearchAnalytics />
            </TabsContent>

            <TabsContent value="activity" className="space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle>Platform Activity Stream</CardTitle>
                  <CardDescription>
                    Real-time updates from across the research platform
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {recentActivities.map((activity, index) => (
                      <div
                        key={index}
                        className="flex items-start gap-4 p-4 rounded-lg hover:bg-slate-50 dark:hover:bg-slate-800 transition-colors"
                      >
                        <div className={`${activity.iconBg} p-3 rounded-lg`}>
                          <activity.icon className={`h-5 w-5 ${activity.iconColor}`} />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-start justify-between">
                            <div>
                              <h4 className="font-semibold text-slate-900 dark:text-white">
                                {activity.title}
                              </h4>
                              <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                                {activity.description}
                              </p>
                            </div>
                            <span className="text-xs text-slate-500 whitespace-nowrap ml-4">
                              {activity.time}
                            </span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </section>

        {/* Bottom Grid */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* System Health */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5 text-green-600" />
                System Health
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {[
                { service: 'Gateway API', status: 'operational', uptime: '99.9%' },
                { service: 'DICOM Server', status: 'operational', uptime: '99.8%' },
                { service: 'ML Service', status: 'operational', uptime: '99.7%' },
                { service: 'Database', status: 'operational', uptime: '99.9%' }
              ].map((item, index) => (
                <div key={index} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <CheckCircle className="h-4 w-4 text-green-600" />
                    <span className="text-sm text-slate-700 dark:text-slate-300">
                      {item.service}
                    </span>
                  </div>
                  <Badge variant="success" className="text-xs">
                    {item.uptime}
                  </Badge>
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Storage Usage */}
          <Card>
            <CardHeader>
              <CardTitle>Storage Usage</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-slate-600 dark:text-slate-400">DICOM Studies</span>
                    <span className="font-medium">186 GB</span>
                  </div>
                  <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                    <div className="h-full bg-blue-500" style={{ width: '62%' }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-slate-600 dark:text-slate-400">ML Models</span>
                    <span className="font-medium">42 GB</span>
                  </div>
                  <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                    <div className="h-full bg-purple-500" style={{ width: '28%' }} />
                  </div>
                </div>
                <div>
                  <div className="flex justify-between text-sm mb-2">
                    <span className="text-slate-600 dark:text-slate-400">Research Data</span>
                    <span className="font-medium">28 GB</span>
                  </div>
                  <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                    <div className="h-full bg-green-500" style={{ width: '18%' }} />
                  </div>
                </div>
                <div className="pt-4 border-t border-slate-200 dark:border-slate-700">
                  <div className="flex justify-between">
                    <span className="text-sm font-medium">Total Used</span>
                    <span className="font-bold">256 GB / 500 GB</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Active Tasks */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Clock className="h-5 w-5 text-blue-600" />
                Active Tasks
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {[
                  { task: 'AI Model Training', progress: 67, status: 'running' },
                  { task: 'DICOM Processing Queue', progress: 89, status: 'running' },
                  { task: 'Dataset Generation', progress: 34, status: 'running' }
                ].map((item, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex items-center justify-between text-sm">
                      <span className="text-slate-700 dark:text-slate-300">{item.task}</span>
                      <span className="text-slate-500">{item.progress}%</span>
                    </div>
                    <div className="h-2 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-blue-500 transition-all duration-500 rounded-full"
                        style={{ width: `${item.progress}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  )
}
