'use client'

import { useEffect, useState } from 'react'
import {
  Activity,
  Users,
  Database,
  Cpu,
  Brain,
  FileText,
  TrendingUp,
  AlertCircle,
  CheckCircle,
  Clock,
  Download,
  Upload,
  Zap
} from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import apiClient from '@/lib/api-client'
import { formatBytes, formatNumber, formatRelativeTime } from '@/lib/utils'

export default function DashboardPage() {
  const [metrics, setMetrics] = useState<any>(null)
  const [studies, setStudies] = useState<any[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      const [metricsData, studiesData] = await Promise.all([
        apiClient.getSystemMetrics().catch(() => ({})),
        apiClient.getStudies({ limit: 5 }).catch(() => ({ items: [] }))
      ])
      setMetrics(metricsData)
      setStudies(studiesData.items || [])
    } catch (error) {
      console.error('Error loading dashboard:', error)
    } finally {
      setLoading(false)
    }
  }

  const stats = [
    {
      title: 'Total Studies',
      value: formatNumber(metrics?.totalStudies || 1248),
      change: '+12.5%',
      icon: Database,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50'
    },
    {
      title: 'Active Users',
      value: formatNumber(metrics?.activeUsers || 156),
      change: '+5.2%',
      icon: Users,
      color: 'text-green-600',
      bgColor: 'bg-green-50'
    },
    {
      title: 'ML Inferences',
      value: formatNumber(metrics?.mlInferences || 892),
      change: '+23.1%',
      icon: Brain,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50'
    },
    {
      title: 'Storage Used',
      value: formatBytes(metrics?.storageUsed || 245678901234),
      change: formatBytes(metrics?.storageTotal || 500000000000) + ' total',
      icon: Database,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50'
    }
  ]

  const recentActivities = [
    {
      type: 'upload',
      message: 'New CT study uploaded for Patient #12345',
      time: '2 minutes ago',
      icon: Upload,
      color: 'text-blue-600'
    },
    {
      type: 'inference',
      message: 'Lung nodule detection completed',
      time: '5 minutes ago',
      icon: Brain,
      color: 'text-purple-600'
    },
    {
      type: 'download',
      message: 'Study #8901 downloaded by Dr. Smith',
      time: '12 minutes ago',
      icon: Download,
      color: 'text-green-600'
    },
    {
      type: 'alert',
      message: 'High priority worklist item assigned',
      time: '18 minutes ago',
      icon: AlertCircle,
      color: 'text-red-600'
    }
  ]

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      {/* Header */}
      <header className="bg-white dark:bg-slate-800 border-b border-slate-200 dark:border-slate-700 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <Activity className="h-8 w-8 text-primary" />
                <h1 className="text-2xl font-bold text-slate-900 dark:text-white">
                  Aurelius Platform
                </h1>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <Button variant="outline" size="sm">
                <Upload className="h-4 w-4 mr-2" />
                Upload Study
              </Button>
              <Button size="sm">
                <Brain className="h-4 w-4 mr-2" />
                Run Inference
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Welcome Section */}
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            Welcome back, Dr. Johnson
          </h2>
          <p className="text-slate-600 dark:text-slate-400">
            Here's what's happening with your medical imaging platform today
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          {stats.map((stat, index) => (
            <Card key={index} className="hover:shadow-lg transition-shadow">
              <CardContent className="pt-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-slate-600 dark:text-slate-400">
                      {stat.title}
                    </p>
                    <h3 className="text-2xl font-bold text-slate-900 dark:text-white mt-2">
                      {stat.value}
                    </h3>
                    <p className="text-sm text-green-600 mt-1">{stat.change}</p>
                  </div>
                  <div className={`${stat.bgColor} p-3 rounded-lg`}>
                    <stat.icon className={`h-6 w-6 ${stat.color}`} />
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Recent Studies */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>Recent Studies</CardTitle>
              <CardDescription>Latest medical imaging studies uploaded</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {[
                  {
                    id: 'STD-001',
                    patient: 'John Doe',
                    modality: 'CT',
                    date: '2024-10-27',
                    status: 'completed',
                    series: 12
                  },
                  {
                    id: 'STD-002',
                    patient: 'Jane Smith',
                    modality: 'MRI',
                    date: '2024-10-27',
                    status: 'processing',
                    series: 8
                  },
                  {
                    id: 'STD-003',
                    patient: 'Bob Johnson',
                    modality: 'X-Ray',
                    date: '2024-10-26',
                    status: 'completed',
                    series: 2
                  }
                ].map((study) => (
                  <div
                    key={study.id}
                    className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-800 rounded-lg hover:bg-slate-100 dark:hover:bg-slate-700 transition-colors cursor-pointer"
                  >
                    <div className="flex items-center space-x-4">
                      <div className="h-12 w-12 bg-primary/10 rounded-lg flex items-center justify-center">
                        <FileText className="h-6 w-6 text-primary" />
                      </div>
                      <div>
                        <p className="font-semibold text-slate-900 dark:text-white">
                          {study.patient}
                        </p>
                        <p className="text-sm text-slate-600 dark:text-slate-400">
                          {study.modality} • {study.series} series • {study.date}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      {study.status === 'completed' ? (
                        <CheckCircle className="h-5 w-5 text-green-600" />
                      ) : (
                        <Clock className="h-5 w-5 text-yellow-600" />
                      )}
                      <Button variant="ghost" size="sm">View</Button>
                    </div>
                  </div>
                ))}
              </div>
              <div className="mt-6">
                <Button variant="outline" className="w-full">
                  View All Studies
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Recent Activity */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Activity</CardTitle>
              <CardDescription>Latest platform activities</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {recentActivities.map((activity, index) => (
                  <div key={index} className="flex items-start space-x-3">
                    <div className={`${activity.color} bg-slate-50 dark:bg-slate-800 p-2 rounded-lg`}>
                      <activity.icon className="h-4 w-4" />
                    </div>
                    <div className="flex-1">
                      <p className="text-sm text-slate-900 dark:text-white">
                        {activity.message}
                      </p>
                      <p className="text-xs text-slate-500 dark:text-slate-400 mt-1">
                        {activity.time}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Quick Actions */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="hover:shadow-lg transition-shadow cursor-pointer">
            <CardContent className="pt-6">
              <div className="flex items-center space-x-4">
                <div className="bg-blue-50 p-3 rounded-lg">
                  <Upload className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-slate-900 dark:text-white">
                    Upload DICOM
                  </h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Import medical images
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow cursor-pointer">
            <CardContent className="pt-6">
              <div className="flex items-center space-x-4">
                <div className="bg-purple-50 p-3 rounded-lg">
                  <Brain className="h-6 w-6 text-purple-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-slate-900 dark:text-white">
                    AI Analysis
                  </h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Run ML inference
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="hover:shadow-lg transition-shadow cursor-pointer">
            <CardContent className="pt-6">
              <div className="flex items-center space-x-4">
                <div className="bg-green-50 p-3 rounded-lg">
                  <TrendingUp className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-slate-900 dark:text-white">
                    View Analytics
                  </h3>
                  <p className="text-sm text-slate-600 dark:text-slate-400">
                    Platform metrics
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  )
}
