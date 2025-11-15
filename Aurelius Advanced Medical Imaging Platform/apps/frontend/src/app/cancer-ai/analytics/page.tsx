'use client'

import { TrendingUp, Brain, Activity, Users, BarChart3, PieChart as PieChartIcon } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

export default function AnalyticsPage() {
  const performanceMetrics = [
    {
      title: 'Model Accuracy',
      value: '94.3%',
      change: '+2.1% vs last month',
      trend: 'up',
      icon: Brain,
      color: 'text-blue-600'
    },
    {
      title: 'Total Predictions',
      value: '12,847',
      change: '+1,234 this month',
      trend: 'up',
      icon: Activity,
      color: 'text-purple-600'
    },
    {
      title: 'Active Users',
      value: '156',
      change: '+23 this week',
      trend: 'up',
      icon: Users,
      color: 'text-green-600'
    },
    {
      title: 'Avg Confidence',
      value: '87.6%',
      change: '+3.4% improvement',
      trend: 'up',
      icon: TrendingUp,
      color: 'text-orange-600'
    }
  ]

  const cancerTypeDistribution = [
    { type: 'Lung Cancer', count: 4523, percentage: 35.2, color: 'bg-blue-500' },
    { type: 'Breast Cancer', count: 3891, percentage: 30.3, color: 'bg-pink-500' },
    { type: 'Prostate Cancer', count: 2314, percentage: 18.0, color: 'bg-purple-500' },
    { type: 'Colorectal Cancer', count: 1456, percentage: 11.3, color: 'bg-green-500' },
    { type: 'No Cancer Detected', count: 663, percentage: 5.2, color: 'bg-gray-400' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            Cancer AI Analytics
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Performance metrics and insights from the cancer detection AI system
          </p>
        </div>

        {/* Performance Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {performanceMetrics.map((metric, idx) => (
            <Card key={idx} className="border-l-4 border-l-blue-500">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-slate-600">
                    {metric.title}
                  </CardTitle>
                  <metric.icon className={`w-5 h-5 ${metric.color}`} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold mb-1">{metric.value}</div>
                <p className="text-xs text-slate-500 flex items-center gap-1">
                  <TrendingUp className="w-3 h-3 text-green-600" />
                  {metric.change}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Cancer Type Distribution */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="flex items-center gap-2">
                  <PieChartIcon className="w-5 h-5 text-blue-600" />
                  Cancer Type Distribution
                </CardTitle>
                <CardDescription>Breakdown of predictions by cancer type</CardDescription>
              </div>
              <Badge variant="secondary">Last 30 Days</Badge>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {cancerTypeDistribution.map((item, idx) => (
                <div key={idx} className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium">{item.type}</span>
                    <span className="text-slate-600">
                      {item.count.toLocaleString()} ({item.percentage}%)
                    </span>
                  </div>
                  <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full ${item.color} transition-all duration-500`}
                      style={{ width: `${item.percentage}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Monthly Trend */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-purple-600" />
              Monthly Prediction Trends
            </CardTitle>
            <CardDescription>Prediction volume over the last 12 months</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-64 flex items-end justify-between gap-2">
              {[650, 720, 890, 1100, 980, 1200, 1150, 1340, 1280, 1420, 1380, 1560].map((value, idx) => (
                <div key={idx} className="flex-1 flex flex-col items-center gap-2">
                  <div
                    className="w-full bg-gradient-to-t from-blue-500 to-purple-500 rounded-t transition-all duration-500 hover:opacity-80"
                    style={{ height: `${(value / 1600) * 100}%` }}
                  />
                  <span className="text-xs text-slate-500">
                    {['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'][idx]}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Model Performance Over Time */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Accuracy Trends</CardTitle>
              <CardDescription>Model accuracy improvements over time</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Lung Cancer Detection</span>
                  <span className="font-bold text-blue-600">96.2%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Breast Cancer Detection</span>
                  <span className="font-bold text-pink-600">94.8%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Prostate Cancer Detection</span>
                  <span className="font-bold text-purple-600">92.1%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Colorectal Cancer Detection</span>
                  <span className="font-bold text-green-600">93.5%</span>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Processing Metrics</CardTitle>
              <CardDescription>System performance statistics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Avg Processing Time</span>
                  <span className="font-bold text-blue-600">2.3s</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Peak Load Handled</span>
                  <span className="font-bold text-purple-600">450/min</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">System Uptime</span>
                  <span className="font-bold text-green-600">99.97%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-600">Avg Confidence Score</span>
                  <span className="font-bold text-orange-600">87.6%</span>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
