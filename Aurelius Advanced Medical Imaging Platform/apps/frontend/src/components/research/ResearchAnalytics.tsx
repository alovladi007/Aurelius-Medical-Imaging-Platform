'use client'

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { BarChart, Bar, LineChart, Line, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const studyVolumeData = [
  { month: 'Jan', studies: 145, aiAnalyzed: 98 },
  { month: 'Feb', studies: 168, aiAnalyzed: 124 },
  { month: 'Mar', studies: 192, aiAnalyzed: 156 },
  { month: 'Apr', studies: 211, aiAnalyzed: 178 },
  { month: 'May', studies: 234, aiAnalyzed: 201 },
  { month: 'Jun', studies: 256, aiAnalyzed: 223 },
]

const modalityDistribution = [
  { name: 'CT', value: 456, color: '#3b82f6' },
  { name: 'MRI', value: 342, color: '#8b5cf6' },
  { name: 'X-Ray', value: 289, color: '#10b981' },
  { name: 'Ultrasound', value: 167, color: '#f59e0b' },
  { name: 'PET', value: 94, color: '#ef4444' },
]

const aiPerformanceData = [
  { metric: 'Lung Nodule', accuracy: 94.2, sensitivity: 92.8, specificity: 95.6 },
  { metric: 'Breast Cancer', accuracy: 91.5, sensitivity: 89.3, specificity: 93.2 },
  { metric: 'Brain Tumor', accuracy: 88.7, sensitivity: 86.1, specificity: 91.3 },
  { metric: 'Fracture', accuracy: 96.3, sensitivity: 95.1, specificity: 97.4 },
]

export function ResearchAnalytics() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Study Volume Trends */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle>Study Volume & AI Analysis Trends</CardTitle>
          <CardDescription>
            Total studies uploaded vs AI-analyzed studies over the past 6 months
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={studyVolumeData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="month" stroke="#64748b" />
              <YAxis stroke="#64748b" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px'
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="studies"
                stroke="#3b82f6"
                strokeWidth={2}
                name="Total Studies"
                dot={{ fill: '#3b82f6', r: 4 }}
              />
              <Line
                type="monotone"
                dataKey="aiAnalyzed"
                stroke="#8b5cf6"
                strokeWidth={2}
                name="AI Analyzed"
                dot={{ fill: '#8b5cf6', r: 4 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Modality Distribution */}
      <Card>
        <CardHeader>
          <CardTitle>Imaging Modality Distribution</CardTitle>
          <CardDescription>
            Breakdown of studies by imaging modality type
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={modalityDistribution}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={100}
                fill="#8884d8"
                dataKey="value"
              >
                {modalityDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
          <div className="mt-4 grid grid-cols-2 gap-4">
            {modalityDistribution.map((modality) => (
              <div key={modality.name} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: modality.color }}
                />
                <span className="text-sm text-slate-600 dark:text-slate-400">
                  {modality.name}: {modality.value}
                </span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* AI Model Performance */}
      <Card>
        <CardHeader>
          <CardTitle>AI Model Performance Metrics</CardTitle>
          <CardDescription>
            Accuracy, sensitivity, and specificity by detection type
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={aiPerformanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis dataKey="metric" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid #e5e7eb',
                  borderRadius: '8px'
                }}
              />
              <Legend />
              <Bar dataKey="accuracy" fill="#3b82f6" name="Accuracy" radius={[4, 4, 0, 0]} />
              <Bar dataKey="sensitivity" fill="#10b981" name="Sensitivity" radius={[4, 4, 0, 0]} />
              <Bar dataKey="specificity" fill="#8b5cf6" name="Specificity" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  )
}
