'use client'

import { useState } from 'react'
import { FileText, Search, Filter, Upload, Eye } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

export default function StudiesPage() {
  const [searchQuery, setSearchQuery] = useState('')

  const studies = [
    { id: '1', patientName: 'John Doe', patientId: 'P001', modality: 'CT', date: '2024-10-27', series: 12, status: 'completed' },
    { id: '2', patientName: 'Jane Smith', patientId: 'P002', modality: 'MRI', date: '2024-10-27', series: 8, status: 'processing' },
    { id: '3', patientName: 'Bob Johnson', patientId: 'P003', modality: 'X-Ray', date: '2024-10-26', series: 2, status: 'completed' },
    { id: '4', patientName: 'Alice Williams', patientId: 'P004', modality: 'CT', date: '2024-10-26', series: 15, status: 'completed' },
    { id: '5', patientName: 'Charlie Brown', patientId: 'P005', modality: 'MRI', date: '2024-10-25', series: 10, status: 'completed' },
  ]

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900">
      <div className="max-w-7xl mx-auto p-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
            Studies Management
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Browse and manage medical imaging studies
          </p>
        </div>

        {/* Toolbar */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-4 flex-1 max-w-2xl">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 h-4 w-4" />
              <input
                type="text"
                placeholder="Search studies, patients, or IDs..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-primary focus:border-transparent"
              />
            </div>
            <Button variant="outline">
              <Filter className="h-4 w-4 mr-2" />
              Filters
            </Button>
          </div>
          <Button>
            <Upload className="h-4 w-4 mr-2" />
            Upload Study
          </Button>
        </div>

        {/* Studies Grid */}
        <div className="grid gap-4">
          {studies.map((study) => (
            <Card key={study.id} className="hover:shadow-lg transition-shadow">
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="h-16 w-16 bg-primary/10 rounded-lg flex items-center justify-center">
                      <FileText className="h-8 w-8 text-primary" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                        {study.patientName}
                      </h3>
                      <p className="text-sm text-slate-600 dark:text-slate-400">
                        Patient ID: {study.patientId} • {study.modality} • {study.date}
                      </p>
                      <div className="flex items-center space-x-4 mt-2">
                        <span className="text-xs text-slate-500">{study.series} series</span>
                        <span className={`text-xs px-2 py-1 rounded ${
                          study.status === 'completed' ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
                        }`}>
                          {study.status}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex space-x-2">
                    <Button variant="outline" size="sm">
                      <Eye className="h-4 w-4 mr-2" />
                      View
                    </Button>
                    <Button size="sm">
                      Run AI Analysis
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    </div>
  )
}
