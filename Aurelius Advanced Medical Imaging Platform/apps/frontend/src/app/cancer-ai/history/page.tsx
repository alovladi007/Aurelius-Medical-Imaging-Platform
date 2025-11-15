'use client'

import { useState } from 'react'
import { Card } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import {
  Clock,
  Search,
  Filter,
  Download,
  FileText,
  AlertCircle,
  CheckCircle,
  XCircle,
  Eye,
  Trash2,
  Calendar
} from 'lucide-react'

interface AnalysisRecord {
  id: string
  timestamp: string
  patientId: string
  studyType: string
  cancerType: string
  confidence: number
  status: 'completed' | 'failed' | 'pending'
  findings: string
  modelVersion: string
}

export default function HistoryPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [filterStatus, setFilterStatus] = useState<string>('all')

  // Mock data - in production, this would come from an API
  const analysisHistory: AnalysisRecord[] = [
    {
      id: 'ANA-001',
      timestamp: '2024-01-15T14:30:00Z',
      patientId: 'PT-45782',
      studyType: 'Chest CT',
      cancerType: 'Lung Cancer',
      confidence: 94.3,
      status: 'completed',
      findings: 'Suspicious nodule detected in upper right lobe',
      modelVersion: 'v2.3.1'
    },
    {
      id: 'ANA-002',
      timestamp: '2024-01-15T13:15:00Z',
      patientId: 'PT-45780',
      studyType: 'Mammography',
      cancerType: 'Breast Cancer',
      confidence: 89.7,
      status: 'completed',
      findings: 'Calcifications present - recommend biopsy',
      modelVersion: 'v2.3.1'
    },
    {
      id: 'ANA-003',
      timestamp: '2024-01-15T12:00:00Z',
      patientId: 'PT-45775',
      studyType: 'Brain MRI',
      cancerType: 'Brain Tumor',
      confidence: 92.1,
      status: 'completed',
      findings: 'Mass detected in frontal lobe',
      modelVersion: 'v2.3.0'
    },
    {
      id: 'ANA-004',
      timestamp: '2024-01-15T10:45:00Z',
      patientId: 'PT-45768',
      studyType: 'Abdominal CT',
      cancerType: 'Liver Cancer',
      confidence: 87.5,
      status: 'completed',
      findings: 'Multiple lesions identified',
      modelVersion: 'v2.3.1'
    },
    {
      id: 'ANA-005',
      timestamp: '2024-01-15T09:30:00Z',
      patientId: 'PT-45760',
      studyType: 'Prostate MRI',
      cancerType: 'Prostate Cancer',
      confidence: 0,
      status: 'failed',
      findings: 'Analysis failed - insufficient image quality',
      modelVersion: 'v2.3.1'
    }
  ]

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-5 w-5 text-green-500" />
      case 'failed':
        return <XCircle className="h-5 w-5 text-red-500" />
      case 'pending':
        return <Clock className="h-5 w-5 text-yellow-500" />
      default:
        return <AlertCircle className="h-5 w-5 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'failed':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      case 'pending':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const formatDate = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const filteredHistory = analysisHistory.filter(record => {
    const matchesSearch =
      record.patientId.toLowerCase().includes(searchQuery.toLowerCase()) ||
      record.studyType.toLowerCase().includes(searchQuery.toLowerCase()) ||
      record.cancerType.toLowerCase().includes(searchQuery.toLowerCase())

    const matchesFilter = filterStatus === 'all' || record.status === filterStatus

    return matchesSearch && matchesFilter
  })

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-purple-50 to-slate-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-slate-900 dark:text-white mb-2">
          Cancer AI Analysis History
        </h1>
        <p className="text-slate-600 dark:text-slate-400">
          View and manage all previous cancer detection analyses
        </p>
      </div>

      {/* Filters and Search */}
      <Card className="p-4 mb-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
            <Input
              placeholder="Search by patient ID, study type, or cancer type..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-10"
            />
          </div>

          <div className="flex gap-2">
            <Button
              variant={filterStatus === 'all' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilterStatus('all')}
            >
              All
            </Button>
            <Button
              variant={filterStatus === 'completed' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilterStatus('completed')}
            >
              Completed
            </Button>
            <Button
              variant={filterStatus === 'failed' ? 'default' : 'outline'}
              size="sm"
              onClick={() => setFilterStatus('failed')}
            >
              Failed
            </Button>
          </div>

          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </Card>

      {/* Analysis Timeline */}
      <div className="space-y-4">
        {filteredHistory.map((record) => (
          <Card key={record.id} className="p-6 bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm hover:shadow-lg transition-shadow">
            <div className="flex items-start justify-between">
              <div className="flex gap-4 flex-1">
                {/* Status Icon */}
                <div className="mt-1">
                  {getStatusIcon(record.status)}
                </div>

                {/* Main Content */}
                <div className="flex-1 space-y-3">
                  <div className="flex items-start justify-between">
                    <div>
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
                          {record.studyType} Analysis
                        </h3>
                        <Badge className={getStatusColor(record.status)}>
                          {record.status}
                        </Badge>
                        {record.status === 'completed' && (
                          <Badge variant="outline" className="bg-purple-50 text-purple-700 dark:bg-purple-900 dark:text-purple-200">
                            {record.confidence}% Confidence
                          </Badge>
                        )}
                      </div>

                      <div className="flex items-center gap-4 text-sm text-slate-600 dark:text-slate-400">
                        <span className="flex items-center gap-1">
                          <FileText className="h-4 w-4" />
                          {record.id}
                        </span>
                        <span className="flex items-center gap-1">
                          <Calendar className="h-4 w-4" />
                          Patient: {record.patientId}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="h-4 w-4" />
                          {formatDate(record.timestamp)}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Findings */}
                  <div className="bg-slate-50 dark:bg-slate-900/50 rounded-lg p-4">
                    <div className="flex items-start gap-2">
                      <AlertCircle className="h-4 w-4 text-slate-500 mt-0.5 flex-shrink-0" />
                      <div className="flex-1">
                        <p className="text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
                          Cancer Type: {record.cancerType}
                        </p>
                        <p className="text-sm text-slate-600 dark:text-slate-400">
                          {record.findings}
                        </p>
                      </div>
                    </div>
                  </div>

                  {/* Model Version */}
                  <div className="text-xs text-slate-500 dark:text-slate-500">
                    Model: {record.modelVersion}
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-2 ml-4">
                <Button variant="outline" size="sm">
                  <Eye className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="sm">
                  <Download className="h-4 w-4" />
                </Button>
                <Button variant="outline" size="sm" className="text-red-600 hover:text-red-700">
                  <Trash2 className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </Card>
        ))}

        {filteredHistory.length === 0 && (
          <Card className="p-12 text-center bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
            <AlertCircle className="h-12 w-12 text-slate-400 mx-auto mb-4" />
            <p className="text-slate-600 dark:text-slate-400">
              No analysis records found matching your criteria
            </p>
          </Card>
        )}
      </div>

      {/* Stats Summary */}
      <Card className="mt-6 p-6 bg-gradient-to-r from-purple-500 to-pink-500 text-white">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div>
            <p className="text-purple-100 text-sm mb-1">Total Analyses</p>
            <p className="text-3xl font-bold">{analysisHistory.length}</p>
          </div>
          <div>
            <p className="text-purple-100 text-sm mb-1">Completed</p>
            <p className="text-3xl font-bold">
              {analysisHistory.filter(r => r.status === 'completed').length}
            </p>
          </div>
          <div>
            <p className="text-purple-100 text-sm mb-1">Average Confidence</p>
            <p className="text-3xl font-bold">
              {(analysisHistory
                .filter(r => r.status === 'completed')
                .reduce((sum, r) => sum + r.confidence, 0) /
                analysisHistory.filter(r => r.status === 'completed').length
              ).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-purple-100 text-sm mb-1">Success Rate</p>
            <p className="text-3xl font-bold">
              {((analysisHistory.filter(r => r.status === 'completed').length /
                analysisHistory.length) * 100).toFixed(1)}%
            </p>
          </div>
        </div>
      </Card>
    </div>
  )
}
