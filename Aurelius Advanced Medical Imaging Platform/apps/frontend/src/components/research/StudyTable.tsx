'use client'

import { useState } from 'react'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/table'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Eye,
  Download,
  MoreHorizontal,
  Search,
  Filter,
  ArrowUpDown,
  CheckCircle,
  Clock,
  AlertCircle,
  Brain
} from 'lucide-react'
import { format } from 'date-fns'

interface Study {
  id: string
  patientId: string
  patientName: string
  modality: string
  studyDate: Date
  description: string
  seriesCount: number
  imagesCount: number
  status: 'completed' | 'processing' | 'pending' | 'error'
  aiAnalysis?: boolean
  annotations?: number
}

const mockStudies: Study[] = [
  {
    id: 'STD-2024-001',
    patientId: 'PAT-12345',
    patientName: 'John Doe',
    modality: 'CT',
    studyDate: new Date('2024-11-15'),
    description: 'Chest CT with Contrast',
    seriesCount: 12,
    imagesCount: 512,
    status: 'completed',
    aiAnalysis: true,
    annotations: 3
  },
  {
    id: 'STD-2024-002',
    patientId: 'PAT-12346',
    patientName: 'Jane Smith',
    modality: 'MRI',
    studyDate: new Date('2024-11-15'),
    description: 'Brain MRI',
    seriesCount: 8,
    imagesCount: 256,
    status: 'processing',
    aiAnalysis: false,
    annotations: 0
  },
  {
    id: 'STD-2024-003',
    patientId: 'PAT-12347',
    patientName: 'Bob Johnson',
    modality: 'X-Ray',
    studyDate: new Date('2024-11-14'),
    description: 'Chest X-Ray PA/Lateral',
    seriesCount: 2,
    imagesCount: 2,
    status: 'completed',
    aiAnalysis: true,
    annotations: 1
  },
]

export function StudyTable() {
  const [studies] = useState<Study[]>(mockStudies)
  const [searchTerm, setSearchTerm] = useState('')
  const [sortField, setSortField] = useState<keyof Study>('studyDate')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc')

  const getStatusIcon = (status: Study['status']) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="h-4 w-4 text-green-600" />
      case 'processing':
        return <Clock className="h-4 w-4 text-yellow-600 animate-spin" />
      case 'pending':
        return <Clock className="h-4 w-4 text-blue-600" />
      case 'error':
        return <AlertCircle className="h-4 w-4 text-red-600" />
    }
  }

  const getStatusBadge = (status: Study['status']) => {
    const variants = {
      completed: 'success',
      processing: 'warning',
      pending: 'info',
      error: 'destructive'
    }
    return (
      <Badge variant={variants[status] as any}>
        {status.charAt(0).toUpperCase() + status.slice(1)}
      </Badge>
    )
  }

  const filteredStudies = studies.filter(study =>
    study.patientName.toLowerCase().includes(searchTerm.toLowerCase()) ||
    study.patientId.toLowerCase().includes(searchTerm.toLowerCase()) ||
    study.modality.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="space-y-4">
      {/* Search and Filters */}
      <div className="flex items-center gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
          <Input
            placeholder="Search studies by patient, ID, or modality..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
          />
        </div>
        <Button variant="outline" size="sm">
          <Filter className="h-4 w-4 mr-2" />
          Filters
        </Button>
        <Button variant="outline" size="sm">
          <ArrowUpDown className="h-4 w-4 mr-2" />
          Sort
        </Button>
      </div>

      {/* Table */}
      <div className="rounded-lg border bg-white dark:bg-slate-800 shadow-sm">
        <Table>
          <TableHeader>
            <TableRow className="bg-slate-50 dark:bg-slate-900">
              <TableHead>Study ID</TableHead>
              <TableHead>Patient</TableHead>
              <TableHead>Modality</TableHead>
              <TableHead>Date</TableHead>
              <TableHead>Description</TableHead>
              <TableHead className="text-center">Series</TableHead>
              <TableHead className="text-center">Images</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>AI</TableHead>
              <TableHead className="text-right">Actions</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredStudies.map((study) => (
              <TableRow key={study.id} className="hover:bg-slate-50 dark:hover:bg-slate-700/50">
                <TableCell className="font-mono text-sm">{study.id}</TableCell>
                <TableCell>
                  <div>
                    <div className="font-medium">{study.patientName}</div>
                    <div className="text-xs text-slate-500">{study.patientId}</div>
                  </div>
                </TableCell>
                <TableCell>
                  <Badge variant="outline">{study.modality}</Badge>
                </TableCell>
                <TableCell className="text-sm">
                  {format(study.studyDate, 'MMM dd, yyyy')}
                </TableCell>
                <TableCell className="max-w-xs truncate text-sm">
                  {study.description}
                </TableCell>
                <TableCell className="text-center">{study.seriesCount}</TableCell>
                <TableCell className="text-center">{study.imagesCount}</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    {getStatusIcon(study.status)}
                    {getStatusBadge(study.status)}
                  </div>
                </TableCell>
                <TableCell>
                  {study.aiAnalysis ? (
                    <div className="flex items-center gap-1">
                      <Brain className="h-4 w-4 text-purple-600" />
                      <span className="text-xs text-purple-600">Analyzed</span>
                    </div>
                  ) : (
                    <span className="text-xs text-slate-400">-</span>
                  )}
                </TableCell>
                <TableCell className="text-right">
                  <div className="flex items-center justify-end gap-2">
                    <Button variant="ghost" size="sm">
                      <Eye className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="sm">
                      <Download className="h-4 w-4" />
                    </Button>
                    <Button variant="ghost" size="sm">
                      <MoreHorizontal className="h-4 w-4" />
                    </Button>
                  </div>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between text-sm text-slate-600 dark:text-slate-400">
        <div>Showing {filteredStudies.length} of {studies.length} studies</div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" disabled>Previous</Button>
          <Button variant="outline" size="sm">Next</Button>
        </div>
      </div>
    </div>
  )
}
