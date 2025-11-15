'use client'

import { FileText, TrendingUp, Users, Calendar, Download, ExternalLink } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'

export default function PublicationsPage() {
  const publications = [
    {
      title: 'Deep Learning for Lung Cancer Detection in CT Scans',
      authors: 'Smith J., Chen L., Rodriguez M.',
      journal: 'Nature Medicine',
      year: 2024,
      citations: 234,
      impact: 'High Impact',
      status: 'Published',
      doi: '10.1038/s41591-024-01234-5'
    },
    {
      title: 'Multimodal AI for Breast Cancer Diagnosis',
      authors: 'Anderson K., Patel R., Williams S.',
      journal: 'The Lancet Oncology',
      year: 2024,
      citations: 156,
      impact: 'High Impact',
      status: 'Published',
      doi: '10.1016/S1470-2045(24)00123-4'
    },
    {
      title: 'Automated Pathology Image Analysis Using Vision Transformers',
      authors: 'Zhang W., Kumar A., Brown T.',
      journal: 'JAMA Oncology',
      year: 2023,
      citations: 89,
      impact: 'Medium Impact',
      status: 'Published',
      doi: '10.1001/jamaoncol.2023.1234'
    },
    {
      title: 'Real-time Cancer Detection in Medical Imaging',
      authors: 'Lee H., Martinez D., Thompson J.',
      journal: 'Radiology',
      year: 2023,
      citations: 67,
      impact: 'Medium Impact',
      status: 'Published',
      doi: '10.1148/radiol.2023230123'
    },
    {
      title: 'AI-Assisted Diagnosis: A Comprehensive Review',
      authors: 'Johnson M., Garcia P., Wilson R.',
      journal: 'Nature Reviews Clinical Oncology',
      year: 2024,
      citations: 12,
      impact: 'High Impact',
      status: 'In Press',
      doi: 'Pending'
    }
  ]

  const stats = [
    { label: 'Total Publications', value: '127', icon: FileText, color: 'text-blue-600' },
    { label: 'Total Citations', value: '3,456', icon: TrendingUp, color: 'text-purple-600' },
    { label: 'Collaborators', value: '89', icon: Users, color: 'text-green-600' },
    { label: 'This Year', value: '23', icon: Calendar, color: 'text-orange-600' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-purple-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900 p-8">
      <div className="max-w-7xl mx-auto space-y-8">
        {/* Header */}
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent mb-2">
            Research Publications
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Published research from the Aurelius Medical AI platform
          </p>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {stats.map((stat, idx) => (
            <Card key={idx}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-sm font-medium text-slate-600">
                    {stat.label}
                  </CardTitle>
                  <stat.icon className={`w-5 h-5 ${stat.color}`} />
                </div>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{stat.value}</div>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Publications List */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5 text-blue-600" />
              Recent Publications
            </CardTitle>
            <CardDescription>
              Latest research papers and studies from our platform
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {publications.map((pub, idx) => (
                <div key={idx} className="border-l-4 border-l-blue-500 pl-4 py-3 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <h3 className="font-semibold text-lg mb-2">{pub.title}</h3>
                      <p className="text-sm text-slate-600 dark:text-slate-400 mb-2">
                        {pub.authors}
                      </p>
                      <div className="flex items-center gap-4 text-sm">
                        <span className="text-slate-600">
                          <strong>{pub.journal}</strong>, {pub.year}
                        </span>
                        <Badge variant="secondary">{pub.status}</Badge>
                        <Badge
                          variant={pub.impact === 'High Impact' ? 'default' : 'secondary'}
                          className={pub.impact === 'High Impact' ? 'bg-purple-600' : ''}
                        >
                          {pub.impact}
                        </Badge>
                        <span className="text-slate-500">
                          {pub.citations} citations
                        </span>
                      </div>
                      {pub.doi !== 'Pending' && (
                        <p className="text-xs text-slate-500 mt-2">
                          DOI: {pub.doi}
                        </p>
                      )}
                    </div>
                    <div className="flex gap-2">
                      <Button size="sm" variant="outline">
                        <Download className="w-4 h-4 mr-1" />
                        PDF
                      </Button>
                      <Button size="sm" variant="outline">
                        <ExternalLink className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Research Areas */}
        <Card>
          <CardHeader>
            <CardTitle>Research Focus Areas</CardTitle>
            <CardDescription>Main research domains and contributions</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {[
                { area: 'Lung Cancer Detection', papers: 34, percentage: 27 },
                { area: 'Breast Cancer Screening', papers: 28, percentage: 22 },
                { area: 'AI-Assisted Pathology', papers: 25, percentage: 20 },
                { area: 'Multimodal Imaging', papers: 22, percentage: 17 },
                { area: 'Clinical Decision Support', papers: 18, percentage: 14 }
              ].map((item, idx) => (
                <div key={idx} className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="font-medium">{item.area}</span>
                    <span className="text-slate-600">{item.papers} papers ({item.percentage}%)</span>
                  </div>
                  <div className="h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-purple-500"
                      style={{ width: `${item.percentage * 4}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
